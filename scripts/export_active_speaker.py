"""Build-only, independent ONNX graph for the published Light-ASD architecture.

Input: the upstream MIT checkpoint at the pinned SHA below. PyTorch is used
only as a safe tensor-checkpoint reader (weights_only=True), never at runtime.
This exporter does not download or execute another application's Python code.
The resulting graph contains standard ONNX operators and is bundled offline.
"""

from pathlib import Path
import argparse
import hashlib
import numpy as np
import onnx
from onnx import TensorProto, helper as h, numpy_helper as nh

CHECKPOINT_SHA = "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317"


class Graph:
    def __init__(self, weights):
        self.weights, self.nodes, self.initializers = weights, [], []
        self.count = 0

    def constant(self, value, name=None):
        name = name or f"constant_{len(self.initializers)}"
        self.initializers.append(nh.from_array(np.asarray(value), name))
        return name

    def op(self, kind, *inputs, **attrs):
        self.count += 1
        name = f"{kind}_{self.count}"
        self.nodes.append(h.make_node(kind, list(inputs), [name], name=name, **attrs))
        return name

    def tensor(self, name):
        return self.constant(self.weights[name].astype(np.float32), name)

    def norm_conv(self, x, prefix, conv, bn, strides=None):
        w = self.weights[f"{prefix}.{conv}.weight"]
        kernel = list(w.shape[2:])
        pads = [k // 2 for k in kernel]
        x = self.op(
            "Conv",
            x,
            self.tensor(f"{prefix}.{conv}.weight"),
            kernel_shape=kernel,
            strides=strides or [1] * len(kernel),
            pads=pads + pads,
        )
        x = self.op(
            "BatchNormalization",
            x,
            *[
                self.tensor(f"{prefix}.{bn}.{part}")
                for part in ("weight", "bias", "running_mean", "running_var")
            ],
            epsilon=0.001,
        )
        return self.op("Relu", x)

    def block(self, x, prefix, visual, down=False):
        branches = []
        for k in (3, 5):
            spatial = "s" if visual else "m"
            stride = [1, 2, 2] if down else None
            part = self.norm_conv(
                x, prefix, f"{spatial}_{k}", f"bn_{spatial}_{k}", stride
            )
            branches.append(self.norm_conv(part, prefix, f"t_{k}", f"bn_t_{k}"))
        return self.norm_conv(self.op("Add", *branches), prefix, "last", "bn_last")

    def gelu(self, x):
        root2 = self.constant(np.float32(np.sqrt(2)))
        one, half = self.constant(np.float32(1)), self.constant(np.float32(0.5))
        return self.op(
            "Mul",
            self.op("Mul", x, half),
            self.op("Add", one, self.op("Erf", self.op("Div", x, root2))),
        )

    def reverse(self, x):
        return self.op(
            "Slice",
            x,
            self.constant(np.array([-1], np.int64)),
            self.constant(np.array([np.iinfo(np.int64).min], np.int64)),
            self.constant(np.array([1], np.int64)),
            self.constant(np.array([-1], np.int64)),
        )

    def gru(self, x, prefix):
        # PyTorch r,z,n gates -> ONNX z,r,h gates; linear_before_reset matches
        # PyTorch's reset gate applied to the recurrent candidate contribution.
        order = np.r_[128:256, 0:128, 256:384]
        w = self.constant(
            self.weights[prefix + ".weight_ih_l0"][order][None].astype(np.float32)
        )
        r = self.constant(
            self.weights[prefix + ".weight_hh_l0"][order][None].astype(np.float32)
        )
        b = self.constant(
            np.r_[
                self.weights[prefix + ".bias_ih_l0"][order],
                self.weights[prefix + ".bias_hh_l0"][order],
            ][None].astype(np.float32)
        )
        x = self.op("Transpose", x, perm=[1, 0, 2])
        x = self.op("GRU", x, w, r, b, hidden_size=128, linear_before_reset=1)
        x = self.op("Squeeze", x, self.constant(np.array([1], np.int64)))
        return self.op("Transpose", x, perm=[1, 0, 2])

    def export(self, output):
        v = self.op("Unsqueeze", "visual", self.constant(np.array([1], np.int64)))
        v = self.op(
            "Div",
            self.op(
                "Sub",
                self.op("Div", v, self.constant(np.float32(255))),
                self.constant(np.float32(0.4161)),
            ),
            self.constant(np.float32(0.1688)),
        )
        a = self.op(
            "Transpose",
            self.op("Unsqueeze", "audio", self.constant(np.array([1], np.int64))),
            perm=[0, 1, 3, 2],
        )
        for i in range(1, 4):
            v = self.block(v, f"model.visualEncoder.block{i}", True, down=i == 1)
            a = self.block(a, f"model.audioEncoder.block{i}", False)
            if i < 3:
                v = self.op(
                    "MaxPool",
                    v,
                    kernel_shape=[1, 3, 3],
                    strides=[1, 2, 2],
                    pads=[0, 1, 1, 0, 1, 1],
                )
                a = self.op(
                    "MaxPool", a, kernel_shape=[1, 3], strides=[1, 2], pads=[0, 1, 0, 1]
                )
        v = self.op(
            "Transpose",
            self.op("ReduceMax", v, axes=[3, 4], keepdims=0),
            perm=[0, 2, 1],
        )
        a = self.op(
            "Transpose", self.op("ReduceMean", a, axes=[2], keepdims=0), perm=[0, 2, 1]
        )
        x = self.gelu(self.gru(self.op("Add", a, v), "model.GRU.gru_forward"))
        x = self.gelu(self.reverse(self.gru(self.reverse(x), "model.GRU.gru_backward")))
        x = self.op(
            "Add",
            self.op(
                "MatMul",
                x,
                self.constant(self.weights["lossAV.FC.weight"].T.astype(np.float32)),
            ),
            self.tensor("lossAV.FC.bias"),
        )
        self.nodes.append(h.make_node("Identity", [x], ["logits"]))
        graph = h.make_graph(
            self.nodes,
            "Light-ASD",
            [
                h.make_tensor_value_info(
                    "audio", TensorProto.FLOAT, [1, "audio_steps", 13]
                ),
                h.make_tensor_value_info(
                    "visual", TensorProto.FLOAT, [1, "frames", 112, 112]
                ),
            ],
            [h.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "frames", 2])],
            self.initializers,
        )
        model = h.make_model(
            graph,
            producer_name="Clipnado independent Light-ASD export",
            opset_imports=[h.make_opsetid("", 13)],
        )
        model.ir_version = 8
        model.doc_string = "Light-ASD, Liao et al. CVPR 2023; MIT (c) 2023 Liao Junhua. Original architecture and pretrained parameters; independently constructed ONNX graph."
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError("Unverified Light-ASD checkpoint")
    import torch

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = {k: v.detach().cpu().numpy() for k, v in state.items()}
    Graph(weights).export(args.output)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest(), args.output)


if __name__ == "__main__":
    main()
