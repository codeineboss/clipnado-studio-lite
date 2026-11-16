"""Isolated CPU-only vision worker. Runs locally; accepts one JSON job file.

Application logic, MFCC frontend, association and crop planning are our own.
YuNet and Light-ASD are bundled, pinned MIT pretrained models. No PyTorch,
model downloader, custom ONNX operators, identity recognition or API is used.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import wave

from .projects import atomic_json
from .reframe import ANALYSIS_FPS, CropPlan, direct_camera, model_bytes


def notify(percent, text):
    print(json.dumps({"percent": int(percent), "text": text}), flush=True)


def mfcc(signal, sample_rate=16000):
    """13 cepstra, 26 mel filters, 25ms/10ms, pre-emphasis .97, lifter 22.

    Raw signed-16-bit amplitude, rectangular window and log-energy coefficient
    match the pretrained model's python_speech_features default frontend.
    """
    import numpy as np

    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        signal = np.zeros(1, dtype=np.float64)
    emphasized = np.r_[signal[0], signal[1:] - 0.97 * signal[:-1]]
    length, step = int(sample_rate * 0.025 + 0.5), int(sample_rate * 0.010 + 0.5)
    count = 1 + max(0, int(np.ceil((len(emphasized) - length) / step)))
    padded = np.pad(
        emphasized, (0, max(0, (count - 1) * step + length - len(emphasized)))
    )
    frames = np.lib.stride_tricks.sliding_window_view(padded, length)[::step][:count]
    power = np.abs(np.fft.rfft(frames, n=512)) ** 2 / 512
    energy = np.maximum(power.sum(axis=1), np.finfo(float).eps)
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    hz = 700 * (10 ** (np.linspace(mel(0), mel(sample_rate / 2), 28) / 2595) - 1)
    bins = np.floor((512 + 1) * hz / sample_rate).astype(int)
    bank = np.zeros((26, 257))
    for i in range(26):
        left, peak, right = bins[i : i + 3]
        for j in range(left, peak):
            bank[i, j] = (j - left) / max(1, peak - left)
        for j in range(peak, right):
            bank[i, j] = (right - j) / max(1, right - peak)
    logmel = np.log(np.maximum(power @ bank.T, np.finfo(float).eps))
    dct = np.cos(
        np.pi * np.arange(13)[:, None] * (np.arange(26) + 0.5)[None, :] / 26
    ) * np.sqrt(2 / 26)
    dct[0] /= np.sqrt(2)
    features = (logmel @ dct.T) * (1 + 11 * np.sin(np.pi * np.arange(13) / 22))
    features[:, 0] = np.log(energy)
    return features.astype(np.float32)


def read_frame(path):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Не удалось прочитать кадр для автокадрирования.")
    return image


def overlap(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return area / max(1, a[2] * a[3] + b[2] * b[3] - area)


def detect_tracks(files, threads):
    import cv2
    import numpy as np

    cv2.setNumThreads(max(1, min(4, threads)))
    detector = cv2.FaceDetectorYN.create(
        "onnx",
        np.frombuffer(model_bytes("yunet-2023mar.onnx"), np.uint8),
        np.empty(0, np.uint8),
        (320, 320),
        0.75,
        0.3,
        5000,
        cv2.dnn.DNN_BACKEND_OPENCV,
        cv2.dnn.DNN_TARGET_CPU,
    )
    scenes = []
    tracks = []
    active = []
    scene = 0
    previous = None
    previous_hist = None
    width = height = 0
    for i, path in enumerate(files):
        image = read_frame(path)
        height, width = image.shape[:2]
        tiny = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (48, 32)).astype(
            np.float32
        )
        hsv = cv2.cvtColor(cv2.resize(image, (64, 36)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = (
            previous is not None
            and float(np.abs(tiny - previous).mean()) > 32
            and cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.35
        )
        if cut:
            scene += 1
            active = []
        scenes.append(scene)
        previous, previous_hist = tiny, hist
        if i % 5 and not cut:
            continue
        detector.setInputSize((width, height))
        _, detected = detector.detect(image)
        boxes = []
        for row in [] if detected is None else detected:
            x, y, w, h = map(float, row[:4])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            box = (x0, y0, x1 - x0, y1 - y0)
            if min(box[2:]) >= 12 and 0.35 <= box[2] / box[3] <= 1.8:
                boxes.append(box)
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:4]
        active = [t for t in active if i - t["last"] <= 12]
        pairs = []
        for index, box in enumerate(boxes):
            for track in active:
                old = track["detections"][-1][1]
                distance = abs((box[0] + box[2] / 2) - (old[0] + old[2] / 2)) + abs(
                    (box[1] + box[3] / 2) - (old[1] + old[3] / 2)
                )
                distance /= max(old[2], old[3], box[2], box[3])
                shared = overlap(box, old)
                if shared > 0.15 or distance < 1.1:
                    pairs.append((shared - 0.1 * distance, index, track["id"]))
        used_boxes = set()
        used_tracks = set()
        matched = {}
        for _, index, tid in sorted(pairs, reverse=True):
            if index not in used_boxes and tid not in used_tracks:
                matched[index] = tid
                used_boxes.add(index)
                used_tracks.add(tid)
        for index, box in enumerate(boxes):
            if index in matched:
                track = tracks[matched[index]]
                track["detections"].append((i, box))
                track["last"] = i
            else:
                track = {
                    "id": len(tracks),
                    "scene": scene,
                    "last": i,
                    "detections": [(i, box)],
                }
                tracks.append(track)
                active.append(track)
        if i % 25 == 0:
            notify(i / max(1, len(files)) * 45, "Автокадр: поиск лиц и смен планов…")
    states = [{"scene": s, "faces": {}} for s in scenes]
    expanded = []
    for track in tracks:
        detections = track["detections"]
        positions = {}
        for (left, a), (right, b) in zip(detections, detections[1:]):
            if right - left > 12:
                continue
            for i in range(left, right + 1):
                positions[i] = tuple(
                    float(x + (y - x) * (i - left) / max(1, right - left))
                    for x, y in zip(a, b)
                )
        for i, box in detections:
            positions[i] = box
        for i in range(
            max(0, detections[0][0] - 2), min(len(files), detections[-1][0] + 5)
        ):
            if scenes[i] != track["scene"]:
                continue
            if i < detections[0][0]:
                positions[i] = detections[0][1]
            if i > detections[-1][0]:
                positions[i] = detections[-1][1]
        if not positions:
            continue
        indexes = sorted(positions)
        raw = np.array([positions[i] for i in indexes], dtype=np.float64)
        for j, i in enumerate(indexes):
            smooth = tuple(map(float, np.median(raw[max(0, j - 4) : j + 5], axis=0)))
            positions[i] = smooth
            states[i]["faces"][track["id"]] = smooth
        track["positions"] = positions
        expanded.append(track)
    return states, expanded, width, height


def mouth_patch(image, box):
    import cv2

    x, y, w, h = box
    half = max(w, h) / 2
    cx, cy = x + w / 2, y + h / 2
    pad = int(half * 1.8) + 4
    image = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110)
    )
    left, right = int(cx + pad - half * 1.4), int(cx + pad + half * 1.4)
    top, bottom = int(cy + pad - half), int(cy + pad + half * 1.8)
    face = cv2.resize(
        image[top:bottom, left:right], (224, 224), interpolation=cv2.INTER_LINEAR
    )
    return cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)[56:168, 56:168]


class SpeakerModel:
    def __init__(self, threads):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(8, threads))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_bytes("light-asd.onnx"), opts, providers=["CPUExecutionProvider"]
        )

    def probabilities(self, audio, visual):
        import numpy as np

        logits = self.session.run(
            ["logits"],
            {
                "audio": np.ascontiguousarray(audio[None], dtype=np.float32),
                "visual": np.ascontiguousarray(visual[None], dtype=np.float32),
            },
        )[0][0]
        values = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return values[:, 1] / values.sum(axis=-1)


def score_tracks(files, states, tracks, audio_path, threads):
    import numpy as np

    if not audio_path or not any(len(s["faces"]) > 1 for s in states):
        return {}
    with wave.open(str(audio_path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getframerate() != 16000
            or stream.getsampwidth() != 2
        ):
            raise ValueError("Неверный формат звука для автокадра")
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    features = mfcc(audio)
    needed = len(files) * 4
    if len(features) < needed:
        features = np.pad(features, ((0, needed - len(features)), (0, 0)), mode="edge")
    model = SpeakerModel(threads)
    relevant = [
        t for t in tracks if any(len(states[i]["faces"]) > 1 for i in t["positions"])
    ]
    scores = {}
    total = sum(len(t["positions"]) for t in relevant)
    done = 0
    for track in relevant:
        first, last = min(track["positions"]), max(track["positions"])
        sums = np.zeros(last - first + 1)
        counts = np.zeros(last - first + 1)
        # Four-second contexts with an overlap keep temporal edge effects small.
        for start in range(first, last + 1, 80):
            stop = min(last + 1, start + 100)
            if stop <= start:
                continue
            if not any(len(states[i]["faces"]) > 1 for i in range(start, stop)):
                continue
            visual = []
            for i in range(start, stop):
                box = track["positions"].get(i)
                if box is None:
                    box = track["positions"][
                        min(track["positions"], key=lambda k: abs(k - i))
                    ]
                visual.append(mouth_patch(read_frame(files[i]), box))
            probabilities = model.probabilities(
                features[start * 4 : stop * 4], np.asarray(visual)
            )
            sums[start - first : stop - first] += probabilities
            counts[start - first : stop - first] += 1
            notify(
                45 + min(50, (done + (stop - first)) / max(1, total) * 50),
                "Автокадр: определение говорящего по звуку и движению рта…",
            )
        result = np.divide(sums, np.maximum(counts, 1))
        # Offline symmetric smoothing avoids reacting to one uncertain frame.
        result = np.convolve(
            np.pad(result, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid"
        )
        scores[track["id"]] = {
            i: float(result[i - first])
            for i in track["positions"]
            if counts[i - first] > 0
        }
        done += len(track["positions"])
    return scores


def analyze(job):
    # Load the existing Qt runtime's MSVC DLLs before native Python extensions.
    # This also works in a portable install without a system VC++ installer.
    from PySide6 import QtCore  # noqa: F401

    files = sorted(Path(job["frames"]).glob("*.jpg"))
    duration = job["candidate"]["end_ms"] - job["candidate"]["start_ms"]
    if not files:
        return CropPlan(
            duration,
            [[0, 0.5, 0.5]],
            {"max_faces": 0, "switches": 0, "frames": 0},
            ["Недостаточно кадров для анализа: использован центр."],
        )
    states, tracks, width, height = detect_tracks(files, int(job["tools"]["threads"]))
    scores = score_tracks(
        files, states, tracks, job.get("audio"), int(job["tools"]["threads"])
    )
    notify(97, "Автокадр: сглаживание камеры и переключений…")
    plan = direct_camera(states, scores, width, height, duration)
    if not job.get("audio") and plan.info["max_faces"] > 1:
        plan.warnings.append(
            "У видео нет звука: автокадр удерживает заметное лицо; определение говорящего недоступно."
        )
    return plan


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        plan = analyze(job)
        atomic_json(Path(job["result"]), plan.to_dict())
    except Exception as exc:
        atomic_json(Path(job["result"]), {"error": str(exc)})
        print(json.dumps({"error": str(exc)}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
