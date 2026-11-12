"""Assemble an offline Windows folder from verified upstream distributions.

This script does not execute Windows code. Use diagnostics.cmd on Windows afterwards.
Inputs: --assets has python-windows.zip, ffmpeg-windows.zip, whisper-cli.exe,
ClipnadoStudioLite.exe and ggml-base.bin; --wheels contains Windows/pure Python wheels.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def unpack(archive, folder):
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            target = (folder / item.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("Unsafe archive path")
        package.extractall(folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--wheels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha(args.assets / "ffmpeg-windows.zip") != "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec":
        raise ValueError("FFmpeg upstream checksum mismatch")
    root = Path(__file__).resolve().parent.parent
    dest = args.output.resolve()
    if dest.exists():
        raise ValueError("Output folder already exists; choose a new one")
    dest.mkdir(parents=True)
    runtime = dest / "runtime"
    runtime.mkdir()
    unpack(args.assets / "python-windows.zip", runtime)
    pth = next(runtime.glob("python*._pth"))
    stdlib = next(runtime.glob("python*.zip")).name
    pth.write_text(f"{stdlib}\n.\n..\nLib/site-packages\nimport site\n", encoding="ascii")
    packages = runtime / "Lib" / "site-packages"
    packages.mkdir(parents=True)
    skipped = ("secretstorage-", "jeepney-", "cryptography-", "cffi-", "pycparser-")
    wheel_manifest = []
    for wheel in sorted(args.wheels.glob("*.whl")):
        if wheel.name.lower().startswith(skipped):
            continue
        if not (wheel.name.endswith("win_amd64.whl") or wheel.name.endswith("none-any.whl")):
            raise ValueError("Non-Windows wheel: " + wheel.name)
        unpack(wheel, packages)
        wheel_manifest.append({"name": wheel.name, "sha256": sha(wheel)})
    # Namespace packages in multiple jaraco wheels naturally share the same folder.
    for folder in ("clipnado", "scripts", "demo", "licenses", "qa"):
        source = root / folder
        if source.exists():
            shutil.copytree(source, dest / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run.py", "start.cmd", "diagnostics.cmd", "README.md", "QA_REPORT.md", "THIRD_PARTY.md", "LICENSE"):
        if (root / name).exists():
            shutil.copy2(root / name, dest / name)
    tools = dest / "tools"
    tools.mkdir()
    with zipfile.ZipFile(args.assets / "ffmpeg-windows.zip") as package:
        for filename in ("ffmpeg.exe", "ffprobe.exe"):
            source = next(n for n in package.namelist() if n.endswith("/bin/" + filename))
            (tools / filename).write_bytes(package.read(source))
        (tools / "FFmpeg-LICENSE.txt").write_bytes(package.read(next(n for n in package.namelist() if n.endswith("/LICENSE"))))
        (tools / "FFmpeg-README.txt").write_bytes(package.read(next(n for n in package.namelist() if n.endswith("/README.txt"))))
    shutil.copy2(args.assets / "whisper-cli.exe", tools / "whisper-cli.exe")
    shutil.copy2(args.assets / "ClipnadoStudioLite.exe", dest / "ClipnadoStudioLite.exe")
    (dest / "models").mkdir()
    model = args.assets / "ggml-base.bin"
    if sha(model) != "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe":
        raise ValueError("Model checksum mismatch")
    shutil.copy2(model, dest / "models" / "ggml-base.bin")
    manifest = {"format": 1, "python": "3.13.9", "whisper_commit": "4979e04f5dcaccb36057e059bbaed8a2f5288315",
                "ffmpeg": "8.1.2-essentials_build", "cpu": "AVX; AVX2/BMI2/FMA/F16C/AVX512 disabled for whisper.cpp",
                "windows_execution_verified": False, "wheels": wheel_manifest,
                "files": {str(p.relative_to(dest)).replace("\\", "/"): sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Portable folder:", dest)


if __name__ == "__main__":
    main()
