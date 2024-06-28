import os
import subprocess

def build():
    command = [
        "python3",
        "-m",
        "PyInstaller",
        "--onefile",
        "UniversalTorchUpscaler.py",
        "--collect-all",
        "torch",
        "--collect-all",
        "torchvision",
        "--collect-all",
        "spandrel",
        "--collect-all",
        "pnnx",
        "--collect-all",
        "onnx",
        "--collect-all",
        "opencv-python",
        "--collect-all",
        "numpy",
        "--collect-all",
        "pillow"

    ]
    subprocess.run(command)

if __name__ == '__main__':
    build()