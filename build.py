import os
import subprocess


def build():
    command = [
        "python3",
        "-m",
        "PyInstaller",
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
        "opencv-python",
    ]
    subprocess.run(command)


if __name__ == "__main__":
    build()
