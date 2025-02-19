import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':
    install("git+https://github.com/warmshao/WiLoR-mini")