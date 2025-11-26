import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':
    install("panda3d")
    install("numpy")
    install("scipy")
    install("tqdm")
    install("opencv-python")
    install("opencv-contrib-python")
    install("rtree")
    install("open3d")
    install("shapely")
    install("networkx")
    install("pyserial")
    install("matplotlib")