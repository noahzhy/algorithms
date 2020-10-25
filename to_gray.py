import cv2
import glob
from PIL import Image

def to_gray(img_file):
    Image.open(img_file).save(img_file)


if __name__ == "__main__":
    for i in glob.glob("FRTAM/data/dataset/*/*/*.jpg"):
        to_gray(i)
