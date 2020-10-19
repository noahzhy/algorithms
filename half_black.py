from PIL import Image
import os
import glob


def half_black(path):
    image = Image.open(path)
    new_image = Image.new('L', (96, 96), (0))
    new_image.paste(image.crop((0,0,96,48)))
    # print(new_image)
    new_image.save(path)


if __name__ == "__main__":
    half_black(r"F:\only_faces\img\anyaping_000.jpg")
