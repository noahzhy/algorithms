import cv2
from PIL import Image

def is_color_image(url):
    im=Image.open(url)
    pix=im.convert('RGB')
    width=im.size[0]
    height=im.size[1]
    oimage_color_type= False
    is_color=[]
    for x in range(width):
        for y in range(height):
            r,g,b=pix.getpixel((x,y))
            r=int(r)
            g=int(g)
            b=int(b)
            if (r==g) and (g==b):
                pass
            else:
                oimage_color_type= True
    return oimage_color_type

# print(is_color_image("F:\chinese_cleaned\高晓松\高晓松_6.jpg"))