import os
import cv2
import dlib
import glob
import random
import imutils
import unicodedata
import numpy as np
# import matplotlib.pyplot as plt
from shutil import copyfile, rmtree
from face_main import AutoCrop as ac
# from numpy import *
from itertools import chain
from PIL import Image
from imutils.face_utils import FaceAligner
from imutils import face_utils
# from pypinyin import lazy_pinyin, Style, pinyin


predictor_path = "models/shape_predictor_5_face_landmarks.dat"


def is_gray_image(url):
    im = Image.open(url)
    pix = im.convert('RGB')
    width = im.size[0]
    height = im.size[1]
    oimage_color_type = True
    is_color = []
    for x in range(width):
        for y in range(height):
            r, g, b = pix.getpixel((x, y))
            r = int(r)
            g = int(g)
            b = int(b)
            if (r == g) and (g == b):
                pass
            else:
                oimage_color_type = False
    return oimage_color_type


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def get_fileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024)
    return round(fsize, 2)


def detected_face(ac, face_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    image = cv_imread(face_path)
    # image = imutils.resize(image, width=800) # added but not runned
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        return None

    # cv2.imshow("Input", image)
    rects = detector(gray)

    if len(rects) == 0:
        return None

    for rect in rects:
        faceAligned = fa.align(image, gray, rect)
        fs = detector(faceAligned)
        if len(fs) == 0:
            continue
        (x, y, w, h) = face_utils.rect_to_bb(fs[0])
        image = ac.zoom_image(faceAligned, (x, y, w, h))
        return image
        # return ac.crop(faceAligned)


c = ac(96, 96, 10)
names = glob.glob(r"k_faces_test_imgs/*/*.jpg")
names = sorted(names)
# random.shuffle(names)
for i in names:
    cropped = detected_face(c, i)
    if (cropped is None):
        print("no face:", i)
        # os.remove(i)
    else:
        print("write:", i)
        img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).save(i)
        # img.show()
        # # .save(i)
    # break



def get_file_size_full(filePath):
    return os.path.getsize(filePath)


def getImageVar(imgPath):
    image = cv_imread(imgPath)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()

    return imageVar


def crop_to_iv(image):
    if len(image.shape) == 3:
        img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img2gray = image
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()

    return imageVar


def del_less_two():
    names = glob.glob(r"F:\chinese_cleaned\*")
    random.shuffle(names)
    for i in names:
        nums = glob.glob(i+"\*.*")
        if len(nums) < 2:
            # rmtree(i)
            print("remove:", i)
            continue
    # temp = list()
    # for n in nums:
    #     pass
        # t = os.path.basename(n)
        # os.rename(n, os.path.join(r"F:\all_faces", t))
    #     if detected_face(n) == False:
    #         os.remove(n)
    #         print("remove:", n)
    #         continue

        # fs = get_file_size_full(n)
        # if fs in temp:
        #     os.remove(n)
        #     print(n)
        # else:
        #     temp.append(fs)

        # if get_fileSize(n) < 15:
        #     os.remove(n)
        #     print("remove:", n)
        #     continue
    # temp.clear()
    #     if is_gray_image(n):
    #         os.remove(n)
    #         print("remove:", n)


def lpls(path, start="0000147", limit=60, test=True):
    c = ac(96, 96, 10)
    if test:
        # im = cv_imread(path)
        cropped = detected_face(c, path)
        plt.imshow(cropped)
        plt.show()
        iv = crop_to_iv(cropped)
        print(iv)
        return None

    names = glob.glob("C:/temp/CASIA-WebFace-112X96/*/*.jpg")
    names = sorted(names)
    # print(names[0])
    # idx = names.index(r"C:/temp/CASIA-WebFace-112X96\{}\001.jpg".format(start))
    # idx_end = names.index(r"C:/temp/CASIA-WebFace-112X96\{}\001.jpg".format("0100793"))
    # print(idx)
    # quit()

    for i in names:
        cropped = detected_face(c, i)
        if (cropped is None):
            print("no face:", i)
            os.remove(i)
        else:
            iv = round(crop_to_iv(cropped), 2)
            print(i, "\t\t{}".format(iv))
            if iv < limit:
                os.remove(i)
                print("< {}:".format(limit), i, "\t\t{}".format(iv))


# lpls("F:\chinese_cleaned\李炜\李炜_2.jpg")

# names = glob.glob("F:/all_faces/*")
# names = sorted(names)
# count = 0
# start = "Angelababy"
# before = ""

# for i in names:
#     base_name = os.path.basename(i).split("_")[0]
#     count += 1
#     if base_name != start:
#         if count < 2:
#             print(start, "count:", count, "path: {}".format(before))
#             os.remove(before)
#         count = 0
#         start = base_name

#     before = i

# names = glob.glob("F:/all_faces/*")
# names = sorted(names)
# for i in names:
#     pic_name = os.path.basename(i)
#     base_name = pic_name.split("_")[0]
#     try:
#         os.rename(i, os.path.join("F:\chinese_cleaned", base_name, pic_name))
#     except:
#         print("No folder:", base_name)
def only_face():
    c = ac(96, 96)
    names = glob.glob("F:/CHN_faces/chinese_cleaned_src/*/*.jpg")
    names = sorted(names)
    print(len(names))
    for i in names:
        cropped = detected_face(c, i)
        if (cropped is None):
            print("no face:", i)
            os.remove(i)
        else:
            print("write:", i)
            Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).save(i)
        # break


def rename_all():
    names = glob.glob("F:\CHN\chinese_cleaned_src\*")
    names = sorted(names)
    for i in names:
        base_name = os.path.basename(i)
        count = 0
        full_path = os.path.join(i, "*.jpg")
        for j in glob.glob(full_path):
            print(j)
            os.rename(j, "{}/{}_{:03d}.jpg".format(i, base_name, count))
            count += 1


def all_to_grayscale():
    names = glob.glob("F:/CHN_faces/chinese_cleaned_96px_grayscale/*/*.jpg")
    names = sorted(names)
    for i in names:
        print(i)
        Image.open(i).convert("L").save(i)


def to_pinyin(s):
    return ''.join(chain.from_iterable(pinyin(s, style=Style.TONE3)))


def labeled_all():
    count = 0
    names = glob.glob("F:/CHN_faces/chinese_cleaned_src_labeled/*")
    # names = sorted(names)
    out = sorted(names, key=to_pinyin)

    for i in out:
        base_name = os.path.basename(i)
        pinyin = ''.join(lazy_pinyin(base_name))
        os.rename(i, os.path.join("F:/CHN_faces/chinese_cleaned_src_labeled/", "{:04d}_{}".format(count, pinyin)))
        # os.rename(i, pinyin)
        count += 1


def labeled_all_imgs():
    names = glob.glob("F:/CHN_faces/chinese_cleaned_src_labeled/*")
    out = sorted(names)

    for i in out:
        count = 0
        base_name = os.path.basename(i).split('_')[-1]
        print(base_name)
        for j in glob.glob(i+"/*.jpg"):
            os.rename(j, ''.join(lazy_pinyin(j)))


def half_black(path):
    image = Image.open(path)
    new_image = Image.new('L', (96, 96), (0))
    new_image.paste(image.crop((0,0,96,48)))
    new_image.save(path)


# names = glob.glob("F:/CHN_faces/chinese_cleaned_96px_grayscale_labeled/*/*.jpg")
# names = sorted(names)
# for i in names:
#     print(i)
#     half_black(i)

# lpls("", start="0100793", test=False)
# lpls("", start="1303492", test=False)
# 0000045
# lpls("", start="0000204", test=False)
def del_gray():
    names = glob.glob("C:/temp/CASIA-WebFace-112X96/*/*.jpg")
    names = sorted(names)

    for i in names:
        image = cv_imread(i)
        if len(image.shape) == 3:
            pass
        else:
            print("gray face:", i)
            os.remove(i)


def crop_bottom(path):
    im = Image.open(path)
    im = im.crop((0,16,96,112))
    im.save(path)


# names = glob.glob("C:/temp/CASIA-WebFace-96X96_G_NG_PF_EO_C/*")
# names = sorted(names)
# for i in names:
#     th = glob.glob(os.path.join(i, "*.jpg"))
#     if len(th) < 2:
#         print(i)
#         rmtree(i)


# names = glob.glob("C:/temp/CASIA-WebFace-96X96_G_nosunglasses_clear/*/*.jpg")
# names = sorted(names)
# num_names = len(names)
# names = random.sample(names, num_names//2)

# for i in names:
#     image = cv_imread(i)
#     half_black(i)

