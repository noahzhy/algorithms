import cv2
import imutils
from itertools import product
from PIL import Image
from glob import glob
from cv2 import (CascadeClassifier, cvtColor, resize, COLOR_BGR2GRAY,
                 CASCADE_FIND_BIGGEST_OBJECT, CASCADE_DO_ROUGH_SEARCH,
                 INTER_AREA)
from numpy import (empty_like, dot, linalg, isscalar, array, asarray, sqrt,
                   savetxt)

import dlib
import numpy as np
from imutils.face_utils import FaceAligner
from imutils import face_utils

miniface_v = 8  # minimum face size ratio; too low and we get false positives
CASCFILE = "models/haarcascade_frontalface_alt.xml"

predictor_path = "models/shape_predictor_5_face_landmarks.dat"
face_path = "img/0005.jpg"


def intersect(v1, v2):
    a1, a2 = v1
    b1, b2 = v2
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = dot(dap, db).astype(float)
    num = dot(dap, dp)
    return (num / denom) * db + b1


def perp(a):
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def distance(pt1, pt2):
    """Returns the euclidian distance in 2D between 2 pts."""
    distance = linalg.norm(pt2 - pt1)
    return distance


def check_positive_scalar(num):
    """Returns True if value if a positive scalar."""
    if num > 0 and not isinstance(num, str) and isscalar(num):
        return int(num)
    raise ValueError("A positive scalar is required")


class AutoCrop():
    def __init__(self, width=112, height=112, miniface_v=12):
        self.minface = miniface_v
        self.casc_path = CASCFILE
        self.width = width
        self.height = height
        self.face_percent = check_positive_scalar(87)
        self.aspect_ratio = 1


    def crop(self, input_image, output_filename=None, to_gray=False):
        image = asarray(input_image)
        img_height, img_width = image.shape[:2]
        minface = int(sqrt(img_height ** 2 + img_width ** 2) / self.minface)
        # create the haar cascade
        face_cascade = CascadeClassifier(self.casc_path)
        # detect faces in the image
        faces = face_cascade.detectMultiScale(
            cvtColor(image, COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(minface, minface),
            flags=CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH,
        )
        # handle no faces
        if len(faces) == 0:
            return None
        # make padding from biggest face found
        x, y, w, h = faces[-1]
        pos = self._crop_positions(img_height, img_width, x, y, w, h,)
        # actual cropping
        image = image[pos[0]: pos[1], pos[2]: pos[3]]
        # resize
        image = resize(image, (self.width, self.height), interpolation=INTER_AREA)
        
        if output_filename:
            if to_gray:
                Image.fromarray(image).convert("L").save(output_filename)
            else:
                cv2.imwrite(output_filename, image)
        
        return image


    def zoom_image(self, img_mat, faces_xywh):
        image = asarray(img_mat)
        img_height, img_width = image.shape[:2]
        x, y, w, h = faces_xywh
        pos = self._crop_positions(img_height, img_width, x, y, w, h,)
        # actual cropping
        # print("pos:", pos)
        image = image[pos[0]: pos[1], pos[2]: pos[3]]
        # resize
        image = resize(image, (self.width, self.height), interpolation=INTER_AREA)
        return image


    def _crop_positions(self, imgh, imgw, x, y, w, h,):
        zoom = self._determine_safe_zoom(imgh, imgw, x, y, w, h)

        # adjust output height based on percent
        if self.height >= self.width:
            height_crop = h * 100.0 / zoom
            width_crop = self.aspect_ratio * float(height_crop)
        else:
            width_crop = w * 100.0 / zoom
            height_crop = float(width_crop) / self.aspect_ratio

        # calculate padding by centering face
        xpad = (width_crop - w) / 2
        ypad = (height_crop - h) / 2

        # calc. positions of crop
        h1 = x - xpad
        h2 = x + w + xpad
        v1 = y - ypad
        v2 = y + h + ypad

        return [int(v1), int(v2), int(h1), int(h2)]


    def _determine_safe_zoom(self, imgh, imgw, x, y, w, h):
        # find out what zoom factor to use given self.aspect_ratio
        corners = product((x, x + w), (y, y + h))
        center = array([x + int(w / 2), y + int(h / 2)])
        # image corners
        i = array([(0, 0), (0, imgh), (imgw, imgh), (imgw, 0), (0, 0)])
        image_sides = [(i[n], i[n + 1]) for n in range(4)]
        corner_ratios = [self.face_percent]
        for c in corners:
            corner_vector = array([center, c])
            a = distance(*corner_vector)
            intersects = list(intersect(corner_vector, side) for side in image_sides)
            for pt in intersects:
                # if intersect within image
                if (pt >= 0).all() and (pt <= i[2]).all():
                    dist_to_pt = distance(center, pt)
                    corner_ratios.append(100 * a / dist_to_pt)
        return max(corner_ratios)


def crop_to_save(ac, face_path, save_path=None, cover=True):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    image = cv2.imread(face_path)
    # image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Input", image)
    rects = detector(gray)

    if cover:
        save_path = face_path

    for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            faceAligned = fa.align(image, gray, rect)
            ac.crop(faceAligned, save_path)

            # cv2.imshow("Original", faceOrig)
            # cv2.imshow("Aligned", faceAligned)
            # cv2.waitKey(0)
            break


# ac = AutoCrop()
# crop_to_save(ac, face_path)