import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model


model = load_model("models/pos_face.h5")
model.summary()

def get_inputs(src=[]):
    pre_x = []
    for s in src:
        im = Image.open(s)
        im = np.array(im).reshape(96,96,1)
        # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(im)
    pre_x = np.array(pre_x) / 255.0
    return pre_x


predict_dir = r'C:\temp\CASIA-WebFace-96X96_grayscale_nosunglasses'
test = os.listdir(predict_dir)
# print(test)

# start = test.index("0500614")
start = 0

for testpath in test[start:]:
    images = []
    for fn in os.listdir(os.path.join(predict_dir, testpath)):
        if fn.endswith('jpg'):
            fd = os.path.join(predict_dir, testpath, fn)
            # print(fd)
            images.append(fd)

    if len(images) == 0:
        continue
    pre_x = get_inputs(images)
    pre_y = model.predict(pre_x)
    # print(pre_y)
    for idx, i in enumerate(pre_y):
        if i[0] > 0.5:
            print(images[idx])
            os.remove(images[idx])
