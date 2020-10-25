import os
import glob
import numpy as np


def do():
    count = 0
    for fname in glob.glob('F:/lfw-deepfunneled_with_mask/*'):
        print(os.path.basename(fname))
        count += 1

        try:
            os.mkdir(os.path.join("F:/lfw-deepfunneled_with_mask_each4", os.path.basename(fname)))
        except:
            pass

        nd = glob.glob(os.path.join(fname, '*_[1-9][0-9][0-9][0-9].jpg'))
        nd.extend(glob.glob(os.path.join(fname, '*_[0-9][0-9][0-9].jpg')))
        nd.extend(glob.glob(os.path.join(fname, '*_[1-9][0-9][0-9][0-9][0-9].jpg')))
        # print(nd)
        # return 0
        n = len(nd)//30
        a = np.array(nd)
        # print(a.shape)
        a = np.reshape(a, (n, 30))
        for i in range(0, n):
            b = np.random.choice(a[i], (1, 3), replace=False)
            for j in b[0]:
                # print(j)
                os.rename(j, j.replace("lfw-deepfunneled_with_mask", "lfw-deepfunneled_with_mask_each4"))
            # print(b)


def add():
    for fname in glob.glob('F:/lfw-deepfunneled_with_mask/*/*_0[0-9][0-9][0-9].jpg'):
        os.rename(fname, fname.replace("lfw-deepfunneled_with_mask", "lfw-deepfunneled_with_mask_each4"))



if __name__ == "__main__":
    # do()
    add()