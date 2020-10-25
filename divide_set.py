import os, glob, random
from shutil import copyfile


def random_choose(path, dst, num=400):
    src = glob.glob(os.path.join(path, "*/*.jpg"))
    res = random.sample(src, 400)
    print(res)
    for idx, i in enumerate(res):
        copyfile(i, os.path.join(dst, "{}_{}".format(idx, os.path.basename(i))))


def train_test_split(path:str, target_dir:str, rate:int):
    try:
        os.mkdir(os.path.join(target_dir))
        os.mkdir(os.path.join(target_dir, "train_dir"))
        os.mkdir(os.path.join(target_dir, "validation_dir"))
    except Exception:
        pass

    dirs = glob.glob(os.path.join(path, "*/"))
    classes = [os.path.dirname(c).split("\\")[-1] for c in dirs]

    for i in classes:
        imgs = glob.glob(os.path.join(path, "{}/*.jpg".format(i)))
        train = random.sample(imgs, int(rate*len(imgs)))
        test = list(set(imgs) - set(train))
        try:
            os.mkdir(os.path.join(target_dir, "train_dir", i))
            os.mkdir(os.path.join(target_dir, "validation_dir", i))
        except Exception:
            pass

        for j in train:
            copyfile(j, os.path.join(target_dir, "train_dir", i, os.path.basename(j)))

        for k in test:
            copyfile(k, os.path.join(target_dir, "validation_dir", i, os.path.basename(k)))
        
        print(len(train)/len(imgs), len(test)/len(imgs))

    return classes


if __name__ == "__main__":
    # random_choose("F:\CASIA\CASIA-WebFace-96X96_grayscale", r"glasses\no_glasses")
    res = train_test_split("glasses", "dataset", .8)
    print(res)
