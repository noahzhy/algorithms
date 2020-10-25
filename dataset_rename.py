import os
import glob


def run():
    count = 0
    _pre_dir_name = "masked"
    for i in glob.glob(r"data\FIR_real_fake\*\*.jpg"):
        count += 1
        dir_name = os.path.dirname(i).split("\\")[-1]
        if _pre_dir_name != dir_name:
            count = 0
        os.rename(i, os.path.join(os.path.dirname(i), "{}_{:04d}.jpg".format(dir_name, count)))
        _pre_dir_name = dir_name
        print(_pre_dir_name)

    print("done")


if __name__ == "__main__":
    run()