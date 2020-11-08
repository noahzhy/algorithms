import shutil
import os
import glob


def move_imgs(path):
    faces = glob.glob(path + "*/*/*/*/*.jpg")
    for i in faces:
        k = i.split("\\")
        t = k[1]
        k = k[2:]
        k = ["_".join(k)]
        # print(k)
        _dir = os.path.join(r'C:\Users\JX_COSMETICS\Downloads\K_faces', t)
        try:
            os.mkdir(_dir)
        except Exception:
            pass
        # print(i)
        os.rename(i, os.path.join(_dir, k[0]))


def remove_sth(path, mark="S004_*.jpg"):
    faces = glob.glob(path + "/*/" + mark)
    for i in faces:
        print(i)
        os.remove(i)




if __name__ == "__main__":
    # app(r"C:/Users/JX_COSMETICS/Downloads/High_Resolution/")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C1.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C2.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C3.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C4.jpg")

    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S003_L*_E*_C15.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S003_L*_E*_C16.jpg")
    remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S006_L2_E*_C*.jpg")
    remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S006_L2_E*_C*.jpg")

    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C11.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C12.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C15.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C16.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L*_E*_C17.jpg")

    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S005_L8_E*_C*.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L9_E*_C*.jpg")

    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L14_E*_C*.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L17_E*_C*.jpg")

    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L20_E*_C*.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L21_E*_C*.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L24_E*_C*.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L27_E*_C*.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L29_E*_C*.jpg")
    # remove_sth(r'C:\Users\JX_COSMETICS\Downloads\K_faces', mark="S*_L30_E*_C*.jpg")


