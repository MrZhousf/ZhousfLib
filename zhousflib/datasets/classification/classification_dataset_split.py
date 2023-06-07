# -*- coding: utf-8 -*-
# @Author  : zhousf
# @Function: 图像分类数据制作
import shutil
from pathlib import Path


def fetch_available_cls_folder(img_dir: Path):
    """
    删除空目录
    :param img_dir:
    :return:
    """
    for folder in img_dir.iterdir():
        cls = [i for i in folder.rglob("*.*")]
        if len(cls) == 0:
            print(folder)
            shutil.rmtree(folder)


def generate_train_txt(image_dir: Path):
    train_txt_file = image_dir.parent.joinpath("train_list.txt")
    label_list_file = image_dir.parent.joinpath("label_list.txt")
    data = []
    label_list = []
    for folder in image_dir.rglob("*.*"):
        if folder.parent.name not in label_list:
            label_list.append(folder.parent.name)
    if not label_list_file.exists():
        with label_list_file.open("w", encoding="utf-8") as f:
            for i, d in enumerate(label_list):
                f.write("{0} {1}\n".format(i, d))
    for folder in image_dir.rglob("*.*"):
        file = "{0}/{1}/{2} {3}\n".format(folder.parent.parent.name, folder.parent.name, folder.name, label_list.index(folder.parent.name))
        print(file)
        data.append(file)
    if not train_txt_file.exists():
        with train_txt_file.open("w", encoding="utf-8") as f:
            for d in data:
                f.write(d)


if __name__ == "__main__":
    generate_train_txt(Path(r"C:\Users\zhousf-a\Desktop\steel_id\result"))





