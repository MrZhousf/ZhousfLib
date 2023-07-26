# -*- coding: utf-8 -*-
# @Author  : zhousf
# @Function:
import numpy
from pathlib import Path
from PIL import Image, ImageDraw


def four_point_convert_bbox(four_points: list):
    """
    四点转换成bbox
    :param four_points: [[252, 140], [300, 140], [300, 189], [252, 189]]
    :return:
    """
    arr = numpy.asarray(four_points)
    x_min = min(arr[:, 0])
    y_min = min(arr[:, 1])
    x_max = max(arr[:, 0])
    y_max = max(arr[:, 1])
    return x_min, y_min, x_max, y_max


def draw_rectangle(bbox: list, image_file: Path = None, image_size: list = None, show=True):
    """
    绘制矩形框
    :param bbox: [(x_min, y_min, x_max, y_max)]
    :param image_file: 空时以空白为背景进行绘制
    :param image_size:
    :param show:
    :return:
    """
    draw_p = []
    for box in bbox:
        x_min, y_min, x_max, y_max = box
        draw_p.append([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    return draw_polygon(polygon=draw_p, image_file=image_file, image_size=image_size, show=show)


def draw_polygon(polygon: list, image_file: Path = None, image_size: list = None, show=True):
    """
    绘制四边形
    :param polygon: [[[255, 376], [291, 409], [255, 443], [218, 409]], [[252, 140], [300, 140], [300, 189], [252, 189]]]
    :param image_file: 空时以空白为背景进行绘制
    :param image_size:
    :param show:
    :return:
    """
    if image_size is None:
        image_size = [500, 500]
    if image_file is None:
        image = Image.new('RGB', (image_size[0], image_size[1]), (255, 255, 255))
    else:
        image = Image.open(image_file)
        if image.mode != "RGB":
            image = image.convert('RGB')
    draw = ImageDraw.ImageDraw(image)
    for point in polygon:
        draw_p = [(p[0], p[1]) for p in point]
        draw.polygon(draw_p, outline=(255, 0, 0))
    if show:
        image.show()
    return image


if __name__ == "__main__":
    draw_rectangle([(218, 376, 291, 443)])
    # draw_polygon([[[255, 376], [291, 409], [255, 443], [218, 409]], [[252, 140], [300, 140], [300, 189], [252, 189]]])
    pass
