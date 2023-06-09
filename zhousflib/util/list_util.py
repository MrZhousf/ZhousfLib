# -*- coding:utf-8 -*-
# Author:  zhousf
# Description:  list交集、并集、差集运算
import random


def random_choices(data_list: list, choose_k=3) -> list:
    """
    从列表中随机抽取choose_k个数（会有重复值）
    :param data_list:
    :param choose_k:
    :return:
    """
    return random.choices(data_list, k=choose_k)


def none_filter(data: list) -> list:
    """
    去掉list中的None值
    :param data:
    :return:
    """
    if isinstance(data, list):
        res = []
        for item in data:
            if isinstance(item, list):
                res.append(list(filter(None, item)))
            else:
                res = list(filter(None, data))
                break
        return res
    return data


def intersection(a, b):
    """
    交集
    :param a: [1, 2, 3, 4, 5]
    :param b: [2, 3, 9]
    :return: [2, 3]
    """
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return list(set(a).intersection(set(b)))


def union(a, b):
    """
    并集
    :param a: [1, 2, 3, 4, 5]
    :param b: [2, 3, 9]
    :return: [1, 2, 3, 4, 5, 9]
    """
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return list(set(a).union(set(b)))


def difference(a, b):
    """
    差集
    :param a: [1, 2, 3, 4, 5]
    :param b: [2, 3, 9]
    :return: [1, 4, 5]
    """
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return list(set(a).difference(set(b)))


