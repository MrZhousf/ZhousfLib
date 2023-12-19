# -*- coding: utf-8 -*-
# @Author  : zhousf
# @Date    : 2023/12/19 
# @Function:
import abc
from pathlib import Path


class Backend(metaclass=abc.ABCMeta):

    def __init__(self, *args):
        pass

    @abc.abstractmethod
    def build(self, **kwargs):
        pass

    @abc.abstractmethod
    def inference(self, inputs_list: list):
        pass

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    @staticmethod
    def get_file_by_suffix(model_dir: Path, suffix: str):
        return [file for file in model_dir.glob("*{0}".format(suffix))]

    @staticmethod
    def pop(kwargs: dict, key: str):
        if len(kwargs) == 0 or len(key) == 0:
            return
        if key in kwargs:
            kwargs.pop(key)