# -*- coding: utf-8 -*-
# @Author  : zhousf
# @Date    : 2023/12/13 
# @Function:
from pathlib import Path

from transformers import AutoModel, AutoConfig, AutoTokenizer


def load_config(model_dir: Path):
    """
    加载配置
    :param model_dir:
    :return:
    """
    return AutoConfig.from_pretrained(pretrained_model_name_or_path=model_dir)


def load_model(model_dir: Path, config):
    """
    加载模型
    :param model_dir:
    :param config:
    :return:
    """
    return AutoModel.from_pretrained(model_dir, config=config)


def load_tokenizer(model_dir: Path):
    """
    加载tokenizer
    :param model_dir: 模型目录
    :return:
    """
    tokenizer = None
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        pass
    return tokenizer
