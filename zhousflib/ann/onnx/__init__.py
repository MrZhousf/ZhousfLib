# -*- coding: utf-8 -*-
# @Author  : zhousf
# @Date    : 2023/12/21 
# @Function:
"""
【onnx && cuda的版本对应关系】
onnx对应cuda的版本：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
注意onnxruntime与opset版本的对应关系


############## 【安装onnxruntime】 ##############
选择版本：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
【cpu】
pip install onnxruntime==1.13.1 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
【gpu】
pip install onnxruntime-gpu==1.13.1 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
验证：
import onnxruntime
onnxruntime.get_device()


############## 【验证导出onnx是否正确】 ##############
可视化网络结构：https://netron.app/
当output有if条件则会存在问题，更换opset版本(opset=10)或降低torch版本(1.8.0)
"""
