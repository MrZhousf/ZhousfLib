# -*- coding: utf-8 -*-
# @Author  : zhousf
# @Date    : 2023/12/18 
# @Function:
from pathlib import Path


class FastInfer(object):

    def __init__(self, model_dir: Path, device_id=-1):
        """
        :param model_dir:
        :param device_id: cpu上运行：-1 | gpu上运行：0 or 1 or 2...
        """
        self.model_dir = model_dir
        self.device_id = device_id
        self.backend = None

    def use_http_backend(self, url: str, model_name: str, model_version: str, http_inputs: list, http_outputs: list, **kwargs):
        """
        设置推理后端：http
        :param url:
        :param model_name:
        :param model_version:
        :param http_inputs:
        :param http_outputs:
        :param kwargs:
        :return:
        example:
        .use_http_backend(url="127.0.0.1:5005",
                          model_name="cosnet_onnx",
                          model_version="1",
                          http_inputs=[("input_ids", "INT64"), ("token_type_ids", "INT64"), ("attention_mask", "INT64")],
                          http_outputs=["output"])
        """
        from backend_http import BackendHttp
        self.backend = BackendHttp(model_dir=self.model_dir, device_id=self.device_id)
        return self.backend.build(url=url, model_name=model_name, model_version=model_version, http_inputs=http_inputs, http_outputs=http_outputs, **kwargs)

    def use_original_backend(self, module=None, **kwargs):
        """
        设置推理后端：原始模型-动态图
        :param module:
        :param kwargs:
        :return:
        example:
        .use_original_backend(module=torch.nn.Module())
        """
        from backend_original import BackendOriginal
        self.backend = BackendOriginal(model_dir=self.model_dir, device_id=self.device_id)
        return self.backend.build(module=module, **kwargs)

    def use_onnx_backend(self, from_platform: str = None, module=None, dynamic_axes: dict = None, opset_version=10, example_inputs=None, **kwargs):
        """
        设置推理后端：onnxruntime
        :param from_platform:
        :param module:
        :param dynamic_axes:
        :param opset_version:
        :param example_inputs:
        :param kwargs:
        :return:
        example:
        .use_onnx_backend(from_platform="torch",
                          module=torch.nn.Module(),
                          example_inputs=example_inputs,
                          dynamic_axes={'input_ids': {0: 'batch_size'},
                                        'token_type_ids': {0: 'batch_size'},
                                        'attention_mask': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        """
        from backend_onnx import BackendONNX
        self.backend = BackendONNX(model_dir=self.model_dir, device_id=self.device_id)
        return self.backend.build(from_platform=from_platform, module=module, dynamic_axes=dynamic_axes, opset_version=opset_version, example_inputs=example_inputs, **kwargs)

    def use_tensorrt_backend(self, from_platform: str = None, module=None, dynamic_axes: dict = None, opset_version=10,
                             example_inputs=None, shape: dict = None, **kwargs):
        """
        设置推理后端：tensorrt
        :param from_platform:
        :param module:
        :param dynamic_axes:
        :param opset_version:
        :param example_inputs:
        :param shape:
        :param kwargs:
        :return:
        example:
        若onnx文件存在时则只需传参数shape
        .use_tensorrt_backend(shape={"input_ids": [(10, 128), (10, 128), (10, 128)],
                                     "token_type_ids": [(10, 128), (10, 128), (10, 128)],
                                     "attention_mask": [(10, 128), (10, 128), (10, 128)]})
        若onnx文件不存在时
        .use_tensorrt_backend(from_platform="torch",
                              module=torch.nn.Module(),
                              example_inputs=example_inputs,
                              dynamic_axes={'input_ids': {0: 'batch_size'},
                                          'token_type_ids': {0: 'batch_size'},
                                          'attention_mask': {0: 'batch_size'},
                                          'output': {0: 'batch_size'}},
                              shape={"input_ids": [(10, 128), (10, 128), (10, 128)],
                                     "token_type_ids": [(10, 128), (10, 128), (10, 128)],
                                     "attention_mask": [(10, 128), (10, 128), (10, 128)]})
        """
        from backend_tensorrt import BackendTensorRT
        self.backend = BackendTensorRT(model_dir=self.model_dir, device_id=self.device_id)
        return self.backend.build(from_platform=from_platform, module=module, dynamic_axes=dynamic_axes,
                                  opset_version=opset_version, example_inputs=example_inputs, shape=shape, **kwargs)

    def use_torch_script_backend(self, module=None, example_inputs=None, **kwargs):
        """
        设置推理后端：torch_script
        :param module:
        :param example_inputs:
        :param kwargs:
        :return:
        example:
        .use_torch_script_backend(module=torch.nn.Module(), example_inputs=example_inputs)
        """
        from backend_torch_script import BackendTorchScript
        self.backend = BackendTorchScript(model_dir=self.model_dir, device_id=self.device_id)
        return self.backend.build(module=module, example_inputs=example_inputs, **kwargs)

    def infer(self, inputs_list: list):
        assert self.backend is not None, "请设置backend，例如.use_onnx_backend()"
        return self.backend.inference(inputs_list=inputs_list)

    @staticmethod
    def demo():
        import torch
        from zhousflib.ann.torch import torch_to_onnx
        fast_infer = FastInfer(model_dir=Path(r"F:\torch\test"), device_id=0)
        fast_infer.use_tensorrt_backend(from_platform="torch",
                                        module=torch.nn.Module(),
                                        example_inputs=(torch_to_onnx.example_inputs_demo(device_id=0),),
                                        dynamic_axes={'input_ids': {0: 'batch_size'},
                                                      'token_type_ids': {0: 'batch_size'},
                                                      'attention_mask': {0: 'batch_size'},
                                                      'output': {0: 'batch_size'}},
                                        shape={"input_ids": [(10, 128), (10, 128), (10, 128)],
                                               "token_type_ids": [(10, 128), (10, 128), (10, 128)],
                                               "attention_mask": [(10, 128), (10, 128), (10, 128)]})
        return fast_infer.infer(torch_to_onnx.example_inputs_demo()).tolist()

