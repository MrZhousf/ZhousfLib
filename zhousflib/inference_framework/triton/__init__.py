# -*- coding: utf-8 -*-
# @Author  : zhousf
# @Date    : 2023/12/14 
# @Function:

"""
官方文档：https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/performance_tuning.html
官方git：https://github.com/triton-inference-server/server/blob/main/docs/README.md#user-guide
# 创建模型仓库目录：
triton/models/模型名称/版本号/模型
# 在模型名称目录下创建模型配置文件：
参考：https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md
config.pbtxt
# 创建一个模型仓库容器并启动一个模型推理服务（gpus=all）
在triton目录下运行：
docker run -it -d --rm --gpus=1 --network=host -v $PWD:/mnt --name zhousf-triton-server nvcr.io/nvidia/tritonserver:23.11-py3 tritonserver --model-repository=/mnt/models --http-port=5005 --grpc-port=5006 --metrics-port=5007 --log-info=true --log-error=true
# 显示如下表示成功：
+---------------+---------+--------+
| Model         | Version | Status |
+---------------+---------+--------+
| cosnet_onnx   | 1       | READY  |
+---------------+---------+--------+
# 测试模型服务是否启动成功
curl -v localhost:5005/v2/health/ready

perf_analyzer -m cosnet_onnx --shape input:-1,3,640,640 -i grpc --concurrency-range 1:50:10

"""



