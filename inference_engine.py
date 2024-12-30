import tensorrt as trt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from model import FC3
from utils import Int8Dataset, int_scalar
import os
import torch.onnx
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit 

batch_size = 64
model_path = os.path.join(os.environ.get("FC3_PATH", ""), "model")
os.makedirs(model_path, exist_ok=True)

data_dir = '../../data/MNIST'
image_dir = os.path.join(data_dir, 'features')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

transform = transforms.Compose(
  [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.MNIST(
  root="../../data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
  root='../../data', train=False, download=True, transform=transform
)

test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)

def prepare_data(data_loader):
    images, labels = [], []
    for img, label in data_loader:
        images.append(img.numpy()) # 转换为 NumPy
        labels.append(label.numpy()) # 转换为 NumPy

    # 确保数据是规则的数组
    images = np.vstack(images) # 堆叠成形状为 (N, 1, 28, 28)
    labels = np.hstack(labels) # 合并为一维数组，形状为 (N,)
    return images, labels


def check_cuda_error(msg=""):
    """
    检查 CUDA 操作中的错误，并打印错误信息。

    Args:
    msg (str): 提示消息，用于标识当前检查的上下文。
    """
    try:
        # 检查最后一次 CUDA 调用的状态
        cuda.Context.synchronize() # 强制同步，捕获潜在的错误
    except cuda.Error as e:
        print(f"CUDA Error {msg}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected Error {msg}: {e}")
        raise
# 在关键的CUDA操作后调用check_cuda_error()


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        # check_cuda_error("检查反序列化引擎时是否有错误")  # 检查反序列化引擎时是否有错误
        return engine

def infer(engine, input_data):
    # 创建上下文
    context = engine.create_execution_context()

    # 分配输入输出的显存
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

    input_buffer = cuda.mem_alloc(trt.volume(input_shape) * input_dtype().itemsize)
    output_buffer = cuda.mem_alloc(trt.volume(output_shape) * output_dtype().itemsize)
    
    # check_cuda_error("检查内存分配时是否有错误")  # 检查内存分配时是否有错误
    bindings = [int(input_buffer), int(output_buffer)]

    # 复制数据到输入缓冲区
    stream = cuda.Stream()
    cuda.memcpy_htod_async(input_buffer, input_data, stream)
    # check_cuda_error("检查数据传输时是否有错误1")  # 检查数据传输时是否有错误

    # 推理
    context.execute_async_v2(bindings, stream.handle)
    # check_cuda_error("检查数据传输时是否有错误2")  # 检查数据传输时是否有错误
    output_data = np.empty(output_shape, dtype=output_dtype)
    cuda.memcpy_dtoh_async(output_data, output_buffer, stream)
    # check_cuda_error("检查数据传输时是否有错误3")  # 检查数据传输时是否有错误
    stream.synchronize()
    # check_cuda_error("检查数据传输时是否有错误4")  # 检查数据传输时是否有错误

    return output_data

images, labels = prepare_data(test_loader)
engine_path = model_path + "/fc3-int8-model.engine"
engine = load_engine(engine_path)
def evaluate_accuracy(engine, images, labels):
    correct = 0
    total = len(labels)

    for i in range(total):
        # 确保输入数据展平并转换为 float32
        input_data = images[i].flatten().astype(np.float32)
        output = infer(engine, input_data)
        predicted_label = np.argmax(output)

        if predicted_label == labels[i]:
            correct += 1

    accuracy = correct / total
    print(f"准确率: {accuracy:.2%}")
    return accuracy

accuracy = evaluate_accuracy(engine, images, labels)

