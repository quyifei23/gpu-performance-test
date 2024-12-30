import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch.ao.quantization
import onnx
from torch.ao.quantization.observer import MinMaxObserver, default_weight_observer

import os, sys

from model import FC3

batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

model_path = os.path.join(os.environ.get("FC3_PATH", ""), "model")
model = None

if len(sys.argv) == 2 and sys.argv[1] == '-q':
  model = FC3(quantized=True).to(device)
  state_dict = torch.load(model_path + '/fc3-quantized.pth')
  # qconfig = torch.ao.quantization.get_default_qconfig('x86')
  qconfig = torch.ao.quantization.QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.qint8),
    weight=default_weight_observer
  )
  torch.backends.quantized.engine = 'qnnpack'
  model.qconfig = qconfig
  model.softmax.qconfig = None
  model_prepared = torch.ao.quantization.prepare(model)
  model_quantized = torch.ao.quantization.convert(model_prepared)

  model_quantized.load_state_dict(state_dict)
  model = model_quantized
elif len(sys.argv) == 2 and sys.argv[1] == '-i':
  # 加载模型和参数
  checkpoint = torch.load(model_path + '/fc3-int8.pth')
  q_params = checkpoint['q_params']

  model = FC3(int8_params=True, q_params=q_params)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  # 创建TensorRT的Logger、Builder、Network和Parser
  G_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(G_LOGGER)
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, G_LOGGER)

  # 解析ONNX模型
  with open(onnx_model_path, "rb") as model:
      if not parser.parse(model.read()):
          print('ERROR: Failed to parse the ONNX file.')
          for error in range(parser.num_errors):
              print(parser.get_error(error))
          exit(1)

  # 配置构建器
  builder.max_workspace_size = 1 << 30  # 1GB
  builder.max_batch_size = 16  # 根据您的需求设置
  
  # 创建 TensorRT 引擎
  engine = builder.build_cuda_engine(network)
  if engine is None:
      print('ERROR: Failed to create the TensorRT engine.')
      exit(1)
  
  # 创建TensorRT运行时和执行上下文
  runtime = trt.Runtime(G_LOGGER)
  context = engine.create_execution_context()
  
  # 输入输出绑定
  input_binding_index = engine.get_binding_index("input")
  output_binding_index = engine.get_binding_index("output")
  
  # 为输入输出分配内存
  input_size = trt.volume(engine.get_binding_shape(input_binding_index)) * engine.max_batch_size
  output_size = trt.volume(engine.get_binding_shape(output_binding_index)) * engine.max_batch_size
  
  d_input = cuda.mem_alloc(input_size * np.dtype(np.float32).itemsize)
  d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)
  
  bindings = [int(d_input), int(d_output)]
  
  # 加载数据集
  dataset = torch.load(model_path + '/int8_dataset.pth')
  batch_size = 16  # 根据您的需求设置
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
  # 推理过程
  with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(loader):
          inputs = inputs.numpy()  # 将 PyTorch tensor 转为 NumPy 数组
  
          # 将输入数据拷贝到 GPU
          cuda.memcpy_htod(d_input, inputs)
  
          # 执行推理
          context.execute_v2(bindings)
  
          # 从 GPU 中拷贝输出结果
          output_array = np.empty(output_size, dtype=np.float32)
          cuda.memcpy_dtoh(output_array, d_output)
  
          # 处理输出结果
          print(f"Batch {batch_idx + 1}: Output = {output_array[:10]}")
  
else:
  model = FC3().to(device)
  model.load_state_dict(torch.load(model_path + '/fc3-model.pth'))
  
  model.eval()
  transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
  )

  dataset = datasets.MNIST(
    root="../../data", train=False, download=True, transform=transform
  )
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  correct, total = 0, 0
  with torch.no_grad():
    for images, labels in loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, pred = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (pred == labels).sum().item()
    
  print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
