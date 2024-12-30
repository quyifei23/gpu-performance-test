# FC-3 model
import torch
import torch.ao.quantization
import torch.nn as nn

# 定义网络结构
class FC3(nn.Module):
    def __init__(self):
        super(FC3, self).__init__()
        self.fc0 = nn.Linear(28*28, 64, bias=False)
        self.fc1 = nn.Linear(64, 10, bias=False)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc0(x))
        x = self.fc1(x)
        # x = self.softmax(x)
        return x


# class FC3(nn.Module):
#   def __init__(self, int8_params=False, quantized=False, use_mlir=False, q_params: dict=None):
#     super(FC3, self).__init__()
#     self.int8_params = int8_params
#     self.quantized = quantized
#     self.q_params = q_params
#     self.use_mlir = use_mlir
#     assert not(int8_params==True and quantized==True), "\'int8_params\' and \'quantized\' can\'t both be true."

#     self.fc0 = nn.Linear(28 * 28, 64, bias=False)
#     self.fc1 = nn.Linear(64, 10, bias=False)
#     self.relu = nn.ReLU()
#     self.softmax = nn.Softmax(dim=1)

#     if self.int8_params:
#       self.fc0.weight.requires_grad = False
#       self.fc1.weight.requires_grad = False

#       with torch.no_grad():
#         self.fc0.weight.data = self.fc0.weight.data.to(torch.int8)
#         self.fc1.weight.data = self.fc1.weight.data.to(torch.int8)

#         # self.quant_scale = q_params['quant_scale']
#         # self.quant_zero_point = q_params['quant_zero_point'].to(torch.int8)
            
#         self.fc0_int_shift = self.q_params['fc0_int_shift']
#         self.fc0_int_scalar = self.q_params['fc0_int_scalar']
#         # self.fc0_weight_zero_point = self.q_params['fc0_weight_zero_point']
#         self.fc0_zero_point = self.q_params['fc0_zero_point']
            
#         self.fc1_int_shift = self.q_params['fc1_int_shift']
#         self.fc1_int_scalar = self.q_params['fc1_int_scalar']
#         # self.fc1_weight_zero_point = self.q_params['fc1_weight_zero_point']
#         self.fc1_zero_point = self.q_params['fc1_zero_point']


#     if self.quantized:
#       self.quant = torch.ao.quantization.QuantStub()
#       self.dequant = torch.ao.quantization.DeQuantStub()

#   def forward(self, x):
#     if self.int8_params:    
#       # reshape
#       x = x.view(x.size(0), -1)

#       # x = relu(fc0(x))
#       if not self.use_mlir:
#         x = x.to(torch.int32)
#         self.fc0.weight.data = self.fc0.weight.data.to(torch.int32)
#       x = (self.fc0(x) * self.fc0_int_scalar) >> self.fc0_int_shift
#       x = x + self.fc0_zero_point
#       x = torch.clamp(x, self.q_params['fc0_zero_point'], 127)
#       x = x.to(torch.int32)
#       x = x.to(torch.int8)
            
#       # x = fc1(x)
#       x = (x - self.fc0_zero_point)
#       if not self.use_mlir:
#         x = x.to(torch.int32)
#         self.fc1.weight.data = self.fc1.weight.data.to(torch.int32)
#       x = (self.fc1(x) * self.fc1_int_scalar) >> self.fc1_int_shift
#       x = x + self.fc1_zero_point
#       x = torch.clamp(x, -128, 127)
#       x = x.to(torch.int32)
#       x = x.to(torch.int8)
#     else:
#       if self.quantized: x = self.quant(x)
#       x = x.view(x.size(0), -1)
#       x = self.relu(self.fc0(x))
#       x = self.fc1(x)
#       if self.quantized: x = self.dequant(x)
#       if not self.int8_params: x = self.softmax(x)
      
#     return x