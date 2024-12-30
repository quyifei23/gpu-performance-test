import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import FC3
from utils import Int8Dataset, int_scalar
import os
import torch.onnx
import tensorrt as trt
import onnx

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

def train_original_model():
  model = FC3()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters())
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  epochs = 20
  for epoch in range(epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}")
  torch.save(model.state_dict(), model_path + "/fc3-model.pth")
  print("Original model saved successfully.")


if __name__ == "__main__":
  # train_original_model()
  # model = FC3()             
  # model.load_state_dict(torch.load(model_path + '/fc3-model.pth'))
  # model.eval()
  # dummy_input = torch.randn(1, 1, 28, 28)
  # torch.onnx.export(model, dummy_input, model_path + "/fc3-model.onnx", verbose=False)
  # print("Original model export to onnx successfully")
  
  # check onnx model
  model = onnx.load(model_path + "/fc3-model.onnx")
  onnx.checker.check_model(model)

