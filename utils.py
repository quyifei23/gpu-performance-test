from torch.utils.data import Dataset
import math

class Int8Dataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels
    
  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
  

def int_scalar(m: float):
  n, m0 = 0, 0
  for i in range(32):
    m0 = int(round(2 ** i * m))
    approx_m = float(m0) / (2 ** i)

    if math.isclose(m, approx_m, rel_tol=1e-9, abs_tol=1e-6):
      n = i
      break
    
  return n, m0