import torch
from src import config
if config.onnx_ongoing:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

