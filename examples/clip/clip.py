import pathlib
import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple, Union
from extra.utils import download_file
from torch import nn
from collections import OrderedDict

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

class Bottleneck(nn.Module):
  expansion = 4
  
  def __init__(self, inplanes, planes, stride=1):
    super().__init__()
    # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
    self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu1 = nn.ReLU(inplace=True)
    
    self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.relu2 = nn.ReLU(inplace=True)
    
    self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
    
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
    self.bn3 = nn.BatchNorm3d(planes * self.expansion)
    self.relu3 = nn.ReLU(inplace=True)
    
    self.downsample = None
    self.stride = stride
    
    if stride > 1 or inplanes != planes * Bottleneck.expansion:
      # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
      self.downsample = nn.Sequential(
        OrderedDict([
          ("-1", nn.AvgPool2d(stride)),
          ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
          ("1", nn.BatchNorm2d(planes * self.expansion))
        ])
      )
      
  def forward(self, x: torch.Tensor):
    identity = x
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.relu2(self.bn2(self.conv2(out)))
    out = self.avgpool(out)
    out = self.bn3(self.conv3(out))
    
    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu3(out)
    return out  
    
if __name__ == "__main__":
  BASE = pathlib.Path(__file__).parent.parent.parent / "weights"
  FILENAME = BASE / "CLIP-ViT-B-32.pt"
  download_file(_MODELS["ViT-B/32"], FILENAME)