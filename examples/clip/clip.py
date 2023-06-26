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

class LayerNorm(nn.LayerNorm):
  """Subclass torch's LayerNorm to handle fp16."""
  def forward(self, x: torch.Tensor):
    orig_type = x.dtype
    ret = super().forward(x.type(torch.float32))
    return ret.type(orig_type)
  
class QuickGELU(nn.Module):
  def forward(self, x: torch.Tensor):
    return x * torch.sigmoid(1.702 * x)

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
  
class AttentionPool2d(nn.Module):
  def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
    super().__init__()
    self.positional_embedding = nn.Parameter(torch.randn(spacial_dim * 2 + 1, embed_dim) / embed_dim ** 0.5)
    self.q_proj = nn.Linear(embed_dim, embed_dim)
    self.k_proj = nn.Linear(embed_dim, embed_dim)
    self.v_proj = nn.Linear(embed_dim, embed_dim)
    self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
    self.num_heads = num_heads
    
  def forward(self, x):
    x = x.flatten(start_dim=2).permute(2, 0, 1) # NCHW -> (HW)NC
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0) # (HW+1)NC
    x = x + self.positional_embedding[:, None, :].to(x.dtype) # (HW+1)NC
    x, _ = F.multi_head_attention_forward(
      query=x[:1], 
      key=x,
      value=x,
      embed_dim_to_check=x.shape[-1],
      num_heads=self.num_heads,
      q_proj_weight=self.q_proj.weight,
      k_proj_weight=self.k_proj.weight,
      v_proj_weight=self.v_proj.weight,
      in_proj_weight=None,
      in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
      bias_k=None,
      bias_v=None,
      add_zero_attn=False,
      dropout_p=0,
      out_proj_weight=self.c_proj.weight,
      out_proj_bias=self.c_proj.bias,
      use_separate_proj_weight=True,
      training=self.training,
      need_weights=False
    )
    return x.squeeze(0)
    
    
# A ResNet class that is similar to torchvision's but contains the following changes:
#   - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
#   - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
#   - The final pooling layer is a QKV attention instead of an average pool
class ModifiedResNet(nn.Module):
  def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
    super().__init__()
    self.output_dim = output_dim
    self.input_resolution = input_resolution
    
    # the 3-layer stem
    self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(width // 2)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(width // 2)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(width)
    self.relu3 = nn.ReLU(inplace=True)
    self.avgpool = nn.AvgPool2d(2)
    
    # residual layers
    self._inplanes = width # this is a *mutable* variable used during construction
    self.layer1 = self._make_layer(width, layers[0])
    self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
    self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
    self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
    embed_dim = width * 32 # the ResNet feature dimension
    self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim=embed_dim, num_heads=heads, output_dim=output_dim)
    
  def _make_layer(self, planes, blocks, stride=1):
    layers = [Bottleneck(self._inplanes, planes=planes, stride=stride)]
    self._inplanes = planes * Bottleneck.expansion
    for _ in range(1, blocks):
      layers.append(Bottleneck(self._inplanes, planes=planes))
    return nn.Sequential(*layers)
  
  def forward(self, x):
    def stem(x):
      x = self.relu1(self.bn1(self.conv1(x)))
      x = self.relu2(self.bn2(self.conv2(x)))
      x = self.relu3(self.bn3(self.conv3(x)))
      x = self.avgpool(x)
      return x
    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.attnpool(x)
    return x
        
if __name__ == "__main__":
  BASE = pathlib.Path(__file__).parent.parent.parent / "weights"
  FILENAME = BASE / "CLIP-ViT-B-32.pt"
  download_file(_MODELS["ViT-B/32"], FILENAME)