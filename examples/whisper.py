import torch
import pathlib

from extra.utils import download_file, sinusoids

import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional

#export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"

class LayerNorm(nn.LayerNorm):
  def forward(self, x: Tensor) -> Tensor:
    return super().forward(x.float()).to(x.dtype)
    
class Linear(nn.Linear):
  def forward(self, x: Tensor) -> Tensor:
    return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
  
class Conv1D(nn.Conv1d):
  def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
    return super()._conv_forward(x.float(), weight.to(x.dtype), None if bias is None else bias.to(x.dtype))
  
class MultiHeadAttention(nn.Module):
  def __init__(self, n_state: int, n_head: int):
    super().__init__()
    self.n_head = n_head
    self.query = nn.Linear(n_state, n_state)
    self.key = nn.Linear(n_state, n_state, bias=False)
    self.value = nn.Linear(n_state, n_state)
    self.out = nn.Linear(n_state, n_state)
    
  def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
    q = self.query(x)
    
    if kv_cache is None or xa is None or self.key not in kv_cache:
      # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
      # otherwise, perform key/value projections for self- or cross-attention as usual.
      k = self.key(xa if xa is not None else x)
      v = self.value(xa if xa is not None else x)
    else:
      # for cross-attention, calculate keys and values once and reuse in subsequent calls.
      k = kv_cache[self.key]
      v = kv_cache[self.value]
      
    wv, qk = self.qkv_attention(q, k, v, mask)
    return self.out(wv), qk
  
  def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state //  self.n_head) ** -0.25
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    qk = q @ k
    if mask is not None:
      qk = qk + mask[:n_ctx, :n_ctx]
    qk = qk.float()
    w = F.softmax(qk, dim=-1).to(q.dtype)
    return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
    
    
class ResidualAttentionBlock(nn.Module):
  def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
    super().__init__()
    self.attn = MultiHeadAttention(n_state, n_head)
    self.attn_ln = LayerNorm(n_state)
    
    self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
    self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
    
    n_mlp = n_state * 4
    self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
    self.mlp_ln = LayerNorm(n_state)
    
  def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
    x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
    if self.cross_attn:
      x = x + self.cross_attn(self.cross_attn_ln(xa), xa, kv_cache=kv_cache)[0]
    x = x + self.mlp(self.mlp_ln(x))
    return x
    
if __name__ == "__main__":
  FILENAME = pathlib.Path(__file__).parent.parent / "weights" / "whisper-tiny.en.pt"
  download_file("https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt", FILENAME)
  state_dict = torch.load(FILENAME)["model_state_dict"]
  for k, v in state_dict.items():
    print(k)