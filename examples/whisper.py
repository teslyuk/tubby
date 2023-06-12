import torch
import pathlib
import numpy as np
import librosa

from extra.utils import download_file, sinusoids, get_encoding, load_audio, pad_or_trim, log_mel_spectrogram

import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import Tensor, nn
from typing import Optional, Iterable, Dict
from dataclasses import dataclass

#export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"

@dataclass
class ModelDimensions:
  n_mels: int
  n_audio_ctx: int
  n_audio_state: int
  n_audio_head: int
  n_audio_layer: int
  n_vocab: int
  n_text_ctx: int
  n_text_state: int
  n_text_head: int
  n_text_layer: int

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
    self.query = Linear(n_state, n_state)
    self.key = Linear(n_state, n_state, bias=False)
    self.value = Linear(n_state, n_state)
    self.out = Linear(n_state, n_state)

  def forward(self,x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
    q = self.query(x)
    if kv_cache is None or xa is None or self.key not in kv_cache:
      # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
      # otherwise, perform key/value projections for self- or cross-attention as usual.
      k = self.key(x if xa is None else xa)
      v = self.value(x if xa is None else xa)
    else:
      # for cross-attention, calculate keys and values once and reuse in subsequent calls.
      k = kv_cache[self.key]
      v = kv_cache[self.value]
    wv, qk = self.qkv_attention(q, k, v, mask)
    return self.out(wv), qk

  def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
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
    self.cross_attn = (MultiHeadAttention(n_state, n_head) if cross_attention else None)
    self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
    n_mlp = n_state * 4
    self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
    self.mlp_ln = LayerNorm(n_state)

  def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
    x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
    if self.cross_attn:
      x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
    x = x + self.mlp(self.mlp_ln(x))
    return x
  
class AudioEncoder(nn.Module):
  def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
    super().__init__()
    self.conv1 = Conv1D(n_mels, n_state, kernel_size=3, padding=1)
    self.conv2 = Conv1D(n_state, n_state, kernel_size=3, stride=2, padding=1)
    self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
    
    self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
    self.ln_post = LayerNorm(n_state)
    
  def forward(self, x: Tensor):
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx) 
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)
    
    assert x.shape[1:] == self.positional_embedding.shape, f"Expected shape {self.positional_embedding.shape}, got {x.shape[1:]}"
    x = x + self.positional_embedding
    
    for block in self.blocks:
      x = block(x)
      
    x = self.ln_post(x)
    return x
    
class TextDecoder(nn.Module):
  def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
    super().__init__()
    self.token_embedding = nn.Embedding(n_vocab, n_state)
    self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
    self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList([ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)])
    self.ln = LayerNorm(n_state)
    mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
    self.register_buffer("mask", mask, persistent=False)
    
  def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
    """
    x : torch.LongTensor, shape = (batch_size, <= n_ctx)
        the text tokens
    xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
        the encoded audio features to be attended on
    """
    offset = next(iter(kv_cache.values())).shape[1] if kv_cache is not None else 0
    x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
    x = x.to(xa.dtype)
    for block in self.blocks:
      x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
    x = self.ln(x)
    logits = x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1).float()
    return logits
  
class Whisper(nn.Module):
  def __init__(self, dims: ModelDimensions):
    super().__init__()
    self.dims = dims
    self.encoder = AudioEncoder(
      self.dims.n_mels,
      self.dims.n_audio_ctx,
      self.dims.n_audio_state,
      self.dims.n_audio_head,
      self.dims.n_audio_layer,
    )
    self.decoder = TextDecoder(
      self.dims.n_vocab,
      self.dims.n_text_ctx,
      self.dims.n_text_state,
      self.dims.n_text_head,
      self.dims.n_text_layer,
    )
    
  def embed_audio(self, mel: torch.Tensor):
    return self.encoder(mel)
  
  def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
    return self.decoder(tokens, audio_features)
  
  def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
    return self.decoder(tokens, self.encoder(mel))

def img(x):
  plt.imshow(x.numpy())
  plt.show()
  
if __name__ == "__main__":
  BASE = pathlib.Path(__file__).parent.parent / "weights"
  FILENAME = BASE / "whisper-tiny.en.pt"
  download_file("https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt", FILENAME)
  state = torch.load(FILENAME)
  # print(state["dims"])
  dims = ModelDimensions(**state["dims"])
  model = Whisper(dims)
  model.load_state_dict(state["model_state_dict"])
  
  mel_spec = load_audio("/Users/tesnik/Desktop/Workspace/tesnikzoo/data/audio.wav")
  mel_spec = pad_or_trim(mel_spec)
  mel_spec = log_mel_spectrogram(mel_spec).unsqueeze(0)
  
  enc = get_encoding(dims.n_vocab)
  
  audio_features = model.encoder(mel_spec)
  tokens = [enc._special_tokens["<|startoftranscript|>"]]
  # print(audio_features.shape)
  # print(tokens)
  # print(tokens.shape)
  
  res: str = ""
  while True:
    text = torch.tensor([tokens])
    out = model.decoder(text, audio_features)
    idx = out[0, -1].detach().numpy().argmax()
    tokens.append(idx)
    trascription = enc.decode(tokens)
    
    # bad code
    if "<|endoftext|>" in trascription.split()[-1]:
      res = trascription
      break
  
  print(res)
  