from .whisper import Whisper
import torch
from torch import Tensor
from typing import Tuple, Dict, Optional
from torch.distributions import Categorical
import torch.nn.functional as F

class Inference:
  def __init__(self) -> None:
    pass

class GreedyDecoder:
  def __init__(self, temperature: float, eot: int):
    self.temperature = temperature
    self.eot = eot
  
  def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
    if self.temperature == 0:
      next_tokens = logits.argmax(dim=-1)
    else:
      next_tokens = Categorical(logits=logits / self.temperature).sample()
    logprobs = F.log_softmax(logits.float(), dim=-1)
    current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
    sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)
    next_tokens[tokens[:, -1] == self.eot] = self.eot
    tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
    completed = (tokens[:, -1] == self.eot).all()
    return tokens, completed
  
  def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
    # make sure each sequence has at least one EOT token at the end
    tokens = F.pad(tokens, (0, 1), value=self.eot)
    return tokens, sum_logprobs.tolist()
  
class BeamSearchDecoder:
  def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None):
    self.beam_size = beam_size
    self.eot = eot
    self.inference = inference
    self.patience = patience or 1.0
    self.max_candidates: int = round(beam_size * self.patience)
    self.finished_sequences = None
    
  def reset(self):
    self.finished_sequences = None
  
  def update():
    pass
  
  def finalize():
    pass