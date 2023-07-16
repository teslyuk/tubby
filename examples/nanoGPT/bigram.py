import torch
import torch.nn as nn
from torch.nn import functional as F
from extra.utils import download_file
import pathlib

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

class BigramLM(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx     

if __name__ == "__main__":
  BASE = pathlib.Path(__file__).parent.parent.parent / "weights"
  FILENAME = BASE / "shakespeare.txt"
  download_file("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", FILENAME)
  
  with open(FILENAME, 'r', encoding='utf-8') as f:
    text = f.read()
  
  print(len(set(text)))