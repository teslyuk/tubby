import torch
import torch.nn as nn
from torch.nn import functional as F
from extra.utils import download_file
import pathlib

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

class AttentionHead(nn.Module):
  def __init__(self, head_size: int):
    super().__init__()
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x: torch.Tensor):
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    B, T, C = x.shape
    q = self.query(x) 
    k = self.key(x)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads: int, head_size: int):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x: torch.Tensor):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
  
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(n_embd * 4, n_embd),
      nn.ReLU()
    )
    
  def forward(self, x: torch.Tensor):
    return self.layers(x)
  
class ResidualAttentionBlock(nn.Module):
  def __init__(self, n_embd: int, n_head: int):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffn = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    
  def forward(self, x: torch.Tensor):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x
    
class GPTLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.positional_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[ResidualAttentionBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    
    self.apply(self._init_weights)
    
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      
  def forward(self, idx: torch.Tensor, targets=None):
    B, T = idx.shape
    
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
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
  
  chars = sorted(list(set(text)))
  vocab_size = len(chars)
  stoi = {ch: i for i, ch in enumerate(chars)}
  itos = {i: ch for i, ch in enumerate(chars)}
  encode = lambda s: [stoi[c] for c in s]
  decode = lambda l: ''.join([itos[i] for i in l])
  
  # Train and test splits
  data = torch.tensor(encode(text), dtype=torch.long)
  n = int(0.9*len(data))
  train_data = data[:n]
  val_data = data[n:]
  
  def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
  
  @torch.no_grad()
  def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
        X, Y = get_batch(split)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
      out[split] = losses.mean()
    model.train()
    return out
      
  model = GPTLanguageModel().to(device)
  print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  
  for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
  
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))