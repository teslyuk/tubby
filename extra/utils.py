from tqdm import tqdm
import tempfile
import numpy as np
import torch

def download_file(url, fp, skip_if_exists=True):
  import requests, os, pathlib
  if skip_if_exists and os.path.isfile(fp) and os.stat(fp).st_size > 0:
    return
  r = requests.get(url, stream=True)
  assert r.status_code == 200
  progress_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True, desc=url)
  with tempfile.NamedTemporaryFile(dir=pathlib.Path(fp).parent, delete=False) as f:
    for chunk in r.iter_content(chunk_size=16384):
      progress_bar.update(f.write(chunk))
    f.close()
    os.rename(f.name, fp)
    
      
# taken from OpenAI's Whisper repo
def sinusoids(length, channels, max_timescale=1000):
  """Returns sinusoids for positional embedding"""
  assert channels % 2 == 0
  log_timescale_increment = np.log(max_timescale) // (channels // 2 - 1)
  inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
  scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
  return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)