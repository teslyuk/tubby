from tqdm import tqdm
import tempfile

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
    
    