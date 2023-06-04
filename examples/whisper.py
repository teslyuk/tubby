import pathlib
from extra.utils import download_file
import torch

if __name__ == "__main__":
  FILENAME = pathlib.Path(__file__).parent.parent / "weights" / "whisper-tiny.en.pt"
  download_file("https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt", FILENAME)
  state_dict = torch.load(FILENAME)
  