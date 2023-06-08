from tqdm import tqdm
import tempfile
import numpy as np
import torch
import librosa
import functools
import pathlib
import base64

LANGUAGES = {
  "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
  "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
  "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
  "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak", "te": "telugu",
  "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
  "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
  "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan", "ka": "georgian",
  "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese", "ht": "haitian creole",
  "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy",
  "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese",
}

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

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160

@functools.lru_cache(maxsize=None)
def get_filters(sample_rate, n_fft, n_mels):
  return torch.tensor(librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels))

def prep_audio(fname: str) -> torch.Tensor:
  import librosa
  y, sr = librosa.load(fname, sr=SAMPLE_RATE)
  audio = torch.from_numpy(y)
  window = torch.hann_window(N_FFT)
  stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
  magnitudes = stft[..., :-1].abs() ** 2
  mel_spec = get_filters(SAMPLE_RATE, N_FFT, N_MELS) @ magnitudes
  log_spec = torch.clamp(mel_spec, min=1e-10).log10()
  log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
  log_spec = (log_spec + 4.0) / 4.0
  return log_spec

BASE = pathlib.Path(__file__).parent.parent / "weights"
@functools.lru_cache(maxsize=None)
def get_encoding(n_vocab_in):
  download_file("https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken", BASE / "gpt2.tiktoken")
  ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in open(BASE / "gpt2.tiktoken") if line)}
  n_vocab = len(ranks)
  special_tokens = {}
  specials = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
    *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
  ]
  for token in specials:
    special_tokens[token] = n_vocab
    n_vocab += 1
  assert n_vocab == n_vocab_in
  import tiktoken
  return tiktoken.Encoding(
    name="alice",
    explicit_n_vocab=n_vocab,
    pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    mergeable_ranks=ranks,
    special_tokens=special_tokens,
  )
  
  