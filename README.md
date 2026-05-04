# Whisper-sanskrit

Transcribe **Sanskrit** (and other language) audio using [OpenAI Whisper](https://github.com/openai/whisper) — a general-purpose speech recognition model that supports 99+ languages including Sanskrit (`sa`).

---

## Requirements

- Python 3.8 or later
- [ffmpeg](https://ffmpeg.org/) installed and available on your `PATH`

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/dasadasanudas/Whisper-sanskrit.git
cd Whisper-sanskrit

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

## Usage

```
python transcribe.py <audio_file> [--model MODEL] [--language LANGUAGE] [--task TASK] [--output OUTPUT]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `audio` | *(required)* | Path to the audio file (mp3, wav, m4a, ogg, flac, …) |
| `--model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` |
| `--language` | `sa` | ISO 639-1 language code. Use `sa` for Sanskrit, or `auto` for automatic detection |
| `--task` | `transcribe` | `transcribe` keeps the original language; `translate` converts to English |
| `--output` | *(none)* | Optional path to save the transcription as a `.txt` file |

### Examples

```bash
# Transcribe a Sanskrit audio file using the default 'base' model
python transcribe.py audio.wav

# Use the larger, more accurate 'medium' model
python transcribe.py audio.mp3 --model medium

# Auto-detect language
python transcribe.py audio.mp3 --language auto

# Transcribe and save result to a file
python transcribe.py audio.wav --model small --output result.txt

# Translate Sanskrit audio to English
python transcribe.py audio.wav --task translate
```

---

## Model sizes

| Model | Parameters | English-only | Multilingual | Required VRAM | Relative speed |
|---|---|---|---|---|---|
| `tiny` | 39 M | ✅ | ✅ | ~1 GB | ~32× |
| `base` | 74 M | ✅ | ✅ | ~1 GB | ~16× |
| `small` | 244 M | ✅ | ✅ | ~2 GB | ~6× |
| `medium` | 769 M | ✅ | ✅ | ~5 GB | ~2× |
| `large` | 1550 M | ❌ | ✅ | ~10 GB | 1× |

For Sanskrit transcription, `small` or `medium` is recommended for a good balance of accuracy and speed.

---

## Notes

- The first run for a given model size will download the model weights automatically (~74 MB for `base`).
- A GPU (CUDA) will be used automatically if available, otherwise the model runs on CPU.
- Sanskrit (`sa`) is supported by all multilingual Whisper models.
