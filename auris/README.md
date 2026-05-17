# Meeting Processor

This project transcribes large audio files using OpenAI's Whisper and generates professional Markdown summaries using DeepSeek via Ollama.

## Prerequisites

### 1. Install FFmpeg
The project uses `pydub` for audio manipulation, which requires FFmpeg to be installed on your system.

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download and install from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to your system PATH.

## Setup and Installation

### 1. Install Dependencies
This project uses `uv` for fast, reliable dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the project environment
uv sync
```

### 2. Setup Ollama and Model
The summarization phase uses the DeepSeek model via Ollama.

1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Pull the required model:
```bash
ollama pull deepseek-r1:1.5b
```

### 3. Download Whisper Weights
The transcription phase requires the Whisper `base` model weights. Download the `base.pt` file and place it in the project root:

```bash
curl -L https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -o base.pt
```

### 4. Download Sample Files for Testing
You can use these sample audio files to test the pipeline:

```bash
# 10-minute sample
curl -L https://samplelib.com/wav/sample-speech-10m.wav -o input/sample-speech-10m.wav

# 30-minute sample
curl -L https://samplelib.com/wav/sample-speech-30m.wav -o input/sample-speech-30m.wav
```


## Usage

Run the transcription and summarization pipeline by providing an audio file path:

```bash
uv run python transcripting.py your-audio-file.wav
```

By default, the output will be saved to `transcript_summary_output.md`.

### Custom Output File
You can specify a custom output file path as the second argument:

```bash
uv run python transcripting.py your-audio-file.wav custom-output.md
```

## Testing

To verify the pipeline is working correctly, you can run the baseline test which uses `sample-speech-10m.wav` to ensure transcription and summarization are functioning as expected.

```bash
# Run using pytest
uv run pytest test_baseline.py

# Or run directly with python
uv run python test_baseline.py
```


