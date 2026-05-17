# Beyond the Hype: Build Your Own Private, Zero-Cost AI Meeting Assistant Locally

You finish a 30-minute project sync. Your calendar is already filling up with the next one. All you need is a clean summary of what was decided and who owns what.

So you look at the AI meeting tools. Then you look at the terms of service. Then you look at your legal team's face.

The commercial options all share the same architecture: your audio goes up to their cloud, gets transcribed on their servers, and gets summarised by their models. You pay $20–30 per user per month for the privilege of handing over your conversations.

Here is an alternative. A 30-minute meeting, recorded as a 150MB WAV file, transcribed and summarised in **150 seconds**, on a standard MacBook with 8GB of RAM, with nothing — not a single byte — leaving the machine.

That is what [magpie](https://github.com/ayanguha/magpie) does. This article explains how it works and, more importantly, *why* the engineering decisions were made the way they were.

---

## The Stack

- **OpenAI Whisper** (`base` model, weights stored locally) — transcription
- **DeepSeek R1 1.5b via Ollama** — summarisation and reasoning
- **pydub** — audio chunking
- **uv** — dependency management

No API keys. No cloud accounts. Runs air-gapped.

---

## Principle 1: Code Manages the System. The Model Does the Reasoning.

The AI hype cycle loves agents — autonomous systems that decide their own next steps, pick their own tools, manage their own loops. For well-scoped tasks, this is engineering overhead in search of a problem.

magpie uses a deliberately boring three-step pipeline:

```
1. Transcribe audio → raw transcript
2. Send transcript to LLM → markdown summary
3. Write output to file
```

Each step is a plain Python function. The pipeline is sequential and deterministic. The LLM never decides what to do next — it only does the one thing LLMs are genuinely good at: reasoning over text.

This is what that looks like in practice:

```python
def main():
    # Step 1: Transcription
    raw_transcript = transcribe_audio_step(audio_file_name)

    # Step 2: Summarization
    markdown_summary = summarize_transcript_step(raw_transcript, task_type)

    # Step 3: Output
    write_output_step(...)
```

Three functions. Linear execution. No loops, no tool selection, no retry logic. When something breaks, you know exactly where to look.

The pipeline is also instrumented from the start with a context manager that times every phase:

```python
class CodeBlockTimer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        logger.info(f"[{self.block_name}] completed in {elapsed:.2f} seconds.")
```

On an 8GB MacBook, a 30-minute meeting breaks down like this:

| Phase | Time |
|---|---|
| Transcription (Whisper base, CPU) | ~100 seconds |
| Summarisation (DeepSeek R1 1.5b) | ~50 seconds |
| **Total** | **~150 seconds** |

You can't optimise what you can't measure. The timer wraps every step, so you always know where time is going.

---

## Principle 2: Treat Memory as a Shared, Finite Resource

This is where most local AI tutorials quietly fail. They show you how to run Whisper. They show you how to run Ollama. They don't show you how to run both on the same 8GB machine without the OS killing one of them.

The problem is that both models want a significant chunk of RAM at the same time, and 8GB unified memory is also shared with the OS, your browser, and everything else running. Left unmanaged, you will hit an out-of-memory crash mid-transcription on a long file.

magpie handles this with two explicit strategies.

**Strategy 1: Audio chunking keeps RAM usage flat**

Rather than loading the entire audio file into memory, pydub slices it into 5-minute segments. Each chunk is exported to a temp file on disk, transcribed, then deleted before the next chunk is processed.

```python
# 5 minutes per chunk. Keeps RAM under ~50MB at any point.
CHUNK_LENGTH_MS = 5 * 60 * 1000

for i in range(0, total_duration_ms, CHUNK_LENGTH_MS):
    audio_chunk = audio[chunk_start:chunk_end]
    audio_chunk.export(TEMP_CHUNK_FILENAME, format="wav")
    result = model.transcribe(TEMP_CHUNK_FILENAME, fp16=False)
    full_transcript.append(result["text"].strip())

    # Clean up before next chunk
    os.remove(TEMP_CHUNK_FILENAME)
    del audio_chunk
```

The 5-minute window is not arbitrary — it keeps the in-memory audio footprint under ~50MB regardless of how long the original file is. A 2-hour meeting and a 30-minute meeting consume the same peak RAM.

**Strategy 2: Explicit model lifecycle management**

After transcription is complete, Whisper is not left sitting in memory. It is explicitly evicted:

```python
del model
logger.info("Full transcription complete. Whisper unloaded from RAM.")
```

Only after that does Ollama load DeepSeek. And once summarisation is done, DeepSeek is immediately unloaded too — via a single Ollama configuration option that most tutorials never mention:

```python
summary_response = ollama.generate(
    model=OLLAMA_MODEL,
    prompt=summary_prompt,
    options={"temperature": 0.0},
    keep_alive=0,  # Unloads model from RAM immediately after completion
)
```

`keep_alive=0` tells Ollama not to keep the model warm in memory between requests. By default, Ollama holds models in RAM for several minutes. On a constrained machine, that is memory you cannot afford. One line fixes it.

The result is a pipeline with a completely flat memory profile: Whisper loads, runs, unloads. DeepSeek loads, runs, unloads. They never compete for the same RAM.

---

## Principle 3: Small Models, Right-Scoped Tasks

The AI headlines are about the biggest models. GPT-4, DeepSeek 671B, Llama 70B. The implicit message is that smaller models are not worth serious consideration.

DeepSeek R1 1.5b summarises a 30-minute meeting transcript in 50 seconds on a MacBook CPU. That is not a compromise — it is a correct match of model size to task complexity.

The task is well-scoped: read a transcript, identify decisions, extract action items, write structured markdown. You do not need 671 billion parameters for that. You need a model that can reason carefully over a bounded text input — and R1's chain-of-thought architecture does that effectively even at 1.5b.

There is one practical wrinkle. DeepSeek R1 externalises its reasoning process in `<think>` tags that appear in the raw output:

```
<think>
Let me read through this transcript carefully. The participants seem to be discussing...
</think>

## Meeting Summary
...
```

If you do not handle this, your output file is full of the model's internal monologue. A two-line guard catches it:

```python
if "</think>" in raw_output:
    markdown_summary = raw_output.split("</think>")[-1].strip()
```

This is the kind of thing you only discover by running the model and reading the output carefully. It is not in the documentation.

---

## Principle 4: Prompts Are Configuration, Not Code

The pipeline supports multiple output formats through a simple template routing system:

```python
task_type_to_prompt_file = {
    "meeting_summary": "meeting_summary.md",
    "base": "base_template.md",
}
```

Prompt templates live in a `templates/` directory as plain markdown files. Switching the pipeline from "meeting summary" to a different output format — interview notes, design review, incident report — means writing a new template file, not touching the pipeline code.

This is a small structural decision with a large practical payoff. The pipeline behaviour is controlled by a file, not a code change. Non-engineers can modify prompts without touching Python. Different use cases share the same infrastructure.

---

## What You Are Actually Trading Off

Honesty matters here. This is not a cloud replacement for every use case.

**Whisper `base` is not perfect.** Accuracy drops on heavy accents, cross-talk, and noisy audio. The `small` model improves accuracy at the cost of more RAM and slower inference. The `large` model is not practical on 8GB.

**DeepSeek R1 1.5b is not GPT-4.** On a messy, jargon-heavy transcript it will occasionally miss nuance or misattribute a decision. For a well-structured meeting it performs well. For a chaotic all-hands with seven people talking over each other, set expectations accordingly.

**150 seconds is CPU inference.** If you have a Mac with Apple Silicon, Whisper can use the Neural Engine and get meaningfully faster. If you are on an older Intel MacBook, it will be slower. The number above is a real measurement, not a best-case benchmark.

What you are getting in return: complete data sovereignty, zero ongoing cost, and a tool that works offline. For many use cases — especially anything involving sensitive internal discussions — those are not minor benefits. They are the reason the tool exists.

---

## Getting Started

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/ayanguha/magpie.git
cd magpie
uv sync

# Install Ollama and pull the model
ollama pull deepseek-r1:1.5b

# Download Whisper base weights
curl -L https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -o models/base.pt

# Run
uv run python src/transcripting.py your-meeting.wav
```

The output is a markdown file with the summary at the top and the raw transcript below it — structured, timestamped, and ready to paste into a ticket or send to your team.

---

## The Point

The gap between "I saw a demo of this" and "I have a working tool on my laptop" is where most AI tutorials abandon you. They show the happy path on a cloud VM with 32GB of RAM and a GPU, then leave you to figure out why it crashes on your actual machine.

The engineering decisions in magpie — the chunk size, the `del model`, the `keep_alive=0`, the `<think>` guard — are all answers to problems you will hit in the real world. None of them are in the headline demos.

That is the point of this series. Not what AI can do in theory. What it can do on the hardware you actually have.

---

*magpie is open source. The full source code, templates, and test suite are available at [github.com/ayanguha/magpie](https://github.com/ayanguha/magpie).*
