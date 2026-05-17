
# Build Your Own Private, Zero-Cost AI Audio Transcriber

You finish a 30-minute project sync. Your calendar is already filling up with the next one. All you need is a clean summary of what was decided and who owns what.

So you look at the AI meeting tools. Then you look at the terms of service. Then you look at your legal team's face.

The commercial options all share the same architecture: your audio goes up to their cloud, gets transcribed on their servers, and gets summarised by their models. You pay $20–30 per user per month for the privilege of handing over your conversations. 

!!! note
    Here is an alternative. A 30-minute meeting, recorded as a 150MB WAV file, transcribed and summarised in **Less Than 150 seconds**, on a standard laptop (tested on MacBook with 8GB of RAM), with nothing — not a single byte — **leaving the machine**. 

!!! warning
    This article is for someone who wants to tinker with AI technology and wants to look under the hood. The focus is **engineering** - with obsession around resource usage optimisation and maintain strong security posture 
    
    That is what [magpie-auris](https://github.com/ayanguha/magpie/tree/main/auris) does. This article explains how it works and, more importantly, *why* the engineering decisions were made the way they were.

---

## Solution 

### Requirements

Our functional requirements are simple, clearly scoped:

- The transcriber will take an audio file as input
- It will generate raw transcription 
- Raw transcripts will be summarized. There can be multiple task_types which enforce distinct instructions and formats during summarization.
- Combine raw transcript and summary in a single markdowm output file. 

We also constrained by few non-functional requirements, namely:

- Memory footprint must be very light - not more than 3G at any given point-in-time 
- Good observability across elapsed time and CPU/memory utilisation must be captured and logged 
- Absolutely no cloud interaction (after initial set) during transcription process 

!!! note

    **Local model hosting**
    Air-gap model consumption is one of the most desired, yet oddly underserved, area of work. Leaving consumption-based commercial model aside, many of the model architectures are not supported on commodity hardwares. But the good news is, there are a long list of models which can be used locally. And **Ollama** is de-facto standard in local open source model management.  



### Solution 


#### Prefer Deterministic Execution - Always

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

This enables us to use well-understood engineering rigours 

- Standard dependency management using **uv** (This is a step-up from `pip` - if you start any new project, use `uv`)
- Testing made simpler - we are using **pytest** like we use it for any other deterministic systems
- Observability - The pipeline is instrumented from the start with a context manager that times every phase:

```python
class CodeBlockTimer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        logger.info(f"[{self.block_name}] completed in {elapsed:.2f} seconds.")
```

You can't optimise what you can't measure. The timer wraps every step, so you always know where time is going.

??? info 
    On an 8GB MacBook, a 30-minute meeting breaks down like this:

    | Phase | Time |
    |---|---|
    | Transcription (Whisper base, CPU) | ~100 seconds |
    | Summarisation (DeepSeek R1 1.5b) | ~50 seconds |
    | **Total** | **~150 seconds** |

**Prompts Are Configuration, Not Code**

By choosing determistic execution model also helps to manage AI related artefacts using common software engineering mechanisms. To highlight this point, we decided to develop a small prompting engine using a simple template routing system

```python
task_type_to_prompt_file = {
    "meeting_summary": "meeting_summary.md",
    "base": "base_template.md",
}
```

Prompt templates live in a `templates/` directory as plain markdown files. Switching the pipeline from "meeting summary" to a different output format — interview notes, design review, incident report — means writing a new template file, not touching the pipeline code.

!!! tip

    This is a small structural decision with a large practical payoff. The pipeline behaviour is controlled via configuration, with very structured approach without the unneccesary complexity introduced by agentic reasoning. Non-engineers can modify prompts without touching the code itself. Different use cases share the same infrastructure.

---

#### Small Models, Right-Scoped Tasks

The AI headlines are about the biggest models. GPT-4, DeepSeek 671B, Llama 70B. The implicit message is that smaller models are not worth serious consideration. It is important to note that even a modern small LLM model can generally handle large context and large text input very nicely. If you have text-only tasks then this area should be explored first before start throwing resource (and consequently money) at the problem.

DeepSeek R1 1.5b can easily summarises a 30-minute meeting transcript. The task is well-scoped: read a transcript, identify decisions, extract action items, write structured markdown. You do not need 671 billion parameters for that. You need a model that can reason carefully over a bounded text input — and R1's chain-of-thought architecture does that effectively even at 1.5b.

??? warning

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

!!! tip

    Remember — Smaller models are not compromise. It should always be a deliberate and careful matching of model size to task complexity.

---

#### Treat Memory as a Shared, Finite Resource

This is where many local AI solutions struggle. It is not about running `ollama` or `whisper` locally. It is about how to run both on the commodity hardware by optimizing available resources. 

The problem is that both models want a significant chunk of RAM at the same time, and 8GB unified memory is also shared with the OS, your browser, and everything else running. Left unmanaged, you will hit an out-of-memory crash mid-transcription on a long file. We address this with two explicit strategies.

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

`keep_alive=0` tells Ollama not to keep the model warm in memory between requests. By default, Ollama holds models in RAM for several minutes. On a constrained machine, that is memory you cannot afford. 

---


## Disclaimer: What You Are Actually Trading Off

`magpie` is not a intended to be cloud replacement for every use case. It is an useful little utility, showcasing how much can be achieved by embracing common AI engineering practices. 

Here are few points to note: 

**Whisper `base` is not perfect.** Accuracy drops on heavy accents, cross-talk, and noisy audio. The `small` model improves accuracy at the cost of more RAM and slower inference. 

**DeepSeek R1 1.5b is not GPT-4.** On a messy, jargon-heavy transcript it will occasionally miss nuance or misattribute a decision. For a well-structured meeting it performs well. For a chaotic all-hands with seven people talking over each other, set expectations accordingly.


!!! note

    What you are getting in return: complete data sovereignty, zero ongoing cost, and a tool that works offline. For many use cases — especially anything involving sensitive internal discussions — those are not minor benefits. They are the reason the tool exists.

---

## Getting Started

Start with [`magpie-auris README.md`](https://github.com/ayanguha/magpie/blob/main/auris/README.md)

---

## Key take away

Agentic AI is a genuinely powerful paradigm — and still rapidly evolving. But it is not the answer to everything.
A large class of real-world workloads are better served by traditional software engineering rigour: structured pipelines, deterministic execution, and common sense about resource constraints. The opportunity is not to choose between AI and engineering discipline — it is to apply both deliberately.
That is what this series is about. Not what AI can do in theory. What it can do when you treat it like an engineering problem.

---

*magpie is open source. The full source code, templates, and test suite are available at [github.com/ayanguha/magpie](https://github.com/ayanguha/magpie).*
