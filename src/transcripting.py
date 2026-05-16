import logging
import os
import sys
import time
import warnings
import ollama
import whisper
from pydub import AudioSegment

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MeetingProcessor")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
AUDIO_FILE_PATH = "input/sample-speech-10m.wav"
OUTPUT_FILE_PATH = "output/transcript_summary_output.md"
TEMP_CHUNK_FILENAME = "output/temp_whisper_chunk.wav"
OLLAMA_MODEL = "deepseek-r1:1.5b"
WHISPER_MODEL_PATH = "models/base.pt"
PROMPT_TEMPLATES_DIR = "templates"
TASK_TYPE = "base"  # Default task type for prompt selection

# 5 minutes per chunk (In milliseconds).
# This keeps the maximum audio file size in RAM under ~50MB at any time.
CHUNK_LENGTH_MS = 5 * 60 * 1000


class CodeBlockTimer:
    """A clean context manager to time specific operational blocks."""

    def __init__(self, block_name):
        self.block_name = block_name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        logger.info(
            f"[{self.block_name}] completed execution in {elapsed:.2f} seconds."
        )


def transcribe_large_audio(file_path: str, model_path: str) -> str:
    """Loads Whisper once, then streams and transcribes the audio file

    in 5-minute increments to keep RAM usage perfectly flat.
    """
    logger.info("Loading local Whisper weights into RAM...")
    model = whisper.load_model(model_path)

    logger.info(f"Opening audio stream for: {file_path}")
    audio = AudioSegment.from_file(file_path)
    total_duration_ms = len(audio)

    full_transcript = []
    

    # Process the file in sequential increments
    for i in range(0, total_duration_ms, CHUNK_LENGTH_MS):
        chunk_start = i
        chunk_end = min(i + CHUNK_LENGTH_MS, total_duration_ms)

        chunk_minutes_start = chunk_start // 60000
        chunk_minutes_end = chunk_end // 60000
        logger.info(
            f"Transcribing segment: {chunk_minutes_start}m to {chunk_minutes_end}m..."
        )

        # Extract chunk and export to disk
        audio_chunk = audio[chunk_start:chunk_end]
        audio_chunk.export(TEMP_CHUNK_FILENAME, format="wav")

        # Core transcription block
        result = model.transcribe(TEMP_CHUNK_FILENAME, fp16=False)
        full_transcript.append(result["text"].strip())

        # Clean up disk footprint
        if os.path.exists(TEMP_CHUNK_FILENAME):
            os.remove(TEMP_CHUNK_FILENAME)

        del audio_chunk

    # Unload Whisper from memory entirely
    del model
    logger.info("Full transcription complete. Whisper unloaded from RAM.")

    return " ".join(full_transcript)


def transcribe_audio_step(audio_file_name: str = AUDIO_FILE_PATH) -> str:
    """Step 1: Run the segmented transcription loop."""
    try:
        with CodeBlockTimer("Audio Transcription Phase"):
            raw_transcript = transcribe_large_audio(
                audio_file_name, WHISPER_MODEL_PATH
            )
        return raw_transcript
    except Exception as e:
        logger.error(f"Error during transcription pipeline: {e}")
        return None

def load_prompts_by_task_type(task_type: str = "base") -> dict:
    """Utility function to load different prompt templates based on task type."""
    task_type_to_prompt_file = {
        "meeting_summary": "meeting_summary.md",
        "base": "base_template.md",
    }
    prompt_file = task_type_to_prompt_file.get(task_type, "base_template.md")
    prompt_path = os.path.join(PROMPT_TEMPLATES_DIR, prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt

def summarize_transcript_step(raw_transcript: str, task_type: str) -> str:
    """Step 2: Markdown Summarization via Ollama."""
    logger.info(
        f"Sending transcript text to {OLLAMA_MODEL} for Markdown summarization..."
    )

    summary_prompt = load_prompts_by_task_type(task_type).replace("{raw_transcript}", raw_transcript)

    
    try:
        with CodeBlockTimer("Ollama DeepSeek Summary Generation"):
            summary_response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=summary_prompt,
                options={"temperature": 0.0},
                keep_alive=0,  # Unloads the model from RAM immediately after completion
            )
            raw_output = summary_response.get("response", "").strip()

            # Handle DeepSeek reasoning thinking tags cleanly
            if "</think>" in raw_output:
                markdown_summary = raw_output.split("</think>")[-1].strip()
            else:
                markdown_summary = raw_output

        logger.info("Summary processing complete.")
        return markdown_summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return None


def write_output_step(audio_file_name: str = AUDIO_FILE_PATH, 
                      output_file_name: str = OUTPUT_FILE_PATH, 
                      raw_transcript: str = "",
                      markdown_summary: str = "",
                      task_type: str = TASK_TYPE) -> bool:
    
    """Step 3: Write Output to file."""
    logger.info(f"Writing final files to {OUTPUT_FILE_PATH}...")
    file_content = f"""# Audio Processing 
**Source File:** `{audio_file_name}`
**Generated on:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Task Type:** {task_type}
---

##  Summary
{markdown_summary}

---

## Raw Transcript
{raw_transcript}
"""

    try:
        with open(output_file_name, "w", encoding="utf-8") as f:
            f.write(file_content)
        logger.info(f"Success! Your meeting logs are saved to {OUTPUT_FILE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        return False


def main():
    total_pipeline_start = time.perf_counter()
    audio_file_name = sys.argv[1] if len(sys.argv) > 1 else AUDIO_FILE_PATH
    output_file_name = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE_PATH
    task_type = sys.argv[3] if len(sys.argv) > 3 else TASK_TYPE
    if not os.path.exists(audio_file_name):
        logger.error(f"Audio file not found at: {audio_file_name}")
        return

    # Step 1: Transcription
    raw_transcript = transcribe_audio_step(audio_file_name)
    if not raw_transcript or raw_transcript.isspace():
        logger.warning("Resulting transcript is empty.")
        return

    # Step 2: Summarization
    markdown_summary = summarize_transcript_step(raw_transcript, task_type)
    if not markdown_summary:
        return

    # Step 3: Output
    success = write_output_step(audio_file_name=audio_file_name, 
                                output_file_name=output_file_name, 
                                raw_transcript=raw_transcript, 
                                markdown_summary=markdown_summary,
                                task_type=task_type)
    
    if success:
        total_pipeline_end = time.perf_counter()
        total_elapsed = total_pipeline_end - total_pipeline_start
        logger.info(f"Total workflow execution time: {total_elapsed:.2f} seconds.")



if __name__ == "__main__":
    main()