import pytest
import os
import logging
import sys
from src.transcripting import main as run_pipeline

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestBaseline")

def test_transcription_pipeline():
    """
    Baseline test to ensure the transcription and summarization pipeline 
    works end-to-end with harvard.wav.
    """
    # Setup
    input_audio = "input/sample-speech-10m.wav"
    output_file = "output/output.md"
    
    # Ensure input file exists
    if not os.path.exists(input_audio):
        pytest.fail(f"Required test file {input_audio} not found. Please ensure it is in the input/ folder.")

    # Clean up previous output if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Mock sys.argv to simulate command line arguments: [script_name, input_file, output_file]
    original_argv = sys.argv
    try:
        sys.argv = ["transcripting.py", input_audio, output_file]
        
        logger.info("Starting baseline pipeline test...")
        run_pipeline()
        logger.info("Pipeline execution completed.")

        # Assertions
        assert os.path.exists(output_file), f"Output file {output_file} was not created."
        
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        assert "**Source File:**" in content, "File name missing from output."
        assert "##  Summary" in content, "Output file is missing the Executive Summary section."
        assert "## Raw Transcript" in content, "Output file is missing the Raw Transcript section."
        assert len(content) > 100, "Output file content is unexpectedly short."
        
        logger.info("Baseline test passed successfully!")

    finally:
        # Restore original sys.argv
        sys.argv = original_argv
        # Destroy stage: delete output file
        if os.path.exists(output_file):
            os.remove(output_file)
            logger.info(f"Cleaned up {output_file}")

def test_performance_pipeline():
    """
    Performance test to ensure the pipeline processes a 30m audio file 
    in less than 5 minutes.
    """
    import time

    input_audio = "input/sample-speech-30m.wav"
    output_file = "output/perf_output.md"
    time_limit_seconds = 5 * 60

    if not os.path.exists(input_audio):
        pytest.fail(f"Required test file {input_audio} not found.")

    if os.path.exists(output_file):
        os.remove(output_file)

    original_argv = sys.argv
    try:
        sys.argv = ["transcripting.py", input_audio, output_file]
        
        logger.info("Starting performance pipeline test (30m audio)...")
        start_time = time.perf_counter()
        run_pipeline()
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        logger.info(f"Performance test completed in {elapsed_time:.2f} seconds.")

        assert os.path.exists(output_file), "Output file was not created."
        assert elapsed_time < time_limit_seconds, f"Pipeline took too long: {elapsed_time:.2f}s (limit: {time_limit_seconds}s)"
        
        logger.info("Performance test passed successfully!")

    finally:
        sys.argv = original_argv
        # Destroy stage: delete output file
        if os.path.exists(output_file):
            os.remove(output_file)
            logger.info(f"Cleaned up {output_file}")

if __name__ == "__main__":
    # Allow running the test directly via python test_baseline.py
    try:
        test_transcription_pipeline()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
