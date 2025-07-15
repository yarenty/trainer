import logging
import os
import subprocess
from trainer.config import MODELS_DIR, MERGED_MODEL, GGUF_MODEL, LLAMA_CPP_PATH

def main():
    """
    Step 4: Convert merged Hugging Face model to GGUF format using llama.cpp's convert_hf_to_gguf.py.
    """
    merged_model_dir = os.path.join(MODELS_DIR, MERGED_MODEL)
    gguf_output_dir = os.path.join(MODELS_DIR, GGUF_MODEL)
    os.makedirs(gguf_output_dir, exist_ok=True)
    gguf_outfile = os.path.join(gguf_output_dir, "model.gguf")
    convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")

    # Check existence
    if not os.path.exists(merged_model_dir):
        logging.error(f"Merged model directory {merged_model_dir} does not exist. Run previous steps first.")
        return
    if not os.path.isfile(convert_script):
        logging.error(f"convert_hf_to_gguf.py not found at {convert_script}. Check your llama.cpp path.")
        return

    # Build command
    cmd = [
        "python3",
        convert_script,
        merged_model_dir,
        "--outfile", gguf_outfile,
        "--outtype", "f16"
    ]
    logging.info(f"Running GGUF conversion: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"GGUF model written to {gguf_outfile}")
    except subprocess.CalledProcessError as e:
        logging.error(f"GGUF conversion failed: {e}")

if __name__ == "__main__":
    main() 