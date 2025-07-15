import logging
import os
import subprocess
from trainer.config import MODELS_DIR, GGUF_MODEL, FINAL_OLLAMA, LLAMA_CPP_PATH, GGUF_QUANT_TYPE

def main():
    """
    Step 5: Quantize GGUF model using llama.cpp's llama-quantize tool.
    """
    gguf_dir = os.path.join(MODELS_DIR, GGUF_MODEL)
    input_gguf = os.path.join(gguf_dir, "model.gguf")
    quantized_gguf = os.path.join(gguf_dir, f"{FINAL_OLLAMA}.gguf")
    quantize_bin = os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize")
    quant_type = GGUF_QUANT_TYPE if 'GGUF_QUANT_TYPE' in globals() else "q4_k_m"

    # Check existence
    if not os.path.isfile(input_gguf):
        logging.error(f"Input GGUF file {input_gguf} does not exist. Run GGUF conversion first.")
        return
    if not os.path.isfile(quantize_bin):
        logging.error(f"llama-quantize binary not found at {quantize_bin}. Build llama.cpp with CMake.")
        return

    # Build command
    cmd = [
        quantize_bin,
        input_gguf,
        quantized_gguf,
        quant_type
    ]
    logging.info(f"Running GGUF quantization: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Quantized GGUF model written to {quantized_gguf}")
    except subprocess.CalledProcessError as e:
        logging.error(f"GGUF quantization failed: {e}")

if __name__ == "__main__":
    main() 