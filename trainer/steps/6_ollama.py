"""
Step 6: Create Ollama Modelfile and Import Instructions

Creates a Modelfile for Ollama and prints import instructions. Uses
config.py for all settings.
"""
import logging
import os
from trainer.config import MODELS_DIR, GGUF_MODEL, FINAL_OLLAMA

def main():
    """
    Creates a Modelfile for Ollama and prints import instructions. Logs
    errors if the quantized GGUF file is missing.
    """
    gguf_dir = os.path.join(MODELS_DIR, GGUF_MODEL)
    quantized_gguf = os.path.join(gguf_dir, f"{FINAL_OLLAMA}.gguf")
    modelfile_path = os.path.join(gguf_dir, "Modelfile")

    if not os.path.isfile(quantized_gguf):
        logging.error(
            f"Quantized GGUF file {quantized_gguf} does not exist. Run quantization first."
        )
        return

    # Write Modelfile
    modelfile_content = (
        f"FROM {FINAL_OLLAMA}.gguf\n"
        f'PARAMETER stop "### Instruction:"\n'
        f'PARAMETER stop "### Response:"\n'
        f'PARAMETER stop "### End"\n'
    )
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    logging.info(f"Modelfile created at {modelfile_path}")

    # Print Ollama import instructions
    print("\n--- Ollama Import Instructions ---")
    print(f"To import your model into Ollama, run:")
    print(f"ollama create {FINAL_OLLAMA} -f {modelfile_path}")
    print(f"\nAfter import, you can run your model with:")
    print(f"ollama run {FINAL_OLLAMA}")
    print(
        "Example query: ollama run {FINAL_OLLAMA} "
        '"### Instruction:\nHow do I use the filter operation in DataFusion?\n\n### Response:"'
    )

if __name__ == "__main__":
    main() 