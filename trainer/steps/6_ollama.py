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
    modelfile_content = f'''FROM {FINAL_OLLAMA}.gguf

# System prompt for all sessions
SYSTEM """You are a helpful, concise, and accurate coding assistant specialized in Rust and the DataFusion SQL engine. Always provide high-level, idiomatic Rust code, DataFusion SQL examples, clear documentation, and robust test cases. Your answers should be precise, actionable, and end with '### End'."""

# Prompt template (optional, but recommended for instruct models)
TEMPLATE """### Instruction:
{{{{ .Prompt }}}}

### Response:
"""

# Stop sequences to end generation
PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
PARAMETER stop "### End"

# Generation parameters to prevent infinite loops
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.2
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# Metadata for public sharing (for reference only)
# TAGS ["llama3", "datafusion", "qa", "rust", "sql", "public"]
# DESCRIPTION "A fine-tuned LLM specialized in Rust and DataFusion (SQL engine) Q&A. Produces idiomatic Rust code, DataFusion SQL examples, clear documentation, and robust test cases, with robust stop sequences and infinite loop prevention."

LICENSE "apache-2.0"
'''
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