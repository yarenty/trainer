import logging
import os
import json
from trainer.config import MODELS_DIR, GGUF_MODEL, FINAL_OLLAMA, DATA_DIR

def main():
    """
    Step 7: Evaluate the quantized model for Datafusion QA and 4GB GPU usability.
    """
    gguf_dir = os.path.join(MODELS_DIR, GGUF_MODEL)
    quantized_gguf = os.path.join(gguf_dir, f"{FINAL_OLLAMA}.gguf")
    qa_data_dir = DATA_DIR

    if not os.path.isfile(quantized_gguf):
        logging.error(f"Quantized GGUF file {quantized_gguf} does not exist. Run quantization first.")
        return

    # Try to load a few test questions from the QA dataset
    import glob
    qa_files = glob.glob(os.path.join(qa_data_dir, "**", "*.jsonl"), recursive=True)
    test_questions = []
    for file in qa_files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'question' in item:
                        test_questions.append(item['question'])
                    if len(test_questions) >= 3:
                        break
                except Exception as e:
                    continue
        if len(test_questions) >= 3:
            break
    if not test_questions:
        logging.warning("No test questions found in QA dataset.")

    # Direct GGUF inference is not supported in Python (as of now)
    print("\n--- Evaluation Instructions ---")
    print(f"To evaluate your quantized model, you can use Ollama or llama.cpp CLI tools.")
    print(f"Example (Ollama):")
    print(f"ollama run {FINAL_OLLAMA} \"### Instruction:\\n{test_questions[0] if test_questions else 'How do I use the filter operation in DataFusion?'}\\n\\n### Response:\"")
    print(f"\nIf you want to test with llama.cpp, use the main executable and pass the GGUF file:")
    print(f"./main -m {quantized_gguf} --prompt '### Instruction: {test_questions[0] if test_questions else 'How do I use the filter operation in DataFusion?'}\n\n### Response:'")
    print("\nMonitor GPU memory usage with nvidia-smi or similar tools to ensure it fits on a 4GB GPU.")
    print("\nYou can also script batch evaluation using the llama.cpp API or Ollama API if desired.")
    logging.info("Evaluation instructions printed.")

if __name__ == "__main__":
    main() 