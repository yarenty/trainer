import os
import logging
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from trainer.config import LOCAL_MODEL_DIR, DEFAULT_MODEL_NAME

def main():
    model_name = DEFAULT_MODEL_NAME
    save_dir = LOCAL_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    logging.info(f"Downloading model '{model_name}' to '{save_dir}' (CPU-only)...")
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=False)
            logging.info("Model (AutoModelForCausalLM) downloaded and loaded.")
        except Exception as e:
            logging.warning(f"AutoModelForCausalLM failed: {e}. Trying AutoModel...")
            model = AutoModel.from_pretrained(model_name, local_files_only=False)
            logging.info("Model (AutoModel) downloaded and loaded.")
        model.save_pretrained(save_dir)
        logging.info("Model saved.")
    except Exception as e:
        logging.error(f"Error downloading model: {e}")

    logging.info(f"Downloading tokenizer for '{model_name}' to '{save_dir}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        tokenizer.save_pretrained(save_dir)
        logging.info("Tokenizer downloaded and saved.")
    except Exception as e:
        logging.error(f"Error downloading tokenizer: {e}")

if __name__ == "__main__":
    main() 