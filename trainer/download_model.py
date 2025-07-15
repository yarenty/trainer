from transformers import AutoModel, AutoTokenizer
import os
from trainer.config import LOCAL_MODEL_DIR, DEFAULT_MODEL_NAME

model_name = DEFAULT_MODEL_NAME
save_dir = LOCAL_MODEL_DIR

os.makedirs(save_dir, exist_ok=True)

print(f"Downloading model '{model_name}' to '{save_dir}'...")
try:
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print("Model downloaded and saved.")
except Exception as e:
    print(f"Error downloading model: {e}")

print(f"Downloading tokenizer for '{model_name}' to '{save_dir}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    print("Tokenizer downloaded and saved.")
except Exception as e:
    print(f"Error downloading tokenizer: {e}") 