from transformers import AutoModel, AutoTokenizer
import os

model_name = "unsloth/Qwen2-7B-Instruct-bnb-4bit"
save_dir = "./qwen2"

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