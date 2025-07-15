import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if len(sys.argv) != 3:
    print("Usage: python dequantize_and_save_fp16.py <input_dir> <output_dir>")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.isdir(input_dir):
    print(f"Error: input_dir {input_dir} does not exist or is not a directory.")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

print(f"Loading model from {input_dir}...")
try:
    model = AutoModelForCausalLM.from_pretrained(input_dir, torch_dtype=torch.float16)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print(f"Saving model in float16 to {output_dir}...")
try:
    model.save_pretrained(output_dir)
    print("Model saved.")
except Exception as e:
    print(f"Error saving model: {e}")
    sys.exit(1)

print(f"Copying tokenizer from {input_dir} to {output_dir}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer saved.")
except Exception as e:
    print(f"Error saving tokenizer: {e}")
    sys.exit(1)

print("Dequantization and save complete.") 