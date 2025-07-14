import os
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# IMPORTANT: This script is for CPU training and will be VERY SLOW.
# It is recommended to run the GPU version (`train.py`) on a CUDA-enabled machine if possible.

# Model name from Hugging Face (using a non-gated model).
model_name = "Qwen/Qwen2-7B-Instruct"
# Path to the JSONL dataset
dataset_path = "output/datafusion-functions-json/docs_qa.jsonl"
# Directory to save the fine-tuned model adapters
output_dir = "models/qwen2-7b-datafusion-instruct-cpu"
# Maximum sequence length
max_seq_length = 2048
# Training parameters
num_train_epochs = 1  # Reduced for CPU training feasibility
per_device_train_batch_size = 1  # Reduced for CPU memory
gradient_accumulation_steps = 4
learning_rate = 2e-4
# LoRA configuration
lora_r = 16 # Reduced for CPU
lora_alpha = 32 # Reduced for CPU
lora_dropout = 0.1

def main():
    """
    Main function to run the CPU fine-tuning pipeline.
    """
    logging.info("--- Starting CPU Fine-Tuning ---")
    logging.warning("Training on CPU is extremely slow. This may take many hours or days.")

    # --- 1. Load Model and Tokenizer ---
    logging.info(f"Loading base model: {model_name}")
    # Qwen models require trusting remote code
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32, # Use full precision for CPU
        trust_remote_code=True,
    )
    logging.info("Model and tokenizer loaded successfully.")

    # --- 2. Configure LoRA ---
    logging.info("Configuring model for LoRA...")
    # Target modules are often specific to the model architecture.
    # These are common for Qwen2 models.
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    logging.info("LoRA configured successfully.")
    model.print_trainable_parameters()


    # --- 3. Load and Format Dataset ---
    def format_chat_template(row):
        """Formats a row using the Qwen2 chat template."""
        row["text"] = f"<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n{row['answer']}<|im_end|>"
        return row

    logging.info(f"Loading and formatting dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(format_chat_template, num_proc=2)
    logging.info(f"Dataset loaded. Number of examples: {len(dataset)}")

    # --- 4. Set up Trainer ---
    logging.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_torch", # Standard AdamW optimizer
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            no_cuda=True, # Explicitly disable CUDA
        ),
    )
    logging.info("Trainer setup complete.")

    # --- 5. Train the model ---
    logging.info("--- Starting model training on CPU ---")
    trainer.train()
    logging.info("--- Model training finished ---")

    # --- 6. Save the final model ---
    logging.info(f"Saving final model adapters to: {output_dir}")
    trainer.save_model(output_dir)
    logging.info("Model saved successfully.")
    logging.info("--- CPU Fine-Tuning Complete ---")

if __name__ == "__main__":
    main()
