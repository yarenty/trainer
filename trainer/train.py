
import os
import torch
import logging
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import glob
from transformers import AutoModel, AutoTokenizer
from trainer.config import DATA_DIR, LOCAL_MODEL_DIR, MODELS_DIR, FINE_TUNED_MODEL

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Model name from Hugging Face (using a 4-bit quantized model for efficiency)
model_name = LOCAL_MODEL_DIR  # Load from local directory
# Directory containing all .jsonl files for training (recursively)
data_root = DATA_DIR
# Directory to save the fine-tuned model adapters
output_dir = "{MODELS_DIR}/{FINE_TUNED_MODEL}"
# Maximum sequence length the model can handle
max_seq_length = 2048
# Number of training epochs
num_train_epochs = 3
# Batch size per GPU
per_device_train_batch_size = 2
# Number of gradients to accumulate before updating weights
gradient_accumulation_steps = 4
# Learning rate for the optimizer
learning_rate = 2e-4
# LoRA rank (dimension of the low-rank matrices)
lora_r = 64
# LoRA alpha (scaling factor)
lora_alpha = 16
# LoRA dropout probability
lora_dropout = 0.1

def main():
    """
    Main function to run the fine-tuning pipeline.
    1.  Load a pre-trained model and tokenizer.
    2.  Configure the model for LoRA (Low-Rank Adaptation).
    3.  Load and format the dataset.
    4.  Set up the training process using SFTTrainer.
    5.  Start the training.
    6.  Save the final model adapters.
    """
    logging.info("--- Starting Fine-Tuning ---")
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

    # --- 1. Load Model and Tokenizer ---
    logging.info(f"Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Let unsloth handle dtype
        load_in_4bit=True,
    )
    logging.info("Model and tokenizer loaded successfully.")

    # --- 2. Configure LoRA ---
    logging.info("Configuring model for LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        max_seq_length=max_seq_length,
    )
    logging.info("LoRA configured successfully.")

    # --- 3. Load and Format Dataset ---
    def format_chat_template(row):
        """Formats a row using the Qwen2 chat template."""
        row["text"] = f"<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n{row['answer']}<|im_end|>"
        return row

    # Recursively find all .jsonl files under output/
    data_files = glob.glob(os.path.join(data_root, "**", "*.jsonl"), recursive=True)
    if not data_files:
        raise FileNotFoundError(f"No .jsonl files found in directory tree: {data_root}")

    logging.info(f"Loading and formatting dataset from: {data_files}")
    dataset = load_dataset("json", data_files={"train": data_files}, split="train")
    dataset = dataset.map(format_chat_template, num_proc=4)
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
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
        ),
    )
    logging.info("Trainer setup complete.")

    # --- 5. Train the model ---
    logging.info("--- Starting model training ---")
    trainer.train()
    logging.info("--- Model training finished ---")

    # --- 6. Save the final model ---
    logging.info(f"Saving final model adapters to: {output_dir}")
    trainer.save_model(output_dir)
    logging.info("Model saved successfully.")
    logging.info("--- Fine-Tuning Complete ---")

if __name__ == "__main__":
    main()
