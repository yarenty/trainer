"""
train_sft.py: Utility for SFTTrainer-based training.

Provides a function to train a model using the trl SFTTrainer.
"""
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from trainer.utils.data_utils import load_and_tokenize_dataset
import torch

def train_sft(model_dir, data_dir, output_dir, config, device='cpu'):
    """
    Trains a model using the trl SFTTrainer.

    Args:
        model_dir (str): Path to the base model directory.
        data_dir (str): Path to the training data directory.
        output_dir (str): Path to save the fine-tuned model.
        config: Configuration object (optional).
        device (str): 'cpu' or 'gpu'.
    """
    logging.info(f"[SFT] Training model from {model_dir} with data in {data_dir}, output to {output_dir} on {device}.")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 2. Load and tokenize dataset
    logging.info("Loading and tokenizing dataset...")
    tokenized_dataset = load_and_tokenize_dataset(data_dir, tokenizer)

    # 3. SFTTrainer arguments
    training_args = {
        'per_device_train_batch_size': getattr(config, 'BATCH_SIZE', 4) if config else 4,
        'gradient_accumulation_steps': getattr(config, 'GRAD_ACCUM_STEPS', 1) if config else 1,
        'num_train_epochs': getattr(config, 'NUM_EPOCHS', 3) if config else 3,
        'learning_rate': getattr(config, 'LEARNING_RATE', 2e-4) if config else 2e-4,
        'logging_steps': 10,
        'output_dir': output_dir,
        'save_steps': 100,
        'save_total_limit': 2,
        'fp16': (device == 'gpu' and torch.cuda.is_available()),
        'bf16': False,
        'report_to': [],
        'remove_unused_columns': False,
        'overwrite_output_dir': True,
    }

    # 4. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
    )

    # 5. Train
    logging.info("Starting SFTTrainer training...")
    trainer.train()
    logging.info("Training complete. Saving fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Fine-tuned model saved to {output_dir}") 