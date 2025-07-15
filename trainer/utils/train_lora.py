"""
train_lora.py: Utility for LoRA/PEFT training.

Provides a function to train a model using LoRA/PEFT adapters.
"""
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trainer.utils.data_utils import load_and_tokenize_dataset
import torch

def train_lora(model_dir, data_dir, output_dir, config, device='cpu'):
    """
    Trains a model using LoRA/PEFT adapters.

    Args:
        model_dir (str): Path to the base model directory.
        data_dir (str): Path to the training data directory.
        output_dir (str): Path to save the adapters.
        config: Configuration object (optional).
        device (str): 'cpu' or 'gpu'.
    """
    logging.info(f"[LoRA] Training model from {model_dir} with data in {data_dir}, output to {output_dir} on {device}.")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 2. Prepare model for LoRA/PEFT
    logging.info("Preparing model for LoRA/PEFT training...")
    if device == 'gpu' and torch.cuda.is_available():
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=getattr(config, 'LORA_R', 8) if config else 8,
        lora_alpha=getattr(config, 'LORA_ALPHA', 16) if config else 16,
        lora_dropout=getattr(config, 'LORA_DROPOUT', 0.05) if config else 0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 3. Load and tokenize dataset
    logging.info("Loading and tokenizing dataset...")
    tokenized_dataset = load_and_tokenize_dataset(data_dir, tokenizer)

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=getattr(config, 'NUM_EPOCHS', 3) if config else 3,
        per_device_train_batch_size=getattr(config, 'BATCH_SIZE', 4) if config else 4,
        gradient_accumulation_steps=getattr(config, 'GRAD_ACCUM_STEPS', 1) if config else 1,
        learning_rate=getattr(config, 'LEARNING_RATE', 2e-4) if config else 2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=(device == 'gpu' and torch.cuda.is_available()),
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        overwrite_output_dir=True,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 6. Train
    logging.info("Starting LoRA/PEFT training...")
    trainer.train()
    logging.info("Training complete. Saving LoRA adapters...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"LoRA adapters saved to {output_dir}") 