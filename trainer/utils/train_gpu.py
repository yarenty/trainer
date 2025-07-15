import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import glob
import json
import torch

def train_gpu(model_dir, data_dir, output_dir, config):
    """
    Train the model on GPU using the dataset in data_dir. Save fine-tuned model to output_dir.
    Uses Hugging Face Trainer and transformers for Llama 3.1. Enables mixed precision and larger batch size.
    """
    logging.info(f"[GPU] Training model from {model_dir} with data in {data_dir}, output to {output_dir}.")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 2. Load and preprocess dataset (multi-level jsonl)
    logging.info("Loading and preprocessing dataset...")
    data_files = glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)
    if not data_files:
        logging.error(f"No .jsonl files found in {data_dir}")
        return
    # Load all QA pairs into a list
    samples = []
    for file in data_files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    samples.append({
                        'question': item.get('question', ''),
                        'answer': item.get('answer', '')
                    })
                except Exception as e:
                    logging.warning(f"Skipping line due to error: {e}")
    if not samples:
        logging.error("No valid samples found in dataset.")
        return
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(samples)

    # 3. Tokenize dataset
    def preprocess(example):
        prompt = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
        return tokenizer(prompt, truncation=True, max_length=2048, padding='max_length')
    logging.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess, batched=False)

    # 4. Training arguments (use config if available, else defaults)
    # Use larger batch size and enable mixed precision if available
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=getattr(config, 'NUM_EPOCHS', 3) if config else 3,
        per_device_train_batch_size=getattr(config, 'BATCH_SIZE', 4) if config else 4,
        gradient_accumulation_steps=getattr(config, 'GRAD_ACCUM_STEPS', 1) if config else 1,
        learning_rate=getattr(config, 'LEARNING_RATE', 2e-4) if config else 2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=[],
        remove_unused_columns=False,
        overwrite_output_dir=True,
        optim="adamw_torch",
        dataloader_num_workers=2,
        torch_compile=True if torch.__version__ >= '2.0.0' else False,
    )

    # 5. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 7. Train
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete. Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Fine-tuned model saved to {output_dir}") 