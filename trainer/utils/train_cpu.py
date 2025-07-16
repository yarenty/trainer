import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import glob
import json

def train_cpu(model_dir, data_dir, output_dir, config):
    """
    Train the model on CPU using the dataset in data_dir. Save fine-tuned model to output_dir.
    Uses Hugging Face Trainer and transformers for Llama 3.1.
    """
    logging.info(f"[CPU] Training model from {model_dir} with data in {data_dir}, output to {output_dir}.")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=getattr(config, 'NUM_EPOCHS', 1) if config else 1,
        per_device_train_batch_size=getattr(config, 'BATCH_SIZE', 1) if config else 1,
        gradient_accumulation_steps=getattr(config, 'GRAD_ACCUM_STEPS', 1) if config else 1,
        learning_rate=getattr(config, 'LEARNING_RATE', 5e-5) if config else 5e-5,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # CPU only
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        overwrite_output_dir=True,
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