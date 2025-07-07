import os
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import json
import time

def load_and_prepare_datasets(data_dir: str) -> DatasetDict:
    """
    Loads all generated JSONL datasets from the data_dir and combines them.
    Filters out entries with error messages.
    """
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_qa.jsonl"):
            file_path = os.path.join(data_dir, filename)
            print(f"Loading data from {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        # Filter out entries that indicate an error in Q&A generation
                        if "Error: Could not generate question." not in entry.get("question", ""):
                            all_data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed JSON line in {filename}: {line.strip()} - {e}")
    
    if not all_data:
        raise ValueError(f"No valid data found in {data_dir}. Please ensure _qa.jsonl files exist and are correctly formatted.")

    # Convert list of dicts to Hugging Face Dataset
    full_dataset = Dataset.from_list(all_data)

    # Split into train and validation sets
    # Adjust test_size as needed, e.g., 0.1 for 10% validation
    train_test_split = full_dataset.train_test_split(test_size=0.05, seed=42)
    
    return DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })

def format_instruction(sample):
    """
    Formats the dataset into an instruction-tuning format.
    Adjust this based on how your LLM expects input.
    """
    return f"### Instruction:\n{sample['question']}\n\n### Response:\n{sample['answer']}"

def main():
    # --- Configuration ---
    print("\n========== [PHASE 1: CONFIGURATION] ==========")
    start_time = time.time()
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct" #"meta-llama/Llama-2-7b-hf" # Replace with Llama 3.2 when available on HF or use a local path
    # Ensure you have access to Llama 3.2 on Hugging Face or download it locally.
    # For Llama 3, you might need to use "meta-llama/Meta-Llama-3-8B" or similar.
    # You'll need to authenticate with Hugging Face if using a gated model.

    data_dir = "/opt/ml/trainer/data"
    output_dir = "/opt/ml/trainer/models/llama3_datafusion_finetuned"
    
    # QLoRA configuration
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    
    # Training arguments
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    num_train_epochs = 3
    learning_rate = 2e-4
    fp16 = True # Set to False if your GPU doesn't support FP16/BF16
    logging_steps = 10
    save_steps = 500
    eval_steps = 500
    
    # --- Load Data ---
    print("\n========== [PHASE 2: DATA LOADING & PREPARATION] ==========")
    data_load_start = time.time()
    print("Loading and preparing datasets...")
    raw_datasets = load_and_prepare_datasets(data_dir)
    print(f"Train dataset size: {len(raw_datasets['train'])}")
    print(f"Validation dataset size: {len(raw_datasets['validation'])}")
    print(f"Data loading completed in {time.time() - data_load_start:.2f} seconds.")

    # --- Load Tokenizer and Model ---
    print("\n========== [PHASE 3: MODEL & TOKENIZER LOADING] ==========")
    model_load_start = time.time()
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token for causal LMs
    print("Tokenizer loaded.")

    # Quantization configuration (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    print("Loading model with quantization config...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # Automatically maps model to available devices (e.g., GPU)
        torch_dtype=torch.float16, # Use float16 for faster training if supported
    )
    print("Model loaded.")
    model.config.use_cache = False # Disable cache for training
    model.config.pretraining_tp = 1 # For Llama models, helps with tensor parallelism
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training.")
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    print("LoRA model prepared:")
    model.print_trainable_parameters()
    print(f"Model and tokenizer loading completed in {time.time() - model_load_start:.2f} seconds.")

    # --- Tokenize Data ---
    print("\n========== [PHASE 4: TOKENIZATION] ==========")
    tokenization_start = time.time()
    print("Tokenizing datasets...")
    def tokenize_function(examples):
        # Apply formatting and then tokenize
        return tokenizer(
            [format_instruction(sample) for sample in examples],
            truncation=True,
            max_length=512, # Adjust max_length based on your data and model context window
            padding="max_length", # Pad to max_length
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names # Remove original columns
    )
    print("Tokenization complete.")
    print(f"Tokenized train dataset: {len(tokenized_datasets['train'])} samples")
    print(f"Tokenized validation dataset: {len(tokenized_datasets['validation'])} samples")
    print(f"Tokenization completed in {time.time() - tokenization_start:.2f} seconds.")

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_total_limit=2, # Only keep the last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none", # Disable reporting to W&B or other platforms for simplicity
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    print("Trainer initialized.")

    # --- Start Training ---
    print("\n========== [PHASE 6: TRAINING] ==========")
    training_start = time.time()
    print("Starting training...")
    trainer.train()
    print(f"Training completed in {time.time() - training_start:.2f} seconds.")

    # --- Save Fine-tuned Model ---
    print("\n========== [PHASE 7: SAVING MODEL] ==========")
    save_start = time.time()
    print("Saving fine-tuned model adapters...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model adapters saved to {output_dir}")
    print(f"Model saving completed in {time.time() - save_start:.2f} seconds.")

    print("\n========== [ALL PHASES COMPLETE] ==========")
    print(f"Total script runtime: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Ensure you have a GPU and necessary drivers installed for this to run efficiently.
    # Also, ensure you have logged into Hugging Face if using a gated model:
    # huggingface-cli login
    main()
