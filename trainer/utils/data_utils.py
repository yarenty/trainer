"""
data_utils.py: Shared utilities for data loading and tokenization.

Provides functions to load and tokenize datasets for all training modes.
"""
import logging
import glob
import json
from datasets import Dataset

def load_and_tokenize_dataset(data_dir, tokenizer, max_length=2048):
    """
    Loads and tokenizes the dataset from data_dir using the provided tokenizer.

    Args:
        data_dir (str): Path to the training data directory.
        tokenizer: Hugging Face tokenizer.
        max_length (int): Maximum sequence length.

    Returns:
        tokenized_dataset: A tokenized Hugging Face Dataset.
    """
    logging.info(f"Loading and tokenizing dataset from {data_dir}...")
    data_files = glob.glob(f"{data_dir}/**/*.jsonl", recursive=True)
    if not data_files:
        logging.error(f"No .jsonl files found in {data_dir}")
        return None
    samples = []
    for file in data_files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    question = item.get('question', '').strip()
                    answer = item.get('answer', '').strip()
                    if question and answer:
                        samples.append({'question': question, 'answer': answer})
                except Exception as e:
                    logging.warning(f"Skipping line in {file} due to error: {e}")
    if not samples:
        logging.error("No valid samples found in dataset.")
        return None
    dataset = Dataset.from_list(samples)

    def preprocess(example):
        prompt = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
        return tokenizer(prompt, truncation=True, max_length=max_length, padding='max_length')

    logging.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess, batched=False)
    return tokenized_dataset 