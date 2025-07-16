
# Install

## checks - todo
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
pip install datasets transformers tokenizers wandb ollama

#Alternative: Axolotl (more configurable)
git clone https://github.com/OpenAccess-AI-Collective/axolotl 
cd axolotl && pip install -e .
```


# Instruct Style Dataset Creation

## Create training data in categories:
A. SQL to DataFusion Translation
```json
{
  "instruction": "Convert this SQL query to DataFusion Rust code",
  "input": "SELECT name, age FROM users WHERE age > 25 ORDER BY name",
  "output": "use datafusion::prelude::*;\n\nlet df = ctx.sql(\"SELECT name, age FROM users WHERE age > 25 ORDER BY name\").await?;\nlet results = df.collect().await?;"
}
```
B. Custom Table Provider Implementation
```json
{
  "instruction": "Create a custom table provider for reading CSV files with schema inference",
  "input": "I need to create a table provider that can read CSV files and automatically infer the schema",
  "output": "use datafusion::datasource::TableProvider;\nuse datafusion::arrow::datatypes::SchemaRef;\n\n#[derive(Debug)]\npub struct CsvTableProvider {\n    schema: SchemaRef,\n    file_path: String,\n}\n\nimpl CsvTableProvider {\n    pub fn new(file_path: String) -> Result<Self> {\n        let schema = Self::infer_schema(&file_path)?;\n        Ok(Self { schema, file_path })\n    }\n}"
}
```
C. Performance Optimization
```json
{
  "instruction": "Optimize this DataFusion query for better performance",
  "input": "How can I optimize a query that joins two large tables?",
  "output": "For joining large tables in DataFusion:\n1. Use appropriate join algorithms: ctx.config().with_target_partitions(num_cpus::get())\n2. Partition your data appropriately\n3. Use columnar formats like Parquet\n4. Consider using broadcast joins for small tables\n5. Enable predicate pushdown with proper filters"
}
```

## Dataset Structure
Create a JSONL file with this format:

```jsonl
{"instruction": "...", "input": "...", "output": "..."}
{"instruction": "...", "input": "...", "output": "..."}
```

## Data Generation Script
```python
import json
import requests
from pathlib import Path

def create_datafusion_dataset():
    """Generate DataFusion training examples"""
    
    # Categories of examples to generate
    categories = [
        "sql_translation",
        "table_providers", 
        "streaming_integration",
        "error_handling",
        "performance_optimization",
        "kafka_integration",
        "custom_functions"
    ]
    
    examples = []
    
    # Add your collected examples here
    # You can scrape from:
    # - DataFusion documentation
    # - GitHub issues and examples
    # - Stack Overflow DataFusion questions
    # - Your own usage patterns
    
    return examples

# Generate dataset
dataset = create_datafusion_dataset()
with open("datafusion_training.jsonl", "w") as f:
    for example in dataset:
        f.write(json.dumps(example) + "\n")
```

# Model Selection and Setup

Recommended Models for Small GPUs
Option A: Code Llama 7B (Recommended)
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/codellama-7b-instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```
Option B: Mistral 7B
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```
Option C: Phi-3 Mini (Most efficient)
```pytho
nmodel, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

## LoRA Configuration
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```


# Fine-tuning Process

## Data Preparation
```python
from datasets import Dataset
import json

def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

# Load and format dataset
dataset = load_dataset("datafusion_training.jsonl")

def format_prompt(example):
    """Format examples for instruction following"""
    if example["input"]:
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    return {"text": prompt}

dataset = dataset.map(format_prompt)
```

## Training Configuration

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="./datafusion-model",
        report_to="wandb",  # Optional: for tracking
        save_steps=100,
        save_total_limit=2,
    ),
)
```

## Training

```python
# Start training
trainer.train()

# Save the model
trainer.model.save_pretrained("datafusion-lora")
trainer.tokenizer.save_pretrained("datafusion-lora")
Phase 5: Model Conversion and Deployment
5.1 Convert to GGUF Format for Ollama
python# Merge LoRA weights back into base model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Save merged model
model.save_pretrained_merged("datafusion-merged", tokenizer, save_method="merged_16bit")

# Convert to GGUF
model.save_pretrained_gguf("datafusion-gguf", tokenizer, quantization_method="q4_k_m")
```

## Create Ollama Modelfile

```dockerfile
# Save as 'Modelfile'
FROM ./datafusion-gguf/unsloth.Q4_K_M.gguf

TEMPLATE """### Instruction:
{{ .System }}

### Input:
{{ .Prompt }}

### Response:
"""

SYSTEM """You are a DataFusion expert assistant. You help users write efficient Rust code using the DataFusion library, optimize queries, create custom table providers, and integrate with various data sources including Kafka and databases."""

PARAMETER stop "###"
PARAMETER stop "### Instruction:"
PARAMETER stop "### Input:"
PARAMETER stop "### Response:"
```

## Import to Ollama
```bash
# Import the model
ollama create datafusion-expert -f Modelfile

# Test the model
ollama run datafusion-expert "How do I create a custom table provider in DataFusion?"
```

# Evaluation and Iteration

##  Create Evaluation Dataset

```python
# Create test cases for evaluation
test_cases = [
    {
        "prompt": "Convert this SQL to DataFusion: SELECT * FROM table1 JOIN table2 ON table1.id = table2.id",
        "expected_topics": ["sql", "join", "datafusion"]
    },
    {
        "prompt": "How do I integrate DataFusion with Kafka?",
        "expected_topics": ["kafka", "streaming", "integration"]
    }
]
```

## Evaluation Script

```python
import ollama

def evaluate_model(model_name, test_cases):
    results = []
    for case in test_cases:
        response = ollama.generate(
            model=model_name,
            prompt=case["prompt"],
            options={"temperature": 0.1}
        )
        results.append({
            "prompt": case["prompt"],
            "response": response["response"],
            "expected": case["expected_topics"]
        })
    return results

# Evaluate your model
results = evaluate_model("datafusion-expert", test_cases)
```

# Advanced Enhancements

## Tool Integration

Add function calling capabilities by including tool use examples in training data:
```json
{
  "instruction": "Use DataFusion to query a Parquet file and return the schema",
  "input": "I need to inspect the schema of a Parquet file using DataFusion",
  "output": "```rust\nuse datafusion::prelude::*;\n\nasync fn get_parquet_schema(file_path: &str) -> Result<()> {\n    let ctx = SessionContext::new();\n    ctx.register_parquet(\"my_table\", file_path, ParquetReadOptions::default()).await?;\n    \n    let df = ctx.sql(\"SELECT * FROM my_table LIMIT 0\").await?;\n    println!(\"Schema: {:?}\", df.schema());\n    Ok(())\n}\n```"
}
```

# Continuous Learning

- Set up a feedback loop to collect user queries
- Periodically retrain with new examples
- Version control your datasets and models



_Remember: Quality over quantity for training data. 500 excellent examples are better than 5000 mediocre ones._


# Fully connected pipeline

Tehre is possibility to create pipeline that will do all those steps automatically from start/data preparation to evaluation.

- prepare data
- post processing (rmoving "bad lines" )
- training - all the steps
- installing in ollama

And 2-3 days later see the results...
