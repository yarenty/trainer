# config.py
"""
Central configuration for the trainer package.
Store all shared variables/constants here for reuse by scripts and modules.
"""

import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent of trainer/
DATA_DIR = os.path.join(BASE_DIR, "qa_data")
SOURCES_DIR = os.path.join(BASE_DIR, "sources")
# model
MODELS_DIR = os.path.join(BASE_DIR,  "models")
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "local_model")
# llama.cpp installation
LLAMA_CPP_PATH = os.path.join(BASE_DIR, "llama.cpp")

# Model and processing settings
QA_MODEL = "llama3.2"
DEFAULT_MODEL_NAME =  "meta-llama/Llama-3.2-3B-Instruct" 
#Note: tried meta-llama/Llama-3.1-8B - but 80GB GPU is too small 

# step-by-step fine-tuning
FINE_TUNED_MODEL = "llama32-datafusion-instruct" 
MERGED_MODEL = "llama32-datafusion-instruct-merged" 
GGUF_MODEL = "llama32-datafusion-instruct-gguf"
FINAL_OLLAMA = "llama32_datafusion"

MAX_WORKERS = 8

GGUF_QUANT_TYPE = "q4_k_m"

# Default training mode and device
TRAIN_MODE = "full"
DEVICE = "gpu"
BATCH_SIZE = 4

