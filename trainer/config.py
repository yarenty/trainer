# config.py
"""
Central configuration for the trainer package.
Store all shared variables/constants here for reuse by scripts and modules.
"""

import os

# Example configuration variables (customize as needed)

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "output")
SOURCES_DIR = os.path.join(BASE_DIR, "..", "sources")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "..", "qwen2")


# Model and processing settings
QA_MODEL = "llama3.2"
DEFAULT_MODEL_NAME_CPU = "Qwen/Qwen2-7B-Instruct"
DEFAULT_MODEL_NAME = "unsloth/Qwen2-7B-Instruct-bnb-4bit"
FINE_TUNED_MODEL = "qwen2-7b-datafusion-instruct" 
MAX_WORKERS = 8


# Add more configuration variables as needed 