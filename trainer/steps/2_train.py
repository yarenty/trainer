"""
Step 2: Train/Fine-Tune Model

Trains or fine-tunes the model using the specified device (CPU or GPU)
and training mode (full, lora, sft). Uses config.py for all settings and
logging for progress.
"""
import logging
from trainer.config import LOCAL_MODEL_DIR, MODELS_DIR, FINE_TUNED_MODEL, DATA_DIR
from trainer.utils.train_cpu import train_cpu
from trainer.utils.train_gpu import train_gpu
from trainer.utils.train_lora import train_lora
from trainer.utils.train_sft import train_sft

def main(device='cpu', train_mode='full'):
    """
    Trains or fine-tunes the model using the specified device and training
    mode.

    Args:
        device (str): 'cpu' or 'gpu'.
        train_mode (str): 'full', 'lora', or 'sft'.
    """
    logging.info(f"[2_train] Starting training on device: {device}, mode: {train_mode}")
    model_dir = LOCAL_MODEL_DIR
    data_dir = DATA_DIR
    output_dir = f"{MODELS_DIR}/{FINE_TUNED_MODEL}"
    config = None  # Pass config if needed
    if train_mode == 'lora':
        train_lora(model_dir, data_dir, output_dir, config, device=device)
    elif train_mode == 'sft':
        train_sft(model_dir, data_dir, output_dir, config, device=device)
    elif train_mode == 'full' and device == 'gpu':
        train_gpu(model_dir, data_dir, output_dir, config)
    else:
        train_cpu(model_dir, data_dir, output_dir, config)
    logging.info("[2_train] Training complete.")

if __name__ == "__main__":
    main() 