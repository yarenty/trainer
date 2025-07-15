import logging
from trainer.config import LOCAL_MODEL_DIR, MODELS_DIR, FINE_TUNED_MODEL, DATA_DIR
from trainer.utils.train_cpu import train_cpu
from trainer.utils.train_gpu import train_gpu

def main(device='cpu'):
    """Train/fine-tune the model using the specified device (cpu or gpu)."""
    logging.info(f"[2_train] Starting training on device: {device}")
    model_dir = LOCAL_MODEL_DIR
    data_dir = DATA_DIR
    output_dir = f"{MODELS_DIR}/{FINE_TUNED_MODEL}"
    config = None  # Pass config if needed
    if device == 'gpu':
        train_gpu(model_dir, data_dir, output_dir, config)
    else:
        train_cpu(model_dir, data_dir, output_dir, config)
    logging.info("[2_train] Training complete.")

if __name__ == "__main__":
    main() 