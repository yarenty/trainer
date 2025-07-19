"""
Step 3: Merge Fine-Tuned Model

Copies the fine-tuned model directory to the merged model directory.
No adapter merging is needed for full fine-tuning.
"""
import logging
import os
import shutil
from trainer.config import MODELS_DIR, FINE_TUNED_MODEL, MERGED_MODEL

def main():
    """
    Copies the fine-tuned model directory to the merged model directory.
    Logs errors if the source directory does not exist or copying fails.
    """
    src_dir = os.path.join(MODELS_DIR, FINE_TUNED_MODEL)
    dst_dir = os.path.join(MODELS_DIR, MERGED_MODEL)
    logging.info(
        f"Copying fine-tuned model from {src_dir} to {dst_dir} (as merged model)..."
    )
    if not os.path.exists(src_dir):
        logging.error(
            f"Source directory {src_dir} does not exist. Run training first."
        )
        return
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    try:
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        logging.info(f"Merged model directory created at {dst_dir}")
    except Exception as e:
        logging.error(f"Error copying model directory: {e}")

if __name__ == "__main__":
    main() 