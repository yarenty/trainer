import argparse
import importlib
import sys
import os
from trainer.config import TRAIN_MODE, DEVICE

STEPS_DIR = os.path.join(os.path.dirname(__file__), 'steps')

STEP_SCRIPTS = {
    '1_download': '1_download',
    '2_train': '2_train',
    '3_merge': '3_merge',
    '4_gguf': '4_gguf',
    '5_quantize': '5_quantize',
    '6_ollama': '6_ollama',
    '7_evaluate': '7_evaluate',
    '8_upload_to_hf': '8_upload_to_hf',
}

def main():
    """
    Entrypoint for running a specific pipeline step.
    Supports device and training mode selection for training steps.
    """
    parser = argparse.ArgumentParser(description="Run a specific pipeline step.")
    parser.add_argument('--step', required=True, choices=STEP_SCRIPTS.keys(), help='Step to run (e.g., 1_download)')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default=DEVICE, help='Device to use for training (cpu or gpu)')
    parser.add_argument('--train-mode', choices=['full', 'lora', 'sft'], default=TRAIN_MODE, help='Training mode: full, lora, or sft')
    args = parser.parse_args()

    # Setup logging
    from trainer.utils.logging_utils import setup_logging
    setup_logging(verbose=args.verbose)

    step_module_name = f"trainer.steps.{STEP_SCRIPTS[args.step]}"
    try:
        step_module = importlib.import_module(step_module_name)
        # Pass device and train_mode arguments to step if it accepts them
        if hasattr(step_module, 'main'):
            import inspect
            sig = inspect.signature(step_module.main)
            params = sig.parameters
            kwargs = {}
            if 'device' in params:
                kwargs['device'] = args.device
            if 'train_mode' in params:
                kwargs['train_mode'] = args.train_mode
            step_module.main(**kwargs)
        else:
            print(f"No main() function found in {step_module_name}")
            sys.exit(1)
    except Exception as e:
        print(f"Error running step {args.step}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 