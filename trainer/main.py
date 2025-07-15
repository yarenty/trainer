import argparse
import importlib
import sys
import os

STEPS_DIR = os.path.join(os.path.dirname(__file__), 'steps')

STEP_SCRIPTS = {
    '1_download': '1_download',
    '2_train': '2_train',
    '3_merge': '3_merge',
    '4_gguf': '4_gguf',
    '5_quantize': '5_quantize',
    '6_ollama': '6_ollama',
    '7_evaluate': '7_evaluate',
}

def main():
    parser = argparse.ArgumentParser(description="Run a specific pipeline step.")
    parser.add_argument('--step', required=True, choices=STEP_SCRIPTS.keys(), help='Step to run (e.g., 1_download)')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Device to use for training (cpu or gpu)')
    args = parser.parse_args()

    # Setup logging
    from trainer.utils.logging_utils import setup_logging
    setup_logging(verbose=args.verbose)

    step_module_name = f"trainer.steps.{STEP_SCRIPTS[args.step]}"
    try:
        step_module = importlib.import_module(step_module_name)
        # Pass device argument to step if it accepts it
        if hasattr(step_module, 'main'):
            import inspect
            if 'device' in inspect.signature(step_module.main).parameters:
                step_module.main(device=args.device)
            else:
                step_module.main()
        else:
            print(f"No main() function found in {step_module_name}")
            sys.exit(1)
    except Exception as e:
        print(f"Error running step {args.step}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 