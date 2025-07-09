#!/usr/bin/env python3
"""
Q&A Generation Script

This script uses the modular classes from the src package to generate Q&A pairs
from documentation and code files in repositories.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from the module
from file_processor import FileProcessor
import ollama


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate Q&A pairs from repository documentation and code files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single repository
  python prepare_data_modular.py --repo-path /opt/ml/trainer/sources --repo-name datafusion --output-dir ./output

  # Process multiple repositories
  uv run prepare_data_modular.py --repo-path /opt/ml/trainer/sources --output-dir ./output --batch-mode --verbose

  # Use a different model
  python prepare_data_modular.py --repo-path /opt/ml/trainer/sources --model llama3.1 --output-dir ./output

  # Enable verbose logging
  python prepare_data_modular.py --repo-path /opt/ml/trainer/sources --verbose --output-dir ./output
        """
    )
    
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to repository or directory containing repositories"
    )
    
    parser.add_argument(
        "--repo-name",
        help="Name of the repository (if processing a single repo)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for generated Q&A pairs (default: ./output)"
    )
    
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model to use for Q&A generation (default: llama3.2)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent workers (default: 8)"
    )
    
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Process all repositories in the specified directory"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Ollama client
    try:
        client = ollama.Client()
        logger.info(f"Connected to Ollama client")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        sys.exit(1)
    
    # Initialize file processor
    processor = FileProcessor(
        ollama_client=client,
        model_name=args.model,
        max_workers=args.max_workers
    )
    
    # Process repositories
    if args.batch_mode:
        # Process all repositories in the directory
        if not repo_path.is_dir():
            logger.error("Batch mode requires a directory path")
            sys.exit(1)
        
        repos = [d for d in repo_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logger.info(f"Found {len(repos)} repositories to process")
        
        total_qa_pairs = 0
        for repo_dir in repos:
            repo_name = repo_dir.name
            logger.info(f"Processing repository: {repo_name}")
            
            try:
                summary = processor.process_repository(
                    repo_name=repo_name,
                    repo_path=str(repo_dir),
                    output_base_dir=str(output_dir)
                )
                total_qa_pairs += summary['total_qa_pairs']
                logger.info(f"Completed {repo_name}: {summary['total_qa_pairs']} Q&A pairs")
            except Exception as e:
                logger.error(f"Error processing {repo_name}: {e}")
                continue
        
        logger.info(f"Batch processing completed: {total_qa_pairs} total Q&A pairs generated")
        
    else:
        # Process single repository
        if not args.repo_name:
            logger.error("Repository name is required when not using batch mode")
            sys.exit(1)
        
        if repo_path.is_file():
            logger.error("Repository path must be a directory")
            sys.exit(1)
        
        try:
            summary = processor.process_repository(
                repo_name=args.repo_name,
                repo_path=str(repo_path),
                output_base_dir=str(output_dir)
            )
            logger.info(f"Processing completed: {summary['total_qa_pairs']} Q&A pairs generated")
        except Exception as e:
            logger.error(f"Error processing repository: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main() 