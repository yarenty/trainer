"""
File processing utilities for handling documentation and code files.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

from .text_cleaner import TextCleaner
from .chunker import Chunker
from .llm_qa import LLM_QA
from .output_converter import OutputConverter


class FileProcessor:
    """Handles file discovery, reading, and processing coordination."""
    
    def __init__(self, ollama_client, model_name: str = "llama3.2", max_workers: int = 8):
        """
        Initialize the file processor.
        
        Args:
            ollama_client: Ollama client instance
            model_name: Name of the model to use
            max_workers: Maximum number of concurrent workers
        """
        self.ollama_client = ollama_client
        self.model_name = model_name
        self.max_workers = max_workers
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.chunker = Chunker()
        self.llm_qa = LLM_QA(ollama_client, model_name)
        self.output_converter = OutputConverter()
        self.output_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def process_repository(self, repo_name: str, repo_path: str, output_base_dir: str) -> Dict[str, Any]:
        """
        Process an entire repository to generate Q&A pairs.
        
        Args:
            repo_name: Name of the repository
            repo_path: Path to the repository
            output_base_dir: Base directory for output files
            
        Returns:
            Processing summary
        """
        self.logger.info(f"Processing repository: {repo_name} at {repo_path}")
        
        # Create output directory
        output_dir = Path(output_base_dir) / repo_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process documentation files
        docs_output = output_dir / "docs_qa.jsonl"
        docs_summary = self._process_documentation_files(repo_path, str(docs_output), repo_name)
        
        # Process code files
        code_output = output_dir / "code_qa.jsonl"
        code_summary = self._process_code_files(repo_path, str(code_output), repo_name)
        
        # Merge results only if files exist and have content
        merged_output = output_dir / "merged_qa.jsonl"
        existing_files = []
        if docs_output.exists() and docs_output.stat().st_size > 0:
            existing_files.append(str(docs_output))
        if code_output.exists() and code_output.stat().st_size > 0:
            existing_files.append(str(code_output))
        
        if existing_files:
            self.output_converter.merge_jsonl_files(existing_files, str(merged_output))
        else:
            self.logger.warning("No Q&A pairs generated, skipping merge")
        
        # Create overall summary
        summary = {
            'repo_name': repo_name,
            'repo_path': repo_path,
            'docs_summary': docs_summary,
            'code_summary': code_summary,
            'total_files_processed': docs_summary['files_processed'] + code_summary['files_processed'],
            'total_qa_pairs': docs_summary['qa_pairs_generated'] + code_summary['qa_pairs_generated']
        }
        
        # Write summary
        summary_file = output_dir / "processing_summary.json"
        self.output_converter.create_output_summary([], str(summary_file))
        
        self.logger.info(f"Completed processing {repo_name}: {summary['total_qa_pairs']} Q&A pairs generated")
        return summary
    
    def _process_documentation_files(self, repo_path: str, output_file: str, repo_name: str) -> Dict[str, Any]:
        """
        Process documentation files in the repository.
        
        Args:
            repo_path: Path to repository
            output_file: Output file path
            repo_name: Repository name
            
        Returns:
            Processing summary
        """
        self.logger.info(f"Processing documentation files in {repo_path}")
        
        # Find documentation files
        doc_files = self._find_documentation_files(repo_path)
        self.logger.info(f"Found {len(doc_files)} documentation files")
        
        # Process files concurrently
        files_processed = 0
        qa_pairs_generated = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(self._process_single_doc_file, file_path, repo_name, output_file): file_path
                for file_path in doc_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        files_processed += 1
                        qa_pairs_generated += result
                        self.logger.debug(f"Processed {file_path}: {result} Q&A pairs")
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
        
        self.logger.info(f"Generated {qa_pairs_generated} Q&A pairs for documentation files")
        if qa_pairs_generated == 0:
            self.logger.warning(f"No Q&A pairs generated for documentation files")
        
        return {
            'files_processed': files_processed,
            'qa_pairs_generated': qa_pairs_generated,
            'files_found': len(doc_files)
        }
    
    def _process_code_files(self, repo_path: str, output_file: str, repo_name: str) -> Dict[str, Any]:
        """
        Process code files in the repository.
        
        Args:
            repo_path: Path to repository
            output_file: Output file path
            repo_name: Repository name
            
        Returns:
            Processing summary
        """
        self.logger.info(f"Processing code files in {repo_path}")
        
        # Find code files
        code_files = self._find_code_files(repo_path)
        self.logger.info(f"Found {len(code_files)} code files")
        
        # Process files concurrently
        files_processed = 0
        qa_pairs_generated = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(self._process_single_code_file, file_path, repo_name, output_file): file_path
                for file_path in code_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        files_processed += 1
                        qa_pairs_generated += result
                        self.logger.debug(f"Processed {file_path}: {result} Q&A pairs")
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
        
        self.logger.info(f"Generated {qa_pairs_generated} Q&A pairs for code files")
        if qa_pairs_generated == 0:
            self.logger.warning(f"No Q&A pairs generated for code files")
        
        return {
            'files_processed': files_processed,
            'qa_pairs_generated': qa_pairs_generated,
            'files_found': len(code_files)
        }
    
    def _find_documentation_files(self, repo_path: str) -> List[str]:
        """
        Find documentation files in the repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of documentation file paths
        """
        doc_extensions = {'.md', '.rst', '.txt', '.adoc', '.asciidoc'}
        doc_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common directories that don't contain documentation
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', 'target', 'build'}]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in doc_extensions:
                    doc_files.append(str(file_path))
        
        return doc_files
    
    def _find_code_files(self, repo_path: str) -> List[str]:
        """
        Find code files in the repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            List of code file paths
        """
        code_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.rs', '.go', 
            '.php', '.rb', '.swift', '.kt', '.scala', '.cs', '.fs', '.clj', '.hs'
        }
        code_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common directories that don't contain source code
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', 'target', 'build', 'dist'}]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in code_extensions:
                    code_files.append(str(file_path))
        
        return code_files
    
    def _process_single_doc_file(self, file_path: str, repo_name: str, output_file: str) -> int:
        """
        Process a single documentation file.
        
        Args:
            file_path: Path to documentation file
            repo_name: Repository name
            
        Returns:
            Number of Q&A pairs generated
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return 0
            
            # Clean text
            cleaned_content = self.text_cleaner.clean_text(content)
            cleaned_content = self.text_cleaner.remove_boilerplate(cleaned_content)
            
            if not cleaned_content.strip():
                return 0
            
            # Chunk text
            chunks = []
            chunking_strategies = [
                lambda t: self.chunker.chunk_by_headings(t, min_chars=200),
                lambda t: self.chunker.chunk_by_paragraphs(t, min_chars=100),
                lambda t: self.chunker.chunk_by_fixed_size(t, chunk_size=500, overlap=50)
            ]
            
            for strategy in chunking_strategies:
                chunks = strategy(cleaned_content)
                if len(chunks) >= 2:  # Prefer strategies that produce multiple chunks
                    break
            
            if not chunks:
                return 0
            
            # Generate Q&A pairs for each chunk
            count = 0
            self.logger.debug(f"Processing {len(chunks)} chunks from {file_path}")
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    self.logger.debug(f"Skipping chunk {i+1} (too short: {len(chunk.strip())} chars)")
                    continue
                
                try:
                    self.logger.debug(f"Generating Q&A for chunk {i+1}/{len(chunks)}")
                    qa_pair = self.llm_qa.generate_qa_pair(chunk)
                    if qa_pair and self.output_converter.validate_qa_pair(qa_pair):
                        # Add metadata
                        formatted_pair = self.output_converter.format_qa_pair_for_output(
                            qa_pair, file_path, repo_name
                        )
                        with self.output_lock:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                import json, os
                                f.write(json.dumps(formatted_pair, ensure_ascii=False) + '\n')
                                f.flush()
                                os.fsync(f.fileno())
                        count += 1
                        self.logger.debug(f"Successfully wrote Q&A pair for chunk {i+1}")
                    else:
                        self.logger.debug(f"Generated Q&A pair for chunk {i+1} failed validation")
                except Exception as e:
                    self.logger.warning(f"Error generating Q&A for chunk {i+1} in {file_path}: {e}")
                    continue
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error processing documentation file {file_path}: {e}")
            return 0
    
    def _process_single_code_file(self, file_path: str, repo_name: str, output_file: str) -> int:
        """
        Process a single code file.
        
        Args:
            file_path: Path to code file
            repo_name: Repository name
            
        Returns:
            Number of Q&A pairs generated
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return 0
            
            # Clean code text
            cleaned_content = self.text_cleaner.clean_code_text(content)
            
            if not cleaned_content.strip():
                return 0
            
            # Chunk code
            chunks = self.chunker.chunk_code_by_blocks(cleaned_content, min_chars=100)
            
            if not chunks:
                return 0
            
            # Generate Q&A pairs for each chunk
            count = 0
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                try:
                    qa_pair = self.llm_qa.generate_qa_pair(chunk)
                    if qa_pair and self.output_converter.validate_qa_pair(qa_pair):
                        # Add metadata
                        formatted_pair = self.output_converter.format_qa_pair_for_output(
                            qa_pair, file_path, repo_name
                        )
                        with self.output_lock:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                import json, os
                                f.write(json.dumps(formatted_pair, ensure_ascii=False) + '\n')
                                f.flush()
                                os.fsync(f.fileno())
                        count += 1
                        self.logger.debug(f"Successfully wrote Q&A pair for chunk {i+1}")
                    else:
                        self.logger.debug(f"Generated Q&A pair for chunk {i+1} failed validation")
                except Exception as e:
                    self.logger.warning(f"Error generating Q&A for chunk in {file_path}: {e}")
                    continue
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error processing code file {file_path}: {e}")
            return 0 