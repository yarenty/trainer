"""
Output conversion and file writing utilities.
"""

import json
import logging
from typing import Dict, List, Any
from pathlib import Path


class OutputConverter:
    """Handles output formatting and file writing operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def write_qa_pairs_to_jsonl(self, qa_pairs: List[Dict[str, str]], output_file: str, 
                               source_info: Dict[str, str] = None) -> None:
        """
        Write Q&A pairs to a JSONL file.
        
        Args:
            qa_pairs: List of Q&A pair dictionaries
            output_file: Path to output file
            source_info: Additional source information to include
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for qa_pair in qa_pairs:
                    # Add source information if provided
                    if source_info:
                        qa_pair.update(source_info)
                    
                    # Write as JSONL
                    json_line = json.dumps(qa_pair, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            self.logger.info(f"Wrote {len(qa_pairs)} Q&A pairs to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error writing to {output_file}: {e}")
            raise
    
    def format_qa_pair_for_output(self, qa_pair: Dict[str, str], 
                                 source_file: str = None, 
                                 repo_name: str = None) -> Dict[str, Any]:
        """
        Format a Q&A pair for output with metadata.
        
        Args:
            qa_pair: Q&A pair dictionary
            source_file: Source file path
            repo_name: Repository name
            
        Returns:
            Formatted Q&A pair with metadata
        """
        formatted_pair = qa_pair.copy()
        
        # Add metadata
        if source_file:
            formatted_pair['source_file'] = source_file
        
        if repo_name:
            formatted_pair['source_repo'] = repo_name
        
        # Add timestamp
        import datetime
        formatted_pair['generated_at'] = datetime.datetime.now().isoformat()
        
        return formatted_pair
    
    def validate_qa_pair(self, qa_pair: Dict[str, str]) -> bool:
        """
        Validate Q&A pair before writing.
        
        Args:
            qa_pair: Q&A pair dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(qa_pair, dict):
            return False
        
        required_keys = ['question', 'answer']
        if not all(key in qa_pair for key in required_keys):
            return False
        
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # Check for minimum content
        if len(question.strip()) < 10:
            return False
        
        if len(answer.strip()) < 20:
            return False
        
        # Check for error indicators
        if 'error:' in question.lower() or 'error:' in answer.lower():
            return False
        
        return True
    
    def filter_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter out invalid Q&A pairs.
        
        Args:
            qa_pairs: List of Q&A pair dictionaries
            
        Returns:
            Filtered list of valid Q&A pairs
        """
        valid_pairs = []
        invalid_count = 0
        
        for qa_pair in qa_pairs:
            if self.validate_qa_pair(qa_pair):
                valid_pairs.append(qa_pair)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            self.logger.warning(f"Filtered out {invalid_count} invalid Q&A pairs")
        
        return valid_pairs
    
    def merge_jsonl_files(self, input_files: List[str], output_file: str) -> None:
        """
        Merge multiple JSONL files into one.
        
        Args:
            input_files: List of input JSONL file paths
            output_file: Output file path
        """
        try:
            all_qa_pairs = []
            
            for input_file in input_files:
                if Path(input_file).exists():
                    with open(input_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    qa_pair = json.loads(line)
                                    all_qa_pairs.append(qa_pair)
                                except json.JSONDecodeError as e:
                                    self.logger.warning(f"Invalid JSON line in {input_file}: {e}")
                else:
                    self.logger.warning(f"Input file not found: {input_file}")
            
            # Write merged file
            self.write_qa_pairs_to_jsonl(all_qa_pairs, output_file)
            self.logger.info(f"Merged {len(all_qa_pairs)} Q&A pairs into {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error merging files: {e}")
            raise
    
    def create_output_summary(self, qa_pairs: List[Dict[str, str]], 
                            output_file: str) -> Dict[str, Any]:
        """
        Create a summary of the generated Q&A pairs.
        
        Args:
            qa_pairs: List of Q&A pair dictionaries
            output_file: Path to save summary
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_pairs': len(qa_pairs),
            'unique_questions': len(set(qa_pair.get('question', '') for qa_pair in qa_pairs)),
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'sources': set(),
            'repos': set()
        }
        
        if qa_pairs:
            question_lengths = [len(qa_pair.get('question', '')) for qa_pair in qa_pairs]
            answer_lengths = [len(qa_pair.get('answer', '')) for qa_pair in qa_pairs]
            
            summary['avg_question_length'] = sum(question_lengths) / len(question_lengths)
            summary['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
            
            for qa_pair in qa_pairs:
                if 'source_file' in qa_pair:
                    summary['sources'].add(qa_pair['source_file'])
                if 'source_repo' in qa_pair:
                    summary['repos'].add(qa_pair['source_repo'])
            
            summary['sources'] = list(summary['sources'])
            summary['repos'] = list(summary['repos'])
        
        # Write summary to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Summary written to {output_file}")
        except Exception as e:
            self.logger.error(f"Error writing summary: {e}")
        
        return summary 