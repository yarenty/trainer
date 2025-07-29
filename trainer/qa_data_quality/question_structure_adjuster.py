"""
question_structure_adjuster.py: Adjusts question structure to follow a standardized format.

This module modifies questions to follow the format:
"Checking <project> at <file>: <original_question>"

Where:
- <project> is extracted from the "source_repo" field
- <file> is extracted from "source_file" field with "/home/jaro/trainer/sources/" prefix removed
- <original_question> is the current question body
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class QAQuestionStructureAdjuster:
    """Adjusts question structure to follow standardized format."""
    
    def __init__(self):
        self.source_prefix = "/home/jaro/trainer/sources/"
    
    def adjust_question_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Adjusts question structure in a single file.
        
        Args:
            file_path: Path to the JSONL file to process
            
        Returns:
            Dictionary containing adjustment report
        """
        report = {
            'file': file_path,
            'total_lines': 0,
            'adjusted_lines': 0,
            'issues': []
        }
        
        try:
            # Read the file and process line by line
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            report['total_lines'] = len(lines)
            adjusted_lines = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    adjusted_lines.append(line)
                    continue
                
                try:
                    # Parse JSON line
                    data = json.loads(line)
                    
                    # Check if this is a Q/A pair with required fields
                    if 'question' in data and 'source_repo' in data and 'source_file' in data:
                        original_question = data['question']
                        
                        # Extract project and file path
                        project = data['source_repo']
                        source_file = data['source_file']
                        
                        # Remove the source prefix from file path
                        if source_file.startswith(self.source_prefix):
                            file_path_clean = source_file[len(self.source_prefix):]
                        else:
                            file_path_clean = source_file
                        
                        # Create new question structure
                        new_question = f"Checking {project} at {file_path_clean}: {original_question}"
                        
                        # Update the data with new question
                        data['question'] = new_question
                        report['adjusted_lines'] += 1
                        
                        logger.debug(f"Line {line_num}: Adjusted question structure")
                    
                    # Write the adjusted line
                    adjusted_lines.append(json.dumps(data, ensure_ascii=False))
                    
                except json.JSONDecodeError as e:
                    report['issues'].append(f"Line {line_num}: Invalid JSON - {e}")
                    adjusted_lines.append(line)  # Keep original line
                except Exception as e:
                    report['issues'].append(f"Line {line_num}: Error processing - {e}")
                    adjusted_lines.append(line)  # Keep original line
            
            # Write the adjusted content back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in adjusted_lines:
                    f.write(line + '\n')
            
            logger.info(f"Processed {file_path}: {report['adjusted_lines']}/{report['total_lines']} lines adjusted")
            
        except Exception as e:
            report['issues'].append(f"File processing error: {e}")
            logger.error(f"Error processing file {file_path}: {e}")
        
        return report
    
    def adjust_all_files(self, data_dir: str = "qa_data") -> Dict[str, Dict[str, Any]]:
        """
        Adjusts question structure in all JSONL files in the data directory.
        
        Args:
            data_dir: Directory containing Q/A data files
            
        Returns:
            Dictionary mapping file paths to their adjustment reports
        """
        all_reports = {}
        
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                logger.error(f"Data directory {data_dir} does not exist")
                return all_reports
            
            # Find all JSONL files
            jsonl_files = list(data_path.rglob("*.jsonl"))
            
            if not jsonl_files:
                logger.warning(f"No JSONL files found in {data_dir}")
                return all_reports
            
            logger.info(f"Found {len(jsonl_files)} JSONL files to process")
            
            for file_path in jsonl_files:
                logger.info(f"Processing {file_path}")
                report = self.adjust_question_structure(str(file_path))
                all_reports[str(file_path)] = report
            
        except Exception as e:
            logger.error(f"Error in adjust_all_files: {e}")
        
        return all_reports 