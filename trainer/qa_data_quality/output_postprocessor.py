"""
output_postprocessor.py: Post-process and review model outputs for Q/A datasets.

This module provides a class to trim outputs at the end marker, flag outputs that are too verbose/short or missing the marker, and is extensible for future LLM-based review.

Usage:
    from trainer.qa_data_quality.output_postprocessor import QAOutputPostProcessor
    postproc = QAOutputPostProcessor()
    report = postproc.process_all_files()
    print(report)

"""
import os
import json
import glob
import logging
from typing import List, Dict, Any
from trainer.config import DATA_DIR, END_MARKER

class QAOutputPostProcessor:
    """
    Post-processes model outputs: trims at end marker, flags verbosity/length issues, and is extensible for LLM-based review.
    """
    def __init__(self, data_dir: str = DATA_DIR, end_marker: str = END_MARKER, min_length: int = 20, max_length: int = 2048):
        self.data_dir = data_dir
        self.end_marker = end_marker
        self.min_length = min_length
        self.max_length = max_length
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all .jsonl files: trim at end marker, flag verbosity/length issues.
        Returns a report with flagged outputs per file.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.process_file(file)
            if file_report['flagged']:
                report[file] = file_report
        return report

    def process_file(self, filepath: str) -> Dict[str, Any]:
        """
        Process outputs in a single .jsonl file: trim at end marker, flag verbosity/length issues.
        Returns a report of flagged outputs.
        """
        flagged = []
        trimmed_lines = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    answer = obj.get('answer', '').strip()
                    trimmed_answer, issues = self._trim_and_flag(answer)
                    if issues:
                        flagged.append({'line': i, 'issues': issues, 'answer': answer[:80] + ('...' if len(answer) > 80 else '')})
                    obj['answer'] = trimmed_answer
                    trimmed_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                except Exception:
                    continue
        # Overwrite file with trimmed answers
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(trimmed_lines)
        return {
            'flagged': flagged,
            'num_flagged': len(flagged)
        }

    def _trim_and_flag(self, answer: str) -> (str, List[str]):
        issues = []
        trimmed = answer
        # Trim at end marker
        if self.end_marker in answer:
            trimmed = answer.split(self.end_marker)[0].rstrip() + '\n' + self.end_marker
        else:
            issues.append('Missing end marker')
        # Flag too short/long
        if len(trimmed) < self.min_length:
            issues.append('Answer too short')
        if len(trimmed) > self.max_length:
            issues.append('Answer too long')
        return trimmed, issues

    # Placeholder for future LLM-based review
    def llm_review(self, answer: str) -> Dict[str, Any]:
        """
        (Future) Use an LLM to review answer for verbosity, relevance, completeness.
        """
        raise NotImplementedError("LLM-based review not yet implemented.") 