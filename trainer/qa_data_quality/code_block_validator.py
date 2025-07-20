"""
code_block_validator.py: Validate code blocks and formatting in Q/A datasets.

This module provides a class to flag answers with unclosed code blocks, mismatched quotes/brackets, or broken markdown.

Usage:
    from trainer.qa_data_quality.code_block_validator import QACodeBlockValidator
    validator = QACodeBlockValidator()
    report = validator.validate_all_files()
    print(report)

"""
import os
import json
import glob
import logging
import re
from typing import List, Dict, Any
from trainer.config import DATA_DIR

class QACodeBlockValidator:
    """
    Validates code blocks and formatting in all .jsonl files under qa_data/*/*.jsonl.
    Flags answers with unclosed code blocks, mismatched quotes/brackets, or broken markdown.
    """
    CODE_BLOCK_PATTERN = re.compile(r'```')
    QUOTE_PATTERN = re.compile(r'"||||')
    SINGLE_QUOTE_PATTERN = re.compile(r"'")
    BRACKETS = {'(': ')', '[': ']', '{': '}'}

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_all_files(self) -> Dict[str, Any]:
        """
        Scan all .jsonl files and flag answers with code block/formatting issues.
        Returns a report with flagged answers per file.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.validate_file(file)
            if file_report['flagged']:
                report[file] = file_report
        return report

    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """
        Validate code blocks/formatting in a single .jsonl file.
        Returns a report of flagged answers.
        """
        flagged = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    answer = obj.get('answer', '').strip()
                    issues = self._find_issues(answer)
                    if issues:
                        flagged.append({'line': i, 'issues': issues, 'answer': answer[:80] + ('...' if len(answer) > 80 else '')})
                except Exception:
                    continue
        return {
            'flagged': flagged,
            'num_flagged': len(flagged)
        }

    def _find_issues(self, text: str) -> List[str]:
        issues = []
        # Unclosed code blocks
        if self.CODE_BLOCK_PATTERN.findall(text) and len(self.CODE_BLOCK_PATTERN.findall(text)) % 2 != 0:
            issues.append('Unclosed code block (odd number of triple backticks)')
        # Mismatched double quotes
        if text.count('"') % 2 != 0:
            issues.append('Mismatched double quotes')
        # Mismatched single quotes
        if text.count("'") % 2 != 0:
            issues.append('Mismatched single quotes')
        # Mismatched brackets
        for open_b, close_b in self.BRACKETS.items():
            if text.count(open_b) != text.count(close_b):
                issues.append(f'Mismatched {open_b}{close_b} brackets')
        # Broken markdown (e.g., lone backticks)
        if text.count('`') % 2 != 0:
            issues.append('Mismatched inline backticks')
        return issues 