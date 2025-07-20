"""
code_block_validator.py: Validate and auto-correct code blocks and formatting in Q/A datasets.

This module provides a class to flag and auto-correct answers with unclosed code blocks, mismatched quotes/brackets, or broken markdown.

Usage:
    from trainer.qa_data_quality.code_block_validator import QACodeBlockValidator
    validator = QACodeBlockValidator()
    report = validator.validate_all_files()
    print(report)
    # For auto-correction:
    corrections = validator.auto_correct_all_files()
    print(corrections)

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
    Validates and auto-corrects code blocks and formatting in all .jsonl files under qa_data/*/*.jsonl.
    Flags and fixes answers with unclosed code blocks, mismatched quotes/brackets, or broken markdown.
    """
    CODE_BLOCK_PATTERN = re.compile(r'```')
    BRACKETS = {'(': ')', '[': ']', '{': '}'}
    QUOTES = {'"': '"', "'": "'"}

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_all_files(self) -> Dict[str, Any]:
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.validate_file(file)
            if file_report['flagged']:
                report[file] = file_report
        return report

    def validate_file(self, filepath: str) -> Dict[str, Any]:
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

    def auto_correct_all_files(self) -> Dict[str, Any]:
        """
        Auto-correct missing closing quotes and brackets in all .jsonl files.
        Returns a report of corrections made per file.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.auto_correct_file(file)
            if file_report['corrected']:
                report[file] = file_report
        return report

    def auto_correct_file(self, filepath: str) -> Dict[str, Any]:
        corrected = []
        new_lines = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    answer = obj.get('answer', '').strip()
                    fixed_answer, fixes = self._auto_correct(answer)
                    if fixes:
                        corrected.append({'line': i, 'fixes': fixes, 'original': answer[:80] + ('...' if len(answer) > 80 else ''), 'fixed': fixed_answer[:80] + ('...' if len(fixed_answer) > 80 else '')})
                    obj['answer'] = fixed_answer
                    new_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                except Exception:
                    continue
        # Overwrite file with corrected answers
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return {
            'corrected': corrected,
            'num_corrected': len(corrected)
        }

    def _auto_correct(self, text: str) -> (str, List[str]):
        fixes = []
        fixed = text
        # Fix unclosed code blocks
        n_code = len(self.CODE_BLOCK_PATTERN.findall(fixed))
        if n_code % 2 != 0:
            fixed += '\n```'
            fixes.append('Added closing code block (``` )')
        # Fix mismatched double quotes
        if fixed.count('"') % 2 != 0:
            fixed += '"'
            fixes.append('Added closing double quote')
        # Fix mismatched single quotes
        if fixed.count("'") % 2 != 0:
            fixed += "'"
            fixes.append('Added closing single quote')
        # Fix mismatched brackets
        for open_b, close_b in self.BRACKETS.items():
            n_open = fixed.count(open_b)
            n_close = fixed.count(close_b)
            if n_open > n_close:
                fixed += close_b * (n_open - n_close)
                fixes.append(f'Added {n_open - n_close} closing {close_b}')
            elif n_close > n_open:
                # Rare, but for completeness
                fixed = (open_b * (n_close - n_open)) + fixed
                fixes.append(f'Added {n_close - n_open} opening {open_b}')
        # Fix mismatched inline backticks
        if fixed.count('`') % 2 != 0:
            fixed += '`'
            fixes.append('Added closing inline backtick')
        return fixed, fixes 