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
        """
        Auto-correct missing or mismatched brackets in text.

        Strategy:
        1. Remove any closing bracket that does not have a corresponding opening bracket.
        2. Apply block-style closure for unclosed opening brackets at the end of a line (using fix_block_closing_brackets).
        3. Scan again and add inline closing brackets for any remaining unclosed opening brackets (at the end of the string).
        4. Fix quotes and backticks as before.

        Args:
            text: Text to auto-correct
        Returns:
            Tuple of (corrected_text, list of fixes made)
        """
        if not text:
            return text, []
        fixes = []
        # 1. Remove all unmatched closing brackets first
        open_to_close = {'(': ')', '{': '}', '[': ']'}
        close_to_open = {v: k for k, v in open_to_close.items()}
        stack = []
        new_text_parts = []
        for char in text:
            if char in open_to_close:
                stack.append(char)
                new_text_parts.append(char)
            elif char in close_to_open:
                if stack and stack[-1] == close_to_open[char]:
                    stack.pop()
                    new_text_parts.append(char)
                else:
                    fixes.append(f"Removed unmatched closing bracket '{char}'")
            else:
                new_text_parts.append(char)
        text = ''.join(new_text_parts)
        
        # 2. Block-style closure for unclosed opening brackets at end of line
        text = fix_block_closing_brackets(text)
        
        # 3. Scan again and add inline closing brackets for any remaining unclosed opening brackets
        temp_stack = []
        for char in text:
            if char in open_to_close:
                temp_stack.append(char)
            elif char in close_to_open:
                if temp_stack and temp_stack[-1] == close_to_open[char]:
                    temp_stack.pop()
        if temp_stack:
            for open_b in reversed(temp_stack):
                text += open_to_close[open_b]
                fixes.append(f"Added inline closing {open_to_close[open_b]}")
        
        # 4. Continue with quote/backtick logic as before
        for q in ['"', "'", '`']:
            if text.count(q) % 2 != 0:
                last_q_idx = text.rfind(q)
                insertion_point = text.find(' ', last_q_idx + 1)
                if insertion_point != -1:
                    text = text[:insertion_point] + q + text[insertion_point:]
                else:
                    text += q
                fixes.append(f'Added closing {q}')
        return text, fixes

def fix_block_closing_brackets(text: str) -> str:
    """
    Fixes unclosed opening brackets at the end of a line by inserting the corresponding closing brackets
    in LIFO (stack) order before the first empty line or before the first line with the same indentation level.

    Only brackets that are actually trailing at the end of the line are closed in this way. Outer blocks (like '{')
    are not closed unless their opening is also at the end of the line.

    Example:
        Input:
            fn foo() {
                let x = bar(
                    1,
                    2

                let y = 3;
            }
        Output:
            fn foo() {
                let x = bar(
                    1,
                    2
                )
                let y = 3;
            }
    """
    lines = text.splitlines()
    open_to_close = {'(': ')', '{': '}', '[': ']'}
    closing_insertions = []  # (line_idx, [closing_brackets], replace_empty, indent)

    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        # Find all consecutive trailing opening brackets at the end of the line
        trailing_opens = []
        i = len(stripped) - 1
        while i >= 0 and stripped[i] in open_to_close:
            trailing_opens.append(stripped[i])
            i -= 1
        # Only close if the trailing opens are at the very end (not after any closing bracket)
        if trailing_opens and (i == -1 or not stripped[i+1:].strip(''.join(open_to_close.values()))):
            indent = len(line) - len(line.lstrip())
            for j in range(idx + 1, len(lines)):
                next_line = lines[j]
                if not next_line.strip():  # treat any whitespace-only line as empty
                    closing_insertions.append((j, [open_to_close[b] for b in trailing_opens], True, indent))
                    break
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent == indent:
                    closing_insertions.append((j, [open_to_close[b] for b in trailing_opens], False, indent))
                    break
            else:
                # Always add a new line at the end with the correct indentation
                closing_insertions.append((len(lines), [open_to_close[b] for b in trailing_opens], False, indent))

    # Insert closing brackets in reverse order to not mess up line indices
    for insert_idx, closings, replace_empty, indent in reversed(closing_insertions):
        for closing in reversed(closings):
            closing_line = ' ' * indent + closing
            if replace_empty and insert_idx < len(lines) and not lines[insert_idx].strip():
                lines[insert_idx] = closing_line
                replace_empty = False  # Only replace the first empty line
            else:
                lines.insert(insert_idx, closing_line)
    return '\n'.join(lines) 