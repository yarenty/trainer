"""
Q&A Generation Package

A modular package for generating question-answer pairs from documentation and code files
using LLM-based processing with robust fallback mechanisms.
"""

from text_cleaner import TextCleaner
from chunker import Chunker
from llm_qa import LLM_QA
from output_converter import OutputConverter
from file_processor import FileProcessor

__version__ = "1.0.0"
__author__ = "Q&A Generator Team"

__all__ = [
    "TextCleaner",
    "Chunker", 
    "LLM_QA",
    "OutputConverter",
    "FileProcessor"
] 