"""
Trainer: Main package for Q&A generation, training, and post-processing.
"""

__version__ = "1.0.0"

from . import qa_prepare
from .config import *

# Expose key classes at the top level for convenience
from .qa_prepare import FileProcessor, TextCleaner, Chunker, LLM_QA, OutputConverter

__all__ = [
    "qa_prepare",
    "config",
    "FileProcessor",
    "TextCleaner",
    "Chunker",
    "LLM_QA",
    "OutputConverter"
] 