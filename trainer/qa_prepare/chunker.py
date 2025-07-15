"""
Text chunking utilities for breaking large documents into manageable pieces.
"""

import re
import logging
from typing import List


class Chunker:
    """Handles different strategies for chunking text content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def chunk_by_headings(self, text: str, min_chars: int = 200) -> List[str]:
        """
        Chunk text by headings (lines starting with #).
        
        Args:
            text: Text content to chunk
            min_chars: Minimum characters per chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            # Check if this is a heading
            if line.strip().startswith('#'):
                # If we have a current chunk and it's long enough, save it
                if current_chunk and current_length >= min_chars:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Start new chunk with heading
                current_chunk = [line]
                current_length = len(line)
            else:
                # Add line to current chunk
                current_chunk.append(line)
                current_length += len(line)
        
        # Add the last chunk if it exists and is long enough
        if current_chunk and current_length >= min_chars:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str, min_chars: int = 100) -> List[str]:
        """
        Chunk text by paragraphs (double line breaks).
        
        Args:
            text: Text content to chunk
            min_chars: Minimum characters per chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split by double line breaks
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would make chunk too long, save current chunk
            if current_chunk and current_length + len(paragraph) > 1000:
                if current_length >= min_chars:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(paragraph)
            current_length += len(paragraph)
        
        # Add the last chunk if it exists and is long enough
        if current_chunk and current_length >= min_chars:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def chunk_by_fixed_size(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk text into fixed-size pieces with overlap.
        
        Args:
            text: Text content to chunk
            chunk_size: Target size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a word boundary
            if end < len(text):
                # Look for a good break point (space, newline, punctuation)
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in ' \n.,;:!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_code_by_functions(self, text: str, min_chars: int = 100) -> List[str]:
        """
        Chunk code by function definitions.
        
        Args:
            text: Code text to chunk
            min_chars: Minimum characters per chunk
            
        Returns:
            List of code chunks
        """
        if not text:
            return []
        
        # Common function patterns for different languages
        function_patterns = [
            r'def\s+\w+\s*\(',  # Python
            r'fn\s+\w+\s*\(',   # Rust
            r'function\s+\w+\s*\(',  # JavaScript
            r'public\s+\w+\s+\w+\s*\(',  # Java
            r'private\s+\w+\s+\w+\s*\(',  # Java
            r'protected\s+\w+\s+\w+\s*\(',  # Java
            r'class\s+\w+',  # Class definitions
            r'struct\s+\w+',  # Rust structs
            r'impl\s+\w+',    # Rust implementations
        ]
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            # Check if this line starts a function/class/struct
            is_function_start = any(re.search(pattern, line) for pattern in function_patterns)
            
            if is_function_start:
                # If we have a current chunk and it's long enough, save it
                if current_chunk and current_length >= min_chars:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Start new chunk
                current_chunk = [line]
                current_length = len(line)
            else:
                # Add line to current chunk
                current_chunk.append(line)
                current_length += len(line)
        
        # Add the last chunk if it exists and is long enough
        if current_chunk and current_length >= min_chars:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def chunk_code_by_blocks(self, text: str, min_chars: int = 100) -> List[str]:
        """
        Chunk code by logical blocks (functions, classes, etc.).
        
        Args:
            text: Code text to chunk
            min_chars: Minimum characters per chunk
            
        Returns:
            List of code chunks
        """
        if not text:
            return []
        
        # Try function-based chunking first
        chunks = self.chunk_code_by_functions(text, min_chars)
        
        # If that didn't produce enough chunks, fall back to fixed-size chunking
        if len(chunks) < 2:
            chunks = self.chunk_by_fixed_size(text, chunk_size=300, overlap=30)
        
        return chunks 