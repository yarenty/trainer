#!/usr/bin/env python3
"""
Test script for the modular Q&A generation package.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from trainer.qa_prepare.text_cleaner import TextCleaner
        print("✓ TextCleaner imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TextCleaner: {e}")
        return False
    
    try:
        from trainer.qa_prepare.chunker import Chunker
        print("✓ Chunker imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Chunker: {e}")
        return False
    
    try:
        from trainer.qa_prepare.llm_qa import LLM_QA
        print("✓ LLM_QA imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LLM_QA: {e}")
        return False
    
    try:
        from trainer.qa_prepare.output_converter import OutputConverter
        print("✓ OutputConverter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OutputConverter: {e}")
        return False
    
    try:
        from trainer.qa_prepare.file_processor import FileProcessor
        print("✓ FileProcessor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FileProcessor: {e}")
        return False
    
    return True

def test_text_cleaner():
    """Test TextCleaner functionality."""
    print("\nTesting TextCleaner...")
    
    from trainer.qa_prepare.text_cleaner import TextCleaner
    
    cleaner = TextCleaner()
    
    # Test basic text cleaning
    test_text = "  This   is   a   test   text  with   extra   spaces  "
    cleaned = cleaner.clean_text(test_text)
    expected = "This is a test text with extra spaces"
    
    if cleaned == expected:
        print("✓ Basic text cleaning works")
    else:
        print(f"✗ Basic text cleaning failed: expected '{expected}', got '{cleaned}'")
        return False
    
    # Test markdown cleaning
    md_text = "# Header\n**Bold text** and `code`"
    cleaned_md = cleaner.clean_text(md_text)
    if "Header" in cleaned_md and "Bold text" in cleaned_md and "code" in cleaned_md:
        print("✓ Markdown cleaning works")
    else:
        print("✗ Markdown cleaning failed")
        return False
    
    return True

def test_chunker():
    """Test Chunker functionality."""
    print("\nTesting Chunker...")
    
    from trainer.qa_prepare.chunker import Chunker
    
    chunker = Chunker()
    
    # Test fixed-size chunking
    test_text = "This is a test text that should be chunked into smaller pieces for processing."
    chunks = chunker.chunk_by_fixed_size(test_text, chunk_size=20, overlap=5)
    
    if len(chunks) > 1:
        print("✓ Fixed-size chunking works")
    else:
        print("✗ Fixed-size chunking failed")
        return False
    
    # Test paragraph chunking
    para_text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    para_chunks = chunker.chunk_by_paragraphs(para_text, min_chars=10)
    
    if len(para_chunks) >= 2:
        print("✓ Paragraph chunking works")
    else:
        print("✗ Paragraph chunking failed")
        return False
    
    return True

def test_output_converter():
    """Test OutputConverter functionality."""
    print("\nTesting OutputConverter...")
    
    from trainer.qa_prepare.output_converter import OutputConverter
    
    converter = OutputConverter()
    
    # Test Q&A validation
    valid_qa = {
        "question": "What is this?",
        "answer": "This is a test answer with sufficient length to be valid."
    }
    
    invalid_qa = {
        "question": "Short",
        "answer": "Short"
    }
    
    if converter.validate_qa_pair(valid_qa):
        print("✓ Valid Q&A validation works")
    else:
        print("✗ Valid Q&A validation failed")
        return False
    
    if not converter.validate_qa_pair(invalid_qa):
        print("✓ Invalid Q&A validation works")
    else:
        print("✗ Invalid Q&A validation failed")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing modular Q&A generation package...\n")
    
    tests = [
        test_imports,
        test_text_cleaner,
        test_chunker,
        test_output_converter
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular structure is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 