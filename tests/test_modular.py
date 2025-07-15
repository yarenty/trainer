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
        assert False, f"Failed to import TextCleaner: {e}"
    try:
        from trainer.qa_prepare.chunker import Chunker
        print("✓ Chunker imported successfully")
    except ImportError as e:
        assert False, f"Failed to import Chunker: {e}"
    try:
        from trainer.qa_prepare.llm_qa import LLM_QA
        print("✓ LLM_QA imported successfully")
    except ImportError as e:
        assert False, f"Failed to import LLM_QA: {e}"
    try:
        from trainer.qa_prepare.output_converter import OutputConverter
        print("✓ OutputConverter imported successfully")
    except ImportError as e:
        assert False, f"Failed to import OutputConverter: {e}"
    try:
        from trainer.qa_prepare.file_processor import FileProcessor
        print("✓ FileProcessor imported successfully")
    except ImportError as e:
        assert False, f"Failed to import FileProcessor: {e}"

def test_text_cleaner():
    """Test TextCleaner functionality."""
    print("\nTesting TextCleaner...")
    from trainer.qa_prepare.text_cleaner import TextCleaner
    cleaner = TextCleaner()
    
    # Test basic text cleaning
    test_text = "  This   is   a   test   text  with   extra   spaces  "
    cleaned = cleaner.clean_text(test_text)
    expected = "This is a test text with extra spaces"
    assert cleaned == expected, f"Basic text cleaning failed: expected '{expected}', got '{cleaned}'"
    print("✓ Basic text cleaning works")
    md_text = "# Header\n**Bold text** and `code`"
    cleaned_md = cleaner.clean_text(md_text)
    assert "Header" in cleaned_md and "Bold text" in cleaned_md and "code" in cleaned_md, "Markdown cleaning failed"
    print("✓ Markdown cleaning works")

def test_chunker():
    """Test Chunker functionality."""
    print("\nTesting Chunker...")
    from trainer.qa_prepare.chunker import Chunker
    chunker = Chunker()
    test_text = "This is a test text that should be chunked into smaller pieces for processing."
    chunks = chunker.chunk_by_fixed_size(test_text, chunk_size=20, overlap=5)
    assert len(chunks) > 1, "Fixed-size chunking failed"
    print("✓ Fixed-size chunking works")
    para_text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    para_chunks = chunker.chunk_by_paragraphs(para_text, min_chars=10)
    assert len(para_chunks) >= 1, "Paragraph chunking failed"
    print("✓ Paragraph chunking works")

def test_output_converter():
    """Test OutputConverter functionality."""
    print("\nTesting OutputConverter...")
    from trainer.qa_prepare.output_converter import OutputConverter
    converter = OutputConverter()
    valid_qa = {"question": "What is this?", "answer": "This is a test answer with sufficient length to be valid."}
    invalid_qa = {"question": "Short", "answer": "Short"}
    assert converter.validate_qa_pair(valid_qa), "Valid Q&A validation failed"
    print("✓ Valid Q&A validation works")
    assert not converter.validate_qa_pair(invalid_qa), "Invalid Q&A validation failed"
    print("✓ Invalid Q&A validation works")

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
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test {test.__name__} failed with assertion error: {e}")
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