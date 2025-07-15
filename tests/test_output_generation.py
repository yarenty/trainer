#!/usr/bin/env python3
"""
Test script to verify output generation is working.
"""

import sys
import logging
from pathlib import Path

from trainer.qa_prepare.text_cleaner import TextCleaner
from trainer.qa_prepare.chunker import Chunker
from trainer.qa_prepare.output_converter import OutputConverter

def test_basic_output():
    """Test basic output generation without LLM."""
    print("Testing basic output generation...")
    
    # Create test data
    test_qa_pairs = [
        {"question": "What is Python?", "answer": "Python is a high-level programming language known for its simplicity and readability."},
        {"question": "How do you define a function in Python?", "answer": "You define a function using the 'def' keyword followed by the function name and parameters."}
    ]
    converter = OutputConverter()
    
    # Test writing to file
    output_file = "test_output.jsonl"
    try:
        converter.write_qa_pairs_to_jsonl(test_qa_pairs, output_file)
        print(f"✓ Successfully wrote {len(test_qa_pairs)} Q&A pairs to {output_file}")
        assert Path(output_file).exists(), "Output file was not created"
        with open(output_file, 'r') as f:
            lines = f.readlines()
            print(f"✓ File contains {len(lines)} lines")
            assert len(lines) == len(test_qa_pairs), f"Expected {len(test_qa_pairs)} lines, got {len(lines)}"
            print("✓ All Q&A pairs were written correctly")
    except Exception as e:
        assert False, f"Error writing output: {e}"

def test_file_discovery():
    """Test file discovery functionality."""
    print("\nTesting file discovery...")
    
    # Test with current directory
    current_dir = Path(".")
    
    # Look for any Python files
    python_files = list(current_dir.glob("*.py"))
    print(f"Found {len(python_files)} Python files in current directory")
    
    # Look for any markdown files
    md_files = list(current_dir.glob("*.md"))
    print(f"Found {len(md_files)} Markdown files in current directory")
    
    # Look for any text files
    txt_files = list(current_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} Text files in current directory")
    total_files = len(python_files) + len(md_files) + len(txt_files)
    print(f"Total files found: {total_files}")
    assert total_files > 0, "No files found - this might be the issue"
    print("✓ File discovery is working")

def test_text_processing():
    """Test text processing pipeline."""
    print("\nTesting text processing pipeline...")
    test_text = """
    # Python Programming
    Python is a high-level programming language.
    ## Functions
    Functions are defined using the def keyword:
    ```python
    def hello_world():
        print("Hello, World!")
    ```
    ## Variables
    Variables are created by assignment:
    ```python
    x = 10
    y = "Hello"
    ```
    """
    
    # Test text cleaner
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_text(test_text)
    print(f"✓ Text cleaning completed. Original: {len(test_text)} chars, Cleaned: {len(cleaned_text)} chars")
    
    # Test chunker
    chunker = Chunker()
    chunks = chunker.chunk_by_headings(cleaned_text, min_chars=50)
    print(f"✓ Text chunking completed. Created {len(chunks)} chunks")
    assert len(chunks) > 0, "Text processing pipeline failed"
    print("✓ Text processing pipeline is working")

def test_directory_creation():
    """Test directory creation functionality."""
    print("\nTesting directory creation...")
    test_dir = Path("test_output_dir")
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Successfully created directory: {test_dir}")
        
        # Create a test file
        test_file = test_dir / "test.txt"
        test_file.write_text("Test content")
        print(f"✓ Successfully created test file: {test_file}")
        
        # Clean up
        test_file.unlink()
        test_dir.rmdir()
        print("✓ Successfully cleaned up test directory")
    except Exception as e:
        assert False, f"Error with directory operations: {e}"

def main():
    """Run all tests."""
    print("Testing output generation functionality...\n")
    
    tests = [
        test_basic_output,
        test_file_discovery,
        test_text_processing,
        test_directory_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test {test.__name__} failed with assertion: {e}")
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Output generation should work.")
        print("\nTo test the full pipeline, run:")
        print("python scripts/prepare_data_modular.py --repo-path . --repo-name test-repo --output-dir ./output --verbose")
        return 0
    else:
        print("❌ Some tests failed. This might explain why no output files are created.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 