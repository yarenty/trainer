#!/usr/bin/env python3
"""
Debug script to check file discovery and processing.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from file_processor import FileProcessor

def debug_file_discovery():
    """Debug file discovery in current directory."""
    print("=== File Discovery Debug ===\n")
    
    current_dir = Path(".")
    print(f"Current directory: {current_dir.absolute()}")
    
    # Check what files exist
    print("\nAll files in current directory:")
    for item in current_dir.iterdir():
        if item.is_file():
            print(f"  File: {item}")
        elif item.is_dir():
            print(f"  Dir:  {item}")
    
    # Test the file discovery methods
    processor = FileProcessor(None, "test", max_workers=1)
    
    print(f"\n=== Testing documentation file discovery ===")
    doc_files = processor._find_documentation_files(str(current_dir))
    print(f"Found {len(doc_files)} documentation files:")
    for file in doc_files:
        print(f"  {file}")
    
    print(f"\n=== Testing code file discovery ===")
    code_files = processor._find_code_files(str(current_dir))
    print(f"Found {len(code_files)} code files:")
    for file in code_files:
        print(f"  {file}")
    
    total_files = len(doc_files) + len(code_files)
    print(f"\nTotal files found: {total_files}")
    
    if total_files == 0:
        print("\n❌ No files found! This is why no output is generated.")
        print("Possible reasons:")
        print("1. No files with supported extensions in the directory")
        print("2. Files are in subdirectories that are being skipped")
        print("3. File discovery logic has an issue")
        
        # Check what extensions are supported
        print("\nSupported documentation extensions: .md, .rst, .txt, .adoc, .asciidoc")
        print("Supported code extensions: .py, .js, .ts, .java, .cpp, .c, .h, .hpp, .rs, .go, .php, .rb, .swift, .kt, .scala, .cs, .fs, .clj, .hs")
        
        # Check if there are any files with these extensions
        all_files = list(current_dir.rglob("*"))
        print(f"\nAll files recursively: {len(all_files)}")
        for file in all_files[:10]:  # Show first 10
            if file.is_file():
                print(f"  {file} (ext: {file.suffix})")
        
    else:
        print("\n✅ Files found! The issue might be elsewhere.")
    
    return total_files > 0

def debug_text_processing():
    """Debug text processing with a sample file."""
    print("\n=== Text Processing Debug ===")
    
    # Find a sample file to test
    sample_file = None
    for ext in ['.py', '.md', '.txt']:
        files = list(Path(".").glob(f"*{ext}"))
        if files:
            sample_file = files[0]
            break
    
    if not sample_file:
        print("❌ No sample file found for testing")
        return False
    
    print(f"Testing with file: {sample_file}")
    
    try:
        # Read the file
        with open(sample_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        print(f"File size: {len(content)} characters")
        
        if len(content.strip()) == 0:
            print("❌ File is empty")
            return False
        
        # Test text cleaning
        from text_cleaner import TextCleaner
        cleaner = TextCleaner()
        cleaned_content = cleaner.clean_text(content)
        print(f"Cleaned content size: {len(cleaned_content)} characters")
        
        if len(cleaned_content.strip()) == 0:
            print("❌ Content was completely cleaned away")
            return False
        
        # Test chunking
        from chunker import Chunker
        chunker = Chunker()
        chunks = chunker.chunk_by_headings(cleaned_content, min_chars=50)
        print(f"Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print("❌ No chunks created")
            return False
        
        print("✅ Text processing pipeline works")
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def main():
    """Run debug checks."""
    print("Debugging file discovery and processing...\n")
    
    # Check file discovery
    files_found = debug_file_discovery()
    
    # Check text processing
    processing_works = debug_text_processing()
    
    print(f"\n=== Summary ===")
    print(f"Files found: {'✅' if files_found else '❌'}")
    print(f"Processing works: {'✅' if processing_works else '❌'}")
    
    if not files_found:
        print("\n🔧 To fix the 'no output' issue:")
        print("1. Make sure you're running the script from a directory with files")
        print("2. Check that files have supported extensions")
        print("3. Try running with --verbose to see more details")
        print("\nExample command:")
        print("python scripts/prepare_data_modular.py --repo-path . --repo-name test --output-dir ./output --verbose")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 