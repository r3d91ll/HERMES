#!/usr/bin/env python3
"""
Test single PDF for OCR needs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hermes.loaders import PDFLoader

def test_single_pdf():
    file_path = Path("/home/todd/olympus/data/papers/1965_Kolmogorov_Three_Approaches_Information.pdf")
    
    loader = PDFLoader()
    result = loader.load(file_path)
    
    text = result.get('text', '')
    page_count = result.get('page_count', 1)
    
    print(f"File: {file_path.name}")
    print(f"Pages: {page_count}")
    print(f"Characters extracted: {len(text)}")
    print(f"Characters per page: {len(text)/page_count if page_count > 0 else 0:.1f}")
    
    if len(text) < 100:
        print("\n❌ This PDF needs OCR - no text extracted")
        print("\nOptions:")
        print("1. Use OCRPDFLoader (requires pytesseract)")
        print("2. Use DoclingLoader with OCR enabled")
        print("3. Pre-process with ocrmypdf")
    else:
        print("\n✓ Text extracted successfully")
        print("\nFirst 300 characters:")
        print("-" * 40)
        print(text[:300])

if __name__ == "__main__":
    test_single_pdf()