#!/usr/bin/env python3
"""
Test if PDFs need OCR by checking text extraction.
"""

import sys
from pathlib import Path

# Add HERMES to path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.loaders import PDFLoader


def needs_ocr(file_path: Path, min_chars_per_page: int = 100) -> bool:
    """
    Check if a PDF needs OCR.
    
    Args:
        file_path: Path to PDF
        min_chars_per_page: Minimum characters per page to consider it readable
        
    Returns:
        True if PDF likely needs OCR
    """
    loader = PDFLoader()
    
    try:
        result = loader.load(file_path)
        text = result.get('text', '')
        page_count = result.get('page_count', 1)
        
        # Calculate average characters per page
        if page_count > 0:
            chars_per_page = len(text) / page_count
            
            print(f"\nFile: {file_path.name}")
            print(f"Pages: {page_count}")
            print(f"Total characters: {len(text)}")
            print(f"Chars per page: {chars_per_page:.1f}")
            
            if chars_per_page < min_chars_per_page:
                print("❌ Needs OCR - too little text extracted")
                return True
            else:
                print("✓ Text extraction successful")
                return False
        else:
            print(f"❌ No pages found in {file_path.name}")
            return True
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return True


def scan_papers_directory():
    """Scan the papers directory and identify which need OCR."""
    papers_dir = Path("/home/todd/olympus/data/papers")
    
    needs_ocr_list = []
    text_ok_list = []
    
    print("Scanning papers for OCR requirements...")
    print("=" * 60)
    
    for pdf_file in sorted(papers_dir.glob("*.pdf")):
        if needs_ocr(pdf_file):
            needs_ocr_list.append(pdf_file.name)
        else:
            text_ok_list.append(pdf_file.name)
            
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n✓ PDFs with extractable text ({len(text_ok_list)}):")
    for name in text_ok_list[:5]:  # Show first 5
        print(f"  - {name}")
    if len(text_ok_list) > 5:
        print(f"  ... and {len(text_ok_list) - 5} more")
        
    print(f"\n❌ PDFs needing OCR ({len(needs_ocr_list)}):")
    for name in needs_ocr_list[:5]:  # Show first 5
        print(f"  - {name}")
    if len(needs_ocr_list) > 5:
        print(f"  ... and {len(needs_ocr_list) - 5} more")
        
    print(f"\nOCR needed for {len(needs_ocr_list)}/{len(needs_ocr_list) + len(text_ok_list)} papers ({len(needs_ocr_list)/(len(needs_ocr_list) + len(text_ok_list))*100:.1f}%)")


if __name__ == "__main__":
    scan_papers_directory()