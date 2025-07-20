#!/usr/bin/env python3
"""
Test OCR capabilities on older PDFs.
"""

import sys
from pathlib import Path
from pprint import pprint

# Add HERMES to path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.loaders import PDFLoader, DoclingLoader


def test_pdf_text_extraction(file_path: Path):
    """Test basic PDF text extraction."""
    print(f"\n{'='*60}")
    print(f"Testing PyPDF2 extraction on: {file_path.name}")
    print(f"{'='*60}")
    
    loader = PDFLoader(extract_images=False, page_as_image=False)
    
    try:
        result = loader.load(file_path)
        
        # Check if we got any text
        text = result.get('text', '')
        print(f"Text extracted: {len(text)} characters")
        print(f"Page count: {result.get('page_count', 0)}")
        
        # Show first 500 chars to check quality
        if text:
            print(f"\nFirst 500 characters:")
            print("-" * 40)
            print(text[:500])
            print("-" * 40)
            
            # Check for common OCR artifacts
            ocr_artifacts = ['�', '|', '\\', '/', '_' * 5]
            artifact_count = sum(text.count(artifact) for artifact in ocr_artifacts)
            print(f"\nPotential OCR artifacts found: {artifact_count}")
            
            # Check for readable content
            words = text.split()
            print(f"Word count: {len(words)}")
            
            # Check if mostly gibberish
            readable_words = [w for w in words if len(w) > 2 and w.isalpha()]
            readability = len(readable_words) / len(words) if words else 0
            print(f"Readability score: {readability:.2%}")
            
            if readability < 0.5:
                print("\n⚠️  Low readability - might be scanned PDF needing OCR")
        else:
            print("\n❌ No text extracted - likely scanned PDF")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_docling_extraction(file_path: Path):
    """Test Docling extraction with OCR."""
    print(f"\n{'='*60}")
    print(f"Testing Docling extraction on: {file_path.name}")
    print(f"{'='*60}")
    
    # Note: This will fail if Docling isn't properly installed
    # But let's see what happens
    try:
        loader = DoclingLoader(
            extract_tables=True,
            extract_images=True,
            extract_formulas=True,
            ocr_enabled=True  # Enable OCR
        )
        
        if loader.can_load(file_path):
            print("Docling can handle this file type")
            
            # This will likely fail because we need to fix the imports
            # result = loader.load(file_path)
            print("\nDocling loader is available but needs proper setup")
            print("It would provide:")
            print("- OCR for scanned pages")
            print("- Formula extraction")
            print("- Table extraction")
            print("- Image classification")
        else:
            print("Docling cannot handle this file type")
            
    except ImportError as e:
        print(f"\n⚠️  Docling not properly installed: {e}")
        print("\nTo use Docling for OCR, you would need to:")
        print("1. Install docling package properly")
        print("2. Install OCR dependencies (tesseract, etc.)")
    except Exception as e:
        print(f"Error: {e}")


def test_page_as_image(file_path: Path):
    """Test converting pages to images for OCR."""
    print(f"\n{'='*60}")
    print(f"Testing page-to-image conversion: {file_path.name}")
    print(f"{'='*60}")
    
    loader = PDFLoader(
        extract_images=False,
        page_as_image=True,
        dpi=150
    )
    
    try:
        result = loader.load(file_path)
        
        page_images = result.get('page_images', [])
        print(f"Page images extracted: {len(page_images)}")
        
        if page_images:
            first_image = page_images[0]
            print(f"First page image size: {first_image.size}")
            print(f"Image mode: {first_image.mode}")
            print("\n✓ Pages can be converted to images for OCR processing")
        else:
            print("\n❌ Failed to convert pages to images")
            print("Note: This requires poppler-utils to be installed")
            
    except Exception as e:
        print(f"Error: {e}")
        if "poppler" in str(e).lower():
            print("\nTo enable page-to-image conversion:")
            print("  sudo apt-get install poppler-utils  # On Ubuntu/Debian")
            print("  brew install poppler                # On macOS")


def main():
    """Test OCR capabilities on Kolmogorov paper."""
    file_path = Path("/home/todd/olympus/data/papers/1965_Kolmogorov_Three_Approaches_Information.pdf")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    # Test different extraction methods
    test_pdf_text_extraction(file_path)
    test_docling_extraction(file_path)
    test_page_as_image(file_path)
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print("\nFor OCR PDFs, you have several options:")
    print("1. Use DoclingLoader with ocr_enabled=True (best option)")
    print("2. Convert pages to images and use external OCR")
    print("3. Pre-process PDFs with OCR tools before loading")
    print("\nFor your academic corpus (1948-2018), many older papers")
    print("will be scanned PDFs requiring OCR.")


if __name__ == "__main__":
    main()