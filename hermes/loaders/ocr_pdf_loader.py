"""
OCR-enabled PDF loader for HERMES.
Handles scanned PDFs using OCR (Optical Character Recognition).
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import tempfile
import os

from pdf2image import convert_from_path
from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    
from hermes.core.base import BaseLoader

logger = logging.getLogger(__name__)


class OCRPDFLoader(BaseLoader):
    """
    Load scanned PDFs using OCR.
    
    This loader is specifically for PDFs that contain images of text
    rather than extractable text (common in older academic papers).
    """
    
    def __init__(
        self,
        dpi: int = 300,  # Higher DPI for better OCR
        language: str = 'eng',  # Tesseract language
        preprocess: bool = True,  # Apply image preprocessing
        page_segmentation_mode: int = 3,  # Tesseract PSM
    ):
        """
        Initialize OCR PDF loader.
        
        Args:
            dpi: DPI for PDF to image conversion
            language: Tesseract language code
            preprocess: Apply image preprocessing for better OCR
            page_segmentation_mode: Tesseract page segmentation mode
        """
        self.dpi = dpi
        self.language = language
        self.preprocess = preprocess
        self.psm = page_segmentation_mode
        
        if not TESSERACT_AVAILABLE:
            logger.warning("pytesseract not available. Install with: pip install pytesseract")
            logger.warning("Also ensure tesseract-ocr is installed on your system")
            
    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() == '.pdf'
        
    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        Load PDF using OCR.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document data with OCR text and metadata
        """
        if not self.can_load(file_path):
            raise ValueError(f"OCRPDFLoader can only load PDF files")
            
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract is required for OCR. Install with: pip install pytesseract")
            
        logger.info(f"Loading PDF with OCR: {file_path}")
        
        # Convert PDF to images
        try:
            images = convert_from_path(file_path, dpi=self.dpi)
            logger.info(f"Converted PDF to {len(images)} images at {self.dpi} DPI")
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            logger.info("Make sure poppler-utils is installed")
            raise
            
        # OCR each page
        page_texts = []
        page_confidences = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)} with OCR...")
            
            # Preprocess image if requested
            if self.preprocess:
                image = self._preprocess_image(image)
                
            # Perform OCR
            try:
                # Get text with confidence scores
                data = pytesseract.image_to_data(
                    image,
                    lang=self.language,
                    config=f'--psm {self.psm}',
                    output_type=pytesseract.Output.DICT
                )
                
                # Extract text
                text = pytesseract.image_to_string(
                    image,
                    lang=self.language,
                    config=f'--psm {self.psm}'
                )
                
                page_texts.append(f"[Page {i+1}]\n{text}")
                
                # Calculate average confidence for non-empty text
                confidences = [
                    conf for conf, txt in zip(data['conf'], data['text']) 
                    if txt.strip() and conf > 0
                ]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                page_confidences.append(avg_confidence)
                
            except Exception as e:
                logger.error(f"OCR failed on page {i+1}: {e}")
                page_texts.append(f"[Page {i+1}]\n[OCR FAILED]")
                page_confidences.append(0)
                
        # Combine results
        full_text = "\n\n".join(page_texts)
        avg_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0
        
        # Extract metadata
        metadata = self._extract_metadata(file_path, full_text, avg_confidence)
        
        return {
            'content': full_text,
            'page_count': len(images),
            'ocr_confidence': avg_confidence,
            'page_confidences': page_confidences,
            'metadata': metadata,
            'file_path': str(file_path),
            'loader': 'OCRPDFLoader',
            'processing_info': {
                'dpi': self.dpi,
                'language': self.language,
                'preprocessed': self.preprocess
            }
        }
        
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
            
        # Could add more preprocessing here:
        # - Deskewing
        # - Denoising
        # - Contrast enhancement
        # - Binarization
        
        return image
        
    def _extract_metadata(self, file_path: Path, text: str, confidence: float) -> Dict[str, Any]:
        """Extract metadata from OCR results."""
        lines = text.splitlines()
        
        metadata = {
            'ocr_performed': True,
            'ocr_confidence': confidence,
            'ocr_quality': self._assess_quality(confidence),
            'line_count': len(lines),
            'word_count': len(text.split()),
            'character_count': len(text),
            'likely_language': self._detect_language(text[:1000]),
            'has_equations': self._detect_equations(text),
            'has_tables': self._detect_tables(text),
            'is_readable': confidence > 70
        }
        
        # Add filesystem metadata
        stats = file_path.stat()
        metadata.update({
            'file_size': stats.st_size,
            'created_time': stats.st_ctime,
            'modified_time': stats.st_mtime
        })
        
        return metadata
        
    def _assess_quality(self, confidence: float) -> str:
        """Assess OCR quality based on confidence."""
        if confidence >= 90:
            return "excellent"
        elif confidence >= 80:
            return "good"
        elif confidence >= 70:
            return "fair"
        elif confidence >= 60:
            return "poor"
        else:
            return "very_poor"
            
    def _detect_language(self, sample: str) -> str:
        """Simple language detection."""
        # Could use langdetect here
        english_words = {'the', 'is', 'and', 'to', 'of', 'in', 'a', 'that'}
        words = set(sample.lower().split())
        
        if len(words & english_words) >= 3:
            return 'english'
            
        return 'unknown'
        
    def _detect_equations(self, text: str) -> bool:
        """Detect mathematical equations in OCR text."""
        # Look for common math symbols that survive OCR
        math_indicators = ['=', '∑', '∫', '+', '×', '÷', 'Σ', 'π', 'θ', 'α', 'β']
        math_count = sum(1 for indicator in math_indicators if indicator in text)
        
        # Also check for equation-like patterns
        import re
        equation_patterns = [
            r'[a-zA-Z]\s*=\s*[^=]',  # x = ...
            r'\d+\s*[+\-*/]\s*\d+',  # arithmetic
            r'[A-Z]\([^)]+\)',        # F(x), P(A), etc.
        ]
        
        pattern_count = sum(
            1 for pattern in equation_patterns 
            if re.search(pattern, text)
        )
        
        return (math_count > 5) or (pattern_count > 3)
        
    def _detect_tables(self, text: str) -> bool:
        """Detect tables in OCR text."""
        lines = text.splitlines()
        
        # Look for table-like patterns
        table_indicators = 0
        
        for line in lines:
            # Multiple spaces or tabs might indicate columns
            if line.count('  ') > 2 or '\t' in line:
                table_indicators += 1
            # Vertical bars often survive OCR in tables
            if '|' in line:
                table_indicators += 1
                
        return table_indicators > 5