"""
Text file loader for HERMES pipeline.
Handles .txt files with encoding detection and metadata extraction.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import chardet

from hermes.core.base import BaseLoader

logger = logging.getLogger(__name__)


class TextLoader(BaseLoader):
    """Load and process plain text files."""
    
    def __init__(self):
        """Initialize text loader."""
        self.supported_extensions = ['.txt', '.text']
        
    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() in self.supported_extensions
        
    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        Load text file with encoding detection.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Document data with content and metadata
        """
        if not self.can_load(file_path):
            raise ValueError(f"TextLoader cannot load {file_path.suffix} files")
            
        logger.info(f"Loading text file: {file_path}")
        
        # Detect encoding
        encoding = self._detect_encoding(file_path)
        
        # Read content
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode with {encoding}, trying latin-1")
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Extract metadata
        metadata = self._extract_metadata(file_path, content, encoding)
        
        return {
            'content': content,
            'metadata': metadata,
            'file_path': str(file_path),
            'loader': 'TextLoader'
        }
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.debug(f"Detected encoding: {encoding} (confidence: {confidence})")
                
                # Use UTF-8 if confidence is low
                if confidence < 0.7:
                    return 'utf-8'
                    
                return encoding or 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
            return 'utf-8'
    
    def _extract_metadata(self, file_path: Path, content: str, encoding: str) -> Dict[str, Any]:
        """Extract metadata from text file."""
        lines = content.splitlines()
        
        # Count various text characteristics
        metadata = {
            'encoding': encoding,
            'line_count': len(lines),
            'word_count': len(content.split()),
            'character_count': len(content),
            'average_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'max_line_length': max(len(line) for line in lines) if lines else 0,
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'has_headers': self._detect_headers(lines),
            'language': self._detect_language(content[:1000])  # First 1KB
        }
        
        # Add filesystem metadata
        stats = file_path.stat()
        metadata.update({
            'file_size': stats.st_size,
            'created_time': stats.st_ctime,
            'modified_time': stats.st_mtime,
            'permissions': oct(stats.st_mode)[-3:]
        })
        
        return metadata
    
    def _detect_headers(self, lines: list) -> bool:
        """Simple heuristic to detect if text has headers/structure."""
        if len(lines) < 3:
            return False
            
        # Check for markdown-style headers
        for line in lines[:20]:  # Check first 20 lines
            if line.strip().startswith('#') or line.strip().startswith('=='):
                return True
                
        # Check for numbered sections
        for line in lines[:20]:
            if line.strip() and (
                line.strip()[0].isdigit() or 
                line.strip().lower().startswith(('chapter', 'section'))
            ):
                return True
                
        return False
    
    def _detect_language(self, sample: str) -> str:
        """Simple language detection based on common words."""
        # This is a placeholder - in production, use langdetect or similar
        english_words = {'the', 'is', 'and', 'to', 'of', 'in', 'a', 'that'}
        words = set(sample.lower().split())
        
        if len(words & english_words) >= 3:
            return 'en'
        
        return 'unknown'