"""
Loaders for various file formats.
"""

from .pdf_loader import PDFLoader
from .python_ast_loader import PythonASTLoader
from .docling_loader import DoclingLoader

# Alias for backward compatibility
TextLoader = DoclingLoader  # DoclingLoader can handle text files

__all__ = ["PDFLoader", "PythonASTLoader", "DoclingLoader", "TextLoader"]