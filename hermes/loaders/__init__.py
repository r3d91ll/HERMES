"""
Loaders for various file formats.
"""

from .pdf_loader import PDFLoader
from .python_ast_loader import PythonASTLoader
from .docling_loader import DoclingLoader
from .text_loader import TextLoader
from .markdown_loader import MarkdownLoader
from .json_loader import JSONLoader
from .ocr_pdf_loader import OCRPDFLoader

__all__ = [
    "PDFLoader", 
    "PythonASTLoader", 
    "DoclingLoader", 
    "TextLoader",
    "MarkdownLoader",
    "JSONLoader",
    "OCRPDFLoader"
]