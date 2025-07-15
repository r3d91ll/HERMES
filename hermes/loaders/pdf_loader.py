"""
PDF loader for HERMES.

Extracts text and images from PDFs for processing by Jina v4.
"""

from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import logging
from PIL import Image
import io
import pypdf
import pdfplumber
from pdf2image import convert_from_path

from hermes.core.base import BaseLoader


logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """
    Load PDFs and extract content for embedding.
    
    Jina v4 can handle:
    - Raw text (extracted from PDF)
    - Images of pages (for visual PDFs with charts/tables)
    - Mixed content (text + images)
    
    This loader prepares PDFs in formats Jina can process.
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        page_as_image: bool = False,  # Convert pages to images for visual docs
        dpi: int = 150,  # DPI for page-to-image conversion
    ):
        """
        Initialize PDF loader.
        
        Args:
            extract_images: Extract embedded images from PDF
            extract_tables: Attempt to extract tables as structured data
            page_as_image: Convert PDF pages to images (for visual documents)
            dpi: DPI for page-to-image conversion
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.page_as_image = page_as_image
        self.dpi = dpi
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a PDF and extract its content.
        
        Returns:
            Dictionary with:
            - text: Extracted text content
            - images: List of PIL images (if extract_images=True)
            - page_images: List of page images (if page_as_image=True)
            - tables: Extracted tables (if extract_tables=True)
            - metadata: PDF metadata
            - page_count: Number of pages
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        
        result = {
            "source_path": str(path),
            "text": "",
            "images": [],
            "page_images": [],
            "tables": [],
            "metadata": {},
            "page_count": 0,
            "content_type": "mixed" if self.page_as_image else "text"
        }
        
        # Extract text and metadata using pypdf
        try:
            with open(path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                result["page_count"] = len(pdf.pages)
                
                # Extract metadata
                if pdf.metadata:
                    result["metadata"] = {
                        "title": pdf.metadata.get("/Title", ""),
                        "author": pdf.metadata.get("/Author", ""),
                        "subject": pdf.metadata.get("/Subject", ""),
                        "creator": pdf.metadata.get("/Creator", ""),
                        "producer": pdf.metadata.get("/Producer", ""),
                        "creation_date": str(pdf.metadata.get("/CreationDate", "")),
                        "modification_date": str(pdf.metadata.get("/ModDate", "")),
                    }
                
                # Extract text from each page
                text_parts = []
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"[Page {i+1}]\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {i+1}: {e}")
                
                result["text"] = "\n\n".join(text_parts)
                
        except Exception as e:
            logger.error(f"Failed to read PDF with pypdf: {e}")
        
        # Extract tables using pdfplumber
        if self.extract_tables:
            try:
                with pdfplumber.open(path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        for j, table in enumerate(tables):
                            result["tables"].append({
                                "page": i + 1,
                                "table_index": j,
                                "data": table
                            })
            except Exception as e:
                logger.warning(f"Failed to extract tables: {e}")
        
        # Convert pages to images for visual documents
        if self.page_as_image:
            try:
                # This requires poppler-utils to be installed
                page_images = convert_from_path(path, dpi=self.dpi)
                result["page_images"] = page_images
                
                # For visual PDFs, we might want both text and images
                result["content_type"] = "visual_document"
                
            except Exception as e:
                logger.warning(f"Failed to convert PDF to images: {e}")
                logger.info("Install poppler-utils for visual PDF support")
        
        # Extract embedded images (if needed)
        if self.extract_images and not self.page_as_image:
            # This is complex and would require additional libraries
            # For now, log that this feature needs implementation
            logger.info("Embedded image extraction not yet implemented")
        
        return result
    
    def can_load(self, path: Union[str, Path]) -> bool:
        """Check if this loader can handle the given path."""
        path = Path(path)
        return path.suffix.lower() in [".pdf"]
    
    def prepare_for_jina(self, pdf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare PDF data for Jina v4 embedding.
        
        Jina v4 can process:
        1. Text directly
        2. Images (including PDF pages as images)
        3. Mixed content
        
        Returns:
            Dictionary with:
            - items: List of content items to embed
            - item_types: Type of each item ("text" or "image")
            - metadata: Preserved metadata
        """
        items = []
        item_types = []
        
        # Decide strategy based on content type
        if pdf_data["content_type"] == "visual_document" and pdf_data["page_images"]:
            # For visual PDFs, use page images
            # This preserves layout, charts, tables, etc.
            for i, page_img in enumerate(pdf_data["page_images"]):
                # Save image temporarily or pass PIL image directly
                items.append(page_img)
                item_types.append("image")
        
        elif pdf_data["text"]:
            # For text-heavy PDFs, use extracted text
            # Could chunk here or use Jina's late chunking
            items.append(pdf_data["text"])
            item_types.append("text")
        
        # Add tables as structured text if present
        if pdf_data["tables"]:
            for table_info in pdf_data["tables"]:
                # Convert table to markdown or structured text
                table_text = self._table_to_text(table_info["data"])
                items.append(f"[Table from page {table_info['page']}]\n{table_text}")
                item_types.append("text")
        
        return {
            "items": items,
            "item_types": item_types,
            "metadata": pdf_data["metadata"],
            "source_path": pdf_data["source_path"],
            "page_count": pdf_data["page_count"]
        }
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to markdown format."""
        if not table_data:
            return ""
        
        # Simple markdown table
        lines = []
        
        # Header
        if len(table_data) > 0:
            header = table_data[0]
            lines.append(" | ".join(str(cell) for cell in header))
            lines.append(" | ".join("---" for _ in header))
        
        # Rows
        for row in table_data[1:]:
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)