"""
IBM Docling integration for HERMES.

Docling provides advanced document understanding with:
- Page layout analysis
- Table structure extraction
- Formula/code detection
- Image classification
- Reading order detection
"""

from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import logging
import json

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult

from hermes.core.base import BaseLoader


logger = logging.getLogger(__name__)


class DoclingLoader(BaseLoader):
    """
    Advanced document loader using IBM's Docling.
    
    Docling handles complex PDFs with:
    - Tables with structure preservation
    - Mathematical formulas
    - Code blocks
    - Charts and figures
    - Multi-column layouts
    - Reading order detection
    """
    
    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True,
        extract_formulas: bool = True,
        extract_code: bool = True,
        ocr_enabled: bool = True,
        chunk_size: Optional[int] = None,  # Let Docling handle chunking
    ):
        """
        Initialize Docling loader.
        
        Args:
            extract_tables: Extract and structure tables
            extract_images: Extract and classify images
            extract_formulas: Extract mathematical formulas
            extract_code: Extract code blocks
            ocr_enabled: Enable OCR for scanned documents
            chunk_size: Optional chunk size for splitting documents
        """
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.extract_formulas = extract_formulas
        self.extract_code = extract_code
        self.ocr_enabled = ocr_enabled
        self.chunk_size = chunk_size
        
        # Initialize Docling processing service
        self.processor = ProcessingService(
            enable_ocr=ocr_enabled,
            extract_tables=extract_tables,
            extract_images=extract_images,
            extract_formulas=extract_formulas,
            extract_code=extract_code,
        )
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a document using Docling.
        
        Returns:
            Dictionary with:
            - document: DoclingDocument object
            - text: Full text content
            - chunks: Document chunks (if chunk_size set)
            - tables: Extracted tables with structure
            - images: Extracted images with classifications
            - formulas: Mathematical formulas
            - code_blocks: Code snippets
            - metadata: Document metadata
            - layout: Page layout information
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        
        try:
            # Process document with Docling
            doc = self.processor.process(str(path))
            
            result = {
                "source_path": str(path),
                "document": doc,  # Keep original DoclingDocument
                "text": "",
                "chunks": [],
                "tables": [],
                "images": [],
                "formulas": [],
                "code_blocks": [],
                "metadata": {},
                "layout": {},
                "content_type": "structured_document"
            }
            
            # Extract full text
            result["text"] = doc.get_text()
            
            # Extract metadata
            result["metadata"] = {
                "title": doc.metadata.get("title", ""),
                "authors": doc.metadata.get("authors", []),
                "abstract": doc.metadata.get("abstract", ""),
                "language": doc.metadata.get("language", ""),
                "page_count": doc.metadata.get("page_count", 0),
                "document_type": doc.metadata.get("document_type", ""),
            }
            
            # Extract structured elements
            if self.extract_tables:
                result["tables"] = self._extract_tables(doc)
            
            if self.extract_images:
                result["images"] = self._extract_images(doc)
            
            if self.extract_formulas:
                result["formulas"] = self._extract_formulas(doc)
            
            if self.extract_code:
                result["code_blocks"] = self._extract_code(doc)
            
            # Extract layout information
            result["layout"] = self._extract_layout(doc)
            
            # Chunk if requested
            if self.chunk_size:
                result["chunks"] = self._chunk_document(doc)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process document with Docling: {e}")
            raise
    
    def can_load(self, path: Union[str, Path]) -> bool:
        """Check if Docling can handle this file type."""
        path = Path(path)
        supported_extensions = {
            ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md",
            ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
            ".wav", ".mp3", ".m4a",  # Audio support
        }
        return path.suffix.lower() in supported_extensions
    
    def prepare_for_jina(self, docling_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare Docling output for Jina v4 embedding.
        
        Docling's structured output maps well to Jina's capabilities:
        - Text content → Jina text embedding
        - Tables → Structured text or markdown
        - Images → Jina image embedding
        - Formulas → Special handling (text or image)
        
        Returns:
            Dictionary ready for Jina v4 processing
        """
        items = []
        item_types = []
        item_metadata = []
        
        # Main text content
        if docling_data["text"]:
            items.append(docling_data["text"])
            item_types.append("text")
            item_metadata.append({"type": "main_content"})
        
        # Tables as structured text
        for table in docling_data["tables"]:
            table_text = self._table_to_markdown(table)
            items.append(table_text)
            item_types.append("text")
            item_metadata.append({
                "type": "table",
                "page": table.get("page", -1),
                "caption": table.get("caption", "")
            })
        
        # Code blocks
        for code in docling_data["code_blocks"]:
            code_text = f"```{code.get('language', '')}\n{code['content']}\n```"
            items.append(code_text)
            item_types.append("text")  # Could use "code" adapter
            item_metadata.append({
                "type": "code",
                "language": code.get("language", ""),
                "page": code.get("page", -1)
            })
        
        # Formulas
        for formula in docling_data["formulas"]:
            # LaTeX formulas as text
            formula_text = f"$${formula['latex']}$$"
            items.append(formula_text)
            item_types.append("text")
            item_metadata.append({
                "type": "formula",
                "page": formula.get("page", -1)
            })
        
        # Images (if paths are available)
        for image in docling_data["images"]:
            if "path" in image:
                items.append(Path(image["path"]))
                item_types.append("image")
                item_metadata.append({
                    "type": "image",
                    "classification": image.get("classification", ""),
                    "caption": image.get("caption", ""),
                    "page": image.get("page", -1)
                })
        
        return {
            "items": items,
            "item_types": item_types,
            "item_metadata": item_metadata,
            "document_metadata": docling_data["metadata"],
            "source_path": docling_data["source_path"]
        }
    
    def _extract_tables(self, doc: ConversionResult) -> List[Dict[str, Any]]:
        """Extract tables from DoclingDocument."""
        tables = []
        for table in doc.tables:
            tables.append({
                "content": table.data,  # Structured table data
                "caption": table.caption,
                "page": table.page_number,
                "bbox": table.bounding_box,
                "rows": table.num_rows,
                "columns": table.num_cols,
            })
        return tables
    
    def _extract_images(self, doc: ConversionResult) -> List[Dict[str, Any]]:
        """Extract images with classifications."""
        images = []
        for image in doc.images:
            images.append({
                "path": image.path,  # If saved
                "classification": image.classification,  # Chart, diagram, photo, etc.
                "caption": image.caption,
                "page": image.page_number,
                "bbox": image.bounding_box,
            })
        return images
    
    def _extract_formulas(self, doc: ConversionResult) -> List[Dict[str, Any]]:
        """Extract mathematical formulas."""
        formulas = []
        for formula in doc.formulas:
            formulas.append({
                "latex": formula.latex,
                "mathml": formula.mathml,  # If available
                "page": formula.page_number,
                "bbox": formula.bounding_box,
                "inline": formula.is_inline,
            })
        return formulas
    
    def _extract_code(self, doc: ConversionResult) -> List[Dict[str, Any]]:
        """Extract code blocks."""
        code_blocks = []
        for code in doc.code_blocks:
            code_blocks.append({
                "content": code.content,
                "language": code.language,
                "page": code.page_number,
                "bbox": code.bounding_box,
            })
        return code_blocks
    
    def _extract_layout(self, doc: ConversionResult) -> Dict[str, Any]:
        """Extract document layout information."""
        return {
            "pages": doc.num_pages,
            "columns": doc.layout_type,  # single, double, etc.
            "reading_order": doc.reading_order,  # Sequence of elements
            "sections": [
                {
                    "type": section.type,  # header, paragraph, list, etc.
                    "level": section.level,
                    "page": section.page_number,
                    "text": section.text[:100] + "..." if len(section.text) > 100 else section.text
                }
                for section in doc.sections
            ]
        }
    
    def _chunk_document(self, doc: ConversionResult) -> List[Dict[str, Any]]:
        """Chunk document while preserving structure."""
        chunks = []
        
        # Use Docling's built-in chunking that respects document structure
        for i, chunk in enumerate(doc.chunks(self.chunk_size)):
            chunks.append({
                "index": i,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "elements": chunk.elements,  # Tables, images, etc. in this chunk
            })
        
        return chunks
    
    def _table_to_markdown(self, table: Dict[str, Any]) -> str:
        """Convert table to markdown format."""
        lines = []
        
        if table.get("caption"):
            lines.append(f"**Table: {table['caption']}**\n")
        
        data = table["content"]
        if not data:
            return ""
        
        # Header
        if len(data) > 0:
            header = data[0]
            lines.append(" | ".join(str(cell) for cell in header))
            lines.append(" | ".join("---" for _ in header))
        
        # Rows
        for row in data[1:]:
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)