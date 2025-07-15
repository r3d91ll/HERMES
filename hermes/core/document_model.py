"""
Document model and schema for HERMES pipeline.

This module defines the unified document representation that flows through
the entire HERMES pipeline, from loading through embedding to storage.

The document model embodies HADES' dimensional theory:
- WHERE: Physical location and structure
- WHAT: Semantic content and meaning
- CONVEYANCE: Implementation quality and actionability
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import numpy as np
from enum import Enum


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    PYTHON = "python"
    MARKDOWN = "markdown"
    JUPYTER = "jupyter"
    TEXT = "text"
    CODE = "code"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"


class ComplexityLevel(Enum):
    """Document complexity levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    UNKNOWN = "unknown"


@dataclass
class Location:
    """
    WHERE dimension: Physical and logical location.
    
    This captures both the absolute physical location (file path)
    and the relative logical location (within project structure).
    """
    absolute_path: Path
    relative_path: Optional[Path] = None
    directory_chain: List[str] = field(default_factory=list)
    file_size: Optional[int] = None
    last_modified: Optional[datetime] = None
    
    # For code files
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    # For chunked documents
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "absolute_path": str(self.absolute_path),
            "relative_path": str(self.relative_path) if self.relative_path else None,
            "directory_chain": self.directory_chain,
            "file_size": self.file_size,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks
        }


@dataclass
class Content:
    """
    WHAT dimension: Semantic content and meaning.
    
    This captures the actual content, its semantic properties,
    and extracted concepts.
    """
    raw_text: str
    cleaned_text: Optional[str] = None
    
    # Semantic properties
    concepts: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)  # Named entities
    
    # For code
    symbols: Optional[Dict[str, Any]] = None  # AST symbols
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Domain classification
    domains: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    
    # Complexity and prerequisites
    complexity: ComplexityLevel = ComplexityLevel.UNKNOWN
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "concepts": self.concepts,
            "keywords": self.keywords,
            "entities": self.entities,
            "symbols": self.symbols,
            "imports": self.imports,
            "dependencies": self.dependencies,
            "domains": self.domains,
            "topics": self.topics,
            "complexity": self.complexity.value,
            "prerequisites": self.prerequisites
        }


@dataclass
class Conveyance:
    """
    CONVEYANCE dimension: Implementation quality and actionability.
    
    This captures how well the content conveys its concepts
    and how actionable it is for practitioners.
    """
    # Core metrics
    implementation_fidelity: float = 0.0  # How well does it implement concepts?
    actionability: float = 0.0  # How actionable is the content?
    bridge_potential: float = 0.0  # Theory-practice bridge potential
    
    # Quality indicators
    completeness: float = 0.0  # Is it complete?
    clarity: float = 0.0  # Is it clear?
    correctness: float = 0.0  # Is it correct?
    
    # For code
    test_coverage: Optional[float] = None
    cyclomatic_complexity: Optional[int] = None
    maintainability_index: Optional[float] = None
    
    # Context amplification potential
    amplification_score: float = 0.0  # How much does context help?
    
    # Relationships that enhance conveyance
    enhances: List[str] = field(default_factory=list)  # Doc IDs this enhances
    enhanced_by: List[str] = field(default_factory=list)  # Doc IDs that enhance this
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "implementation_fidelity": self.implementation_fidelity,
            "actionability": self.actionability,
            "bridge_potential": self.bridge_potential,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "correctness": self.correctness,
            "test_coverage": self.test_coverage,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "maintainability_index": self.maintainability_index,
            "amplification_score": self.amplification_score,
            "enhances": self.enhances,
            "enhanced_by": self.enhanced_by
        }


@dataclass
class Embeddings:
    """
    Vector embeddings from different models.
    
    Supports both full embeddings (Jina) and dimensional embeddings (HADES).
    """
    # Jina v4 embeddings (2048 dimensions)
    jina_semantic: Optional[np.ndarray] = None
    jina_task_adapted: Optional[Dict[str, np.ndarray]] = None  # Task-specific
    
    # HADES dimensional embeddings
    hades_where: Optional[np.ndarray] = None  # 102 dims
    hades_what: Optional[np.ndarray] = None  # 1024 dims
    hades_conveyance: Optional[np.ndarray] = None  # 922 dims
    hades_full: Optional[np.ndarray] = None  # 2048 dims concatenated
    
    # Metadata about embeddings
    embedding_model_versions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "embedding_model_versions": self.embedding_model_versions
        }
        
        # Convert numpy arrays to lists for JSON storage
        if self.jina_semantic is not None:
            result["jina_v4_semantic"] = self.jina_semantic.tolist()
            
        if self.jina_task_adapted:
            result["jina_v4_task_adapted"] = {
                task: emb.tolist() for task, emb in self.jina_task_adapted.items()
            }
        
        if any([self.hades_where is not None, 
                self.hades_what is not None, 
                self.hades_conveyance is not None]):
            result["hades_dimensional"] = {}
            if self.hades_where is not None:
                result["hades_dimensional"]["where_vector"] = self.hades_where.tolist()
            if self.hades_what is not None:
                result["hades_dimensional"]["what_vector"] = self.hades_what.tolist()
            if self.hades_conveyance is not None:
                result["hades_dimensional"]["conveyance_vector"] = self.hades_conveyance.tolist()
        
        if self.hades_full is not None:
            result["hades_full"] = self.hades_full.tolist()
            
        return result


@dataclass
class Document:
    """
    Unified document representation for HERMES pipeline.
    
    This is the core data structure that flows through the entire pipeline,
    accumulating information at each stage.
    """
    # Unique identifier
    doc_id: str
    
    # Document type
    doc_type: DocumentType
    
    # Core dimensions
    location: Location
    content: Content
    conveyance: Conveyance
    
    # Embeddings
    embeddings: Optional[Embeddings] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline state
    pipeline_stage: str = "loaded"
    processing_timestamp: datetime = field(default_factory=datetime.now)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Relationships
    parent_doc_id: Optional[str] = None  # For chunks
    child_doc_ids: List[str] = field(default_factory=list)  # For documents with chunks
    related_doc_ids: List[str] = field(default_factory=list)  # Semantic relationships
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type.value,
            "location": self.location.to_dict(),
            "content": self.content.to_dict(),
            "conveyance": self.conveyance.to_dict(),
            "embeddings": self.embeddings.to_dict() if self.embeddings else None,
            "metadata": self.metadata,
            "pipeline_stage": self.pipeline_stage,
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "errors": self.errors,
            "parent_doc_id": self.parent_doc_id,
            "child_doc_ids": self.child_doc_ids,
            "related_doc_ids": self.related_doc_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        # Reconstruct Location
        loc_data = data["location"]
        location = Location(
            absolute_path=Path(loc_data["absolute_path"]),
            relative_path=Path(loc_data["relative_path"]) if loc_data.get("relative_path") else None,
            directory_chain=loc_data.get("directory_chain", []),
            file_size=loc_data.get("file_size"),
            last_modified=datetime.fromisoformat(loc_data["last_modified"]) if loc_data.get("last_modified") else None,
            line_start=loc_data.get("line_start"),
            line_end=loc_data.get("line_end"),
            chunk_index=loc_data.get("chunk_index"),
            total_chunks=loc_data.get("total_chunks")
        )
        
        # Reconstruct Content
        cont_data = data["content"]
        content = Content(
            raw_text=cont_data["raw_text"],
            cleaned_text=cont_data.get("cleaned_text"),
            concepts=cont_data.get("concepts", []),
            keywords=cont_data.get("keywords", []),
            entities=cont_data.get("entities", []),
            symbols=cont_data.get("symbols"),
            imports=cont_data.get("imports", []),
            dependencies=cont_data.get("dependencies", []),
            domains=cont_data.get("domains", []),
            topics=cont_data.get("topics", []),
            complexity=ComplexityLevel(cont_data.get("complexity", "unknown")),
            prerequisites=cont_data.get("prerequisites", [])
        )
        
        # Reconstruct Conveyance
        conv_data = data["conveyance"]
        conveyance = Conveyance(
            implementation_fidelity=conv_data.get("implementation_fidelity", 0.0),
            actionability=conv_data.get("actionability", 0.0),
            bridge_potential=conv_data.get("bridge_potential", 0.0),
            completeness=conv_data.get("completeness", 0.0),
            clarity=conv_data.get("clarity", 0.0),
            correctness=conv_data.get("correctness", 0.0),
            test_coverage=conv_data.get("test_coverage"),
            cyclomatic_complexity=conv_data.get("cyclomatic_complexity"),
            maintainability_index=conv_data.get("maintainability_index"),
            amplification_score=conv_data.get("amplification_score", 0.0),
            enhances=conv_data.get("enhances", []),
            enhanced_by=conv_data.get("enhanced_by", [])
        )
        
        # Reconstruct Embeddings if present
        embeddings = None
        if data.get("embeddings"):
            emb_data = data["embeddings"]
            embeddings = Embeddings(
                embedding_model_versions=emb_data.get("embedding_model_versions", {})
            )
            
            # Restore numpy arrays
            if "jina_v4_semantic" in emb_data:
                embeddings.jina_semantic = np.array(emb_data["jina_v4_semantic"])
            
            if "hades_dimensional" in emb_data:
                hades = emb_data["hades_dimensional"]
                if "where_vector" in hades:
                    embeddings.hades_where = np.array(hades["where_vector"])
                if "what_vector" in hades:
                    embeddings.hades_what = np.array(hades["what_vector"])
                if "conveyance_vector" in hades:
                    embeddings.hades_conveyance = np.array(hades["conveyance_vector"])
        
        # Create document
        return cls(
            doc_id=data["doc_id"],
            doc_type=DocumentType(data["doc_type"]),
            location=location,
            content=content,
            conveyance=conveyance,
            embeddings=embeddings,
            metadata=data.get("metadata", {}),
            pipeline_stage=data.get("pipeline_stage", "loaded"),
            processing_timestamp=datetime.fromisoformat(data["processing_timestamp"]),
            errors=data.get("errors", []),
            parent_doc_id=data.get("parent_doc_id"),
            child_doc_ids=data.get("child_doc_ids", []),
            related_doc_ids=data.get("related_doc_ids", [])
        )
    
    def add_error(self, stage: str, error: Exception):
        """Add error to document for tracking."""
        self.errors.append({
            "stage": stage,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
    
    def update_stage(self, stage: str):
        """Update pipeline stage."""
        self.pipeline_stage = stage
        self.processing_timestamp = datetime.now()


@dataclass
class DocumentChunk(Document):
    """
    A chunk of a larger document.
    
    Inherits all properties from Document but represents a portion
    of a larger document for processing efficiency.
    """
    # Reference to parent document
    parent_doc_id: str = ""
    
    # Chunk-specific metadata
    chunk_index: int = 0
    total_chunks: int = 1
    overlap_tokens: int = 0
    
    # Inherited context from parent
    inherited_metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentFactory:
    """
    Factory for creating documents from different sources.
    
    This provides convenient methods for creating documents
    from loader outputs while ensuring consistency.
    """
    
    @staticmethod
    def from_loader_output(
        file_path: Path,
        loader_output: Dict[str, Any],
        doc_type: Optional[DocumentType] = None
    ) -> Document:
        """
        Create document from loader output.
        
        Args:
            file_path: Path to the file
            loader_output: Output from a HERMES loader
            doc_type: Document type (auto-detected if not provided)
            
        Returns:
            Document instance
        """
        # Auto-detect document type if not provided
        if doc_type is None:
            doc_type = DocumentFactory._detect_type(file_path)
        
        # Create location
        location = Location(
            absolute_path=file_path,
            file_size=file_path.stat().st_size if file_path.exists() else None,
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime) if file_path.exists() else None,
            directory_chain=list(file_path.parts[:-1])
        )
        
        # Extract content from loader output
        content = Content(
            raw_text=loader_output.get("content", ""),
            cleaned_text=loader_output.get("cleaned_content"),
            concepts=loader_output.get("metadata", {}).get("concepts", []),
            keywords=loader_output.get("metadata", {}).get("keywords", []),
            symbols=loader_output.get("symbols"),
            imports=loader_output.get("imports", []),
            complexity=ComplexityLevel(loader_output.get("metadata", {}).get("complexity", "unknown"))
        )
        
        # Initialize conveyance (will be populated by analysis)
        conveyance = Conveyance()
        
        # Create document
        doc = Document(
            doc_id=f"doc:{file_path}",
            doc_type=doc_type,
            location=location,
            content=content,
            conveyance=conveyance,
            metadata=loader_output.get("metadata", {})
        )
        
        return doc
    
    @staticmethod
    def _detect_type(file_path: Path) -> DocumentType:
        """Detect document type from file extension."""
        ext_map = {
            ".pdf": DocumentType.PDF,
            ".py": DocumentType.PYTHON,
            ".md": DocumentType.MARKDOWN,
            ".ipynb": DocumentType.JUPYTER,
            ".txt": DocumentType.TEXT,
            ".js": DocumentType.CODE,
            ".ts": DocumentType.CODE,
            ".java": DocumentType.CODE,
            ".cpp": DocumentType.CODE,
            ".c": DocumentType.CODE,
            ".rs": DocumentType.CODE,
            ".go": DocumentType.CODE
        }
        
        return ext_map.get(file_path.suffix.lower(), DocumentType.UNKNOWN)