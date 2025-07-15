"""
Base classes for HERMES components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedding models."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed a list of text documents."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the output embedding dimension."""
        pass


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load a document from path."""
        pass
    
    @abstractmethod
    def can_load(self, path: Union[str, Path]) -> bool:
        """Check if this loader can handle the given path."""
        pass


class BaseExtractor(ABC):
    """Abstract base class for metadata extractors."""
    
    @abstractmethod
    def extract(self, content: str, source_path: Optional[Path] = None) -> Dict[str, Any]:
        """Extract metadata from content."""
        pass


class BaseStorage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def store_document(self, doc_id: str, data: Dict[str, Any]) -> bool:
        """Store a document."""
        pass
    
    @abstractmethod
    def retrieve_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass


class BaseAnalyzer(ABC):
    """Base class for content analyzers."""
    
    @abstractmethod
    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze content and return analysis results."""
        pass