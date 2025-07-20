"""
Embedding models for semantic representation.
"""

from .jina_v4 import JinaV4Embedder
from .sentence_transformer import SentenceTransformerEmbedder

__all__ = ["JinaV4Embedder", "SentenceTransformerEmbedder"]