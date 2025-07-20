"""
Sentence Transformer embedder for HERMES.
Local alternative to Jina for testing and development.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

from hermes.core.base import BaseEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local sentence transformer embedder for cost-effective testing.
    
    Good for development and testing without API costs.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        truncate_dim: Optional[int] = None,
    ):
        """
        Initialize sentence transformer embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on (cpu/cuda)
            batch_size: Batch size for encoding
            show_progress_bar: Show progress during encoding
            truncate_dim: Optionally truncate embeddings to this dimension
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.truncate_dim = truncate_dim
        
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        if truncate_dim and truncate_dim > self.dimension:
            logger.warning(f"Truncate dimension {truncate_dim} larger than model dimension {self.dimension}")
            self.truncate_dim = None
            
        logger.info(f"Initialized {model_name} with dimension {self.dimension}")
        
    def embed_texts(
        self,
        texts: List[str],
        show_progress: Optional[bool] = None
    ) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Override default progress bar setting
            
        Returns:
            Embeddings array of shape (n_texts, dimension)
        """
        if show_progress is None:
            show_progress = self.show_progress_bar
            
        # Encode texts
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Truncate if requested
        if self.truncate_dim and self.truncate_dim < embeddings.shape[1]:
            embeddings = embeddings[:, :self.truncate_dim]
            
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed_texts([text], show_progress=False)[0]
    
    def get_dimension(self) -> int:
        """Return the output embedding dimension."""
        return self.truncate_dim if self.truncate_dim else self.dimension
    
    def encode_batch_with_metadata(
        self,
        texts: List[str],
        metadata: List[dict]
    ) -> List[dict]:
        """
        Encode texts with associated metadata.
        
        Returns list of dicts with 'embedding' and 'metadata' keys.
        """
        embeddings = self.embed_texts(texts)
        
        results = []
        for embedding, meta in zip(embeddings, metadata):
            results.append({
                'embedding': embedding,
                'metadata': meta
            })
            
        return results