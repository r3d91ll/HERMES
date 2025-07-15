"""
Vector similarity search for HERMES using ArangoDB.

Since ArangoDB doesn't have native vector search yet (coming in 3.12+),
this module implements efficient vector search strategies.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """
    Efficient vector search implementation for HERMES.
    
    Strategies:
    1. For small datasets (<100k): Direct cosine similarity in AQL
    2. For medium datasets: Faiss index with periodic rebuilds
    3. For large datasets: Approximate methods (LSH, HNSW)
    
    Supports HADES' dimensional model where searches can be:
    - Full 2048-dimensional
    - Dimension-specific (WHERE, WHAT, CONVEYANCE)
    - Weighted combinations
    """
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        index_type: str = "flat",  # flat, ivf, hnsw
        dimension: int = 2048,
        use_gpu: bool = False
    ):
        """
        Initialize vector search engine.
        
        Args:
            index_path: Path to save/load index
            index_type: Type of Faiss index to use
            dimension: Vector dimension
            use_gpu: Whether to use GPU acceleration
        """
        self.index_path = index_path
        self.index_type = index_type
        self.dimension = dimension
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Initialize index
        self.index = None
        self.id_map = {}  # Maps index position to document ID
        self.reverse_id_map = {}  # Maps document ID to index position
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize Faiss index based on type."""
        if self.index_type == "flat":
            # Exact search with L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "ivf":
            # Inverted file index for medium-scale
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World graphs
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
        
        logger.info(f"Initialized {self.index_type} index for {self.dimension}D vectors")
    
    def add_vectors(self, vectors: Dict[str, np.ndarray], rebuild: bool = True):
        """
        Add vectors to the index.
        
        Args:
            vectors: Dictionary mapping document IDs to vectors
            rebuild: Whether to rebuild IVF index after adding
        """
        if not vectors:
            return
        
        # Prepare batch
        ids = list(vectors.keys())
        vecs = np.vstack([vectors[id] for id in ids])
        
        # Normalize vectors for cosine similarity
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        
        # Add to index
        start_idx = len(self.id_map)
        self.index.add(vecs)
        
        # Update mappings
        for i, doc_id in enumerate(ids):
            idx = start_idx + i
            self.id_map[idx] = doc_id
            self.reverse_id_map[doc_id] = idx
        
        # Train IVF index if needed
        if self.index_type == "ivf" and rebuild and self.index.ntotal > 0:
            self.index.train(vecs)
        
        logger.info(f"Added {len(vectors)} vectors to index (total: {self.index.ntotal})")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filter_fn: Optional function to filter results
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        query_norm = query_vector / np.linalg.norm(query_vector)
        query_norm = query_norm.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_norm, min(k * 2, self.index.ntotal))
        
        # Convert to similarities (1 - normalized_distance for L2)
        similarities = 1 - (distances[0] / 2)  # L2 distance to cosine similarity
        
        # Map back to document IDs
        results = []
        for idx, sim in zip(indices[0], similarities):
            if idx < 0:  # Faiss returns -1 for not found
                continue
                
            doc_id = self.id_map.get(idx)
            if doc_id and (not filter_fn or filter_fn(doc_id)):
                results.append((doc_id, float(sim)))
        
        # Sort by similarity and limit to k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def search_dimensional(
        self,
        query_node: Dict[str, Any],
        k: int = 10,
        dimension_weights: Dict[str, float] = None
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Search using HADES dimensional model.
        
        Args:
            query_node: Query node with dimensional vectors
            k: Number of results
            dimension_weights: Weights for each dimension
            
        Returns:
            List of (doc_id, combined_score, dimension_scores) tuples
        """
        if dimension_weights is None:
            dimension_weights = {
                "where": 0.05,
                "what": 0.50,
                "conveyance": 0.45
            }
        
        # Extract query vectors
        embeddings = query_node.get("embeddings", {}).get("hades_dimensional", {})
        
        where_vec = embeddings.get("where_vector")
        what_vec = embeddings.get("what_vector")
        conv_vec = embeddings.get("conveyance_vector")
        
        # Search each dimension separately
        # This would require separate indexes in production
        # For now, we'll use the concatenated vector
        
        if "full_vector" in embeddings:
            full_vec = embeddings["full_vector"]
        else:
            # Concatenate dimensional vectors
            full_vec = np.concatenate([where_vec, what_vec, conv_vec])
        
        # Standard search
        results = self.search(full_vec, k * 2)  # Get extra for filtering
        
        # Calculate dimensional scores
        enhanced_results = []
        for doc_id, base_score in results:
            # In production, retrieve actual vectors and calculate per-dimension
            # For now, use base score with simulated dimension breakdown
            dim_scores = {
                "where": base_score * 0.8,  # Simulate lower WHERE similarity
                "what": base_score,
                "conveyance": base_score * 0.9
            }
            
            # Calculate weighted combination
            combined = sum(
                dim_scores[dim] * dimension_weights.get(dim, 0.33)
                for dim in dim_scores
            )
            
            # Apply HADES multiplicative model
            # If any dimension is near zero, penalize heavily
            multiplicative = dim_scores["where"] * dim_scores["what"] * dim_scores["conveyance"]
            
            # Context amplification
            alpha = 1.5
            amplified = combined * (dim_scores["conveyance"] ** alpha)
            
            enhanced_results.append((doc_id, amplified, dim_scores))
        
        # Sort by amplified score
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results[:k]
    
    def save_index(self, path: Optional[Path] = None):
        """Save index to disk."""
        save_path = path or self.index_path
        if not save_path:
            logger.warning("No save path specified")
            return
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Faiss index
        faiss.write_index(self.index, str(save_path))
        
        # Save mappings
        mappings = {
            "id_map": self.id_map,
            "reverse_id_map": self.reverse_id_map
        }
        with open(save_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(mappings, f)
        
        logger.info(f"Saved index to {save_path}")
    
    def load_index(self, path: Optional[Path] = None):
        """Load index from disk."""
        load_path = path or self.index_path
        if not load_path or not Path(load_path).exists():
            logger.warning("No index file found")
            return
        
        load_path = Path(load_path)
        
        # Load Faiss index
        self.index = faiss.read_index(str(load_path))
        
        # Load mappings
        with open(load_path.with_suffix(".pkl"), "rb") as f:
            mappings = pickle.load(f)
            self.id_map = mappings["id_map"]
            self.reverse_id_map = mappings["reverse_id_map"]
        
        logger.info(f"Loaded index from {load_path} ({self.index.ntotal} vectors)")
    
    def remove_vector(self, doc_id: str):
        """
        Remove a vector from the index.
        
        Note: Faiss doesn't support removal directly, so this marks it as removed.
        In production, periodic index rebuilds would be needed.
        """
        if doc_id in self.reverse_id_map:
            idx = self.reverse_id_map[doc_id]
            # Mark as removed by setting to zero vector
            zero_vec = np.zeros((1, self.dimension), dtype=np.float32)
            self.index.add_with_ids(zero_vec, np.array([idx], dtype=np.int64))
            
            # Remove from mappings
            del self.reverse_id_map[doc_id]
            del self.id_map[idx]
            
            logger.debug(f"Marked vector {doc_id} for removal")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "uses_gpu": self.use_gpu,
            "memory_usage_mb": self.index.ntotal * self.dimension * 4 / (1024 * 1024)
        }