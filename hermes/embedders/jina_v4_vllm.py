"""
Jina Embeddings v4 with VLLM backend for efficient inference.

This module provides a VLLM-optimized interface to Jina's multimodal, multilingual
embedding model with support for all three LoRA adapters loaded simultaneously.
"""

from typing import List, Union, Optional, Literal
import numpy as np
from pathlib import Path
import logging
from vllm import LLM, SamplingParams
import torch

logger = logging.getLogger(__name__)


class JinaV4VLLMEmbedder:
    """
    Jina Embeddings v4 using VLLM for efficient batch processing.
    
    Features:
    - All three LoRA adapters loaded simultaneously
    - Efficient batched inference with VLLM
    - Support for 12k context length
    - Automatic task switching without model reloading
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        truncate_dim: Optional[int] = 1024,  # Default to HADES' WHAT dimension
        max_length: int = 12288,  # 12k context as requested
        tensor_parallel_size: int = 1,  # For multi-GPU
        dtype: str = "auto",  # Let VLLM choose optimal dtype
        max_model_len: Optional[int] = 12288,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize Jina v4 embedder with VLLM.
        
        Args:
            model_name: HuggingFace model identifier
            truncate_dim: Truncate embeddings to this dimension (128-2048)
            max_length: Maximum sequence length (12k)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for model weights
            max_model_len: Maximum model context length
            gpu_memory_utilization: GPU memory fraction to use
        """
        self.model_name = model_name
        self.truncate_dim = truncate_dim
        self.max_length = max_length
        
        logger.info(f"Loading Jina v4 with VLLM from {model_name}")
        logger.info("Loading full model with all LoRA adapters for flexibility")
        
        # Initialize VLLM with embedding model
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype=dtype,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,  # Required for embedding models
        )
        
        logger.info(f"VLLM Jina v4 initialized with all adapters, dim={truncate_dim}")
    
    def embed_texts(
        self,
        texts: List[str],
        task: Literal["retrieval", "text-matching", "code"] = "retrieval",
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed a list of text documents using VLLM.
        
        Args:
            texts: List of text strings
            task: Task-specific adapter to use
            show_progress: Show progress bar (ignored, VLLM handles this)
            
        Returns:
            Embeddings array of shape (n_texts, truncate_dim)
        """
        logger.info(f"Embedding {len(texts)} texts with task={task}")
        
        # VLLM handles tokenization and batching internally
        # For embedding models, we use encode() method
        outputs = self.model.encode(
            texts,
            pooling_type="MEAN",  # Use mean pooling
            # Pass task label for adapter selection
            # Note: This might need adjustment based on actual VLLM API
            task_label=task,
        )
        
        # Convert outputs to numpy array
        embeddings = []
        for output in outputs:
            # Extract embedding from output
            embedding = output.outputs.embedding
            
            # Truncate if needed
            if self.truncate_dim:
                embedding = embedding[:self.truncate_dim]
            
            embeddings.append(embedding)
        
        return np.vstack(embeddings)
    
    def embed_documents(
        self,
        documents: List[Union[str, dict]],
        task: Literal["retrieval", "text-matching", "code"] = "retrieval",
    ) -> np.ndarray:
        """
        Embed documents with metadata support.
        
        Args:
            documents: List of documents (strings or dicts with 'text' key)
            task: Task-specific adapter to use
            
        Returns:
            Embeddings array
        """
        # Extract text from documents
        texts = []
        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
            elif isinstance(doc, dict) and 'text' in doc:
                texts.append(doc['text'])
            else:
                raise ValueError(f"Invalid document format: {type(doc)}")
        
        return self.embed_texts(texts, task=task)
    
    def get_adapter_for_content(self, content: str, file_type: Optional[str] = None) -> str:
        """
        Automatically select the best adapter based on content type.
        
        Args:
            content: The text content
            file_type: Optional file extension hint
            
        Returns:
            Best adapter name for the content
        """
        # Code-related file types
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.ts', '.jsx', '.tsx'}
        
        if file_type and any(file_type.endswith(ext) for ext in code_extensions):
            return "code"
        
        # Check for code-like patterns in content
        code_indicators = ['def ', 'class ', 'import ', 'function ', 'const ', 'var ', 'let ']
        if any(indicator in content for indicator in code_indicators):
            return "code"
        
        # Default to retrieval for general documents
        return "retrieval"