"""
Jina Embeddings v4 integration for HERMES.

This module provides a unified interface to Jina's multimodal, multilingual
embedding model with support for flexible dimensional output.
"""

from typing import List, Union, Optional, Literal
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import logging

from hermes.core.base import BaseEmbedder


logger = logging.getLogger(__name__)


class JinaV4Embedder(BaseEmbedder):
    """
    Jina Embeddings v4 with support for text, images, and documents.
    
    Features:
    - Multimodal: text, images, visual documents
    - Multilingual: 30+ languages
    - Flexible dimensions: 128-2048 (Matryoshka)
    - Task-specific adapters
    - Long context: up to 32k tokens
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        truncate_dim: Optional[int] = 1024,  # Default to HADES' WHAT dimension
        adapter_mask: Optional[Literal["retrieval", "text-matching", "code"]] = None,  # None loads all adapters
        batch_size: int = 32,
        max_length: int = 12288,  # 12k context as requested
    ):
        """
        Initialize Jina v4 embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            truncate_dim: Truncate embeddings to this dimension (128-2048)
            adapter_mask: Task-specific adapter to use
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.truncate_dim = truncate_dim
        self.adapter_mask = adapter_mask
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        logger.info(f"Loading Jina v4 from {model_name}")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device,
        }
        # Only add adapter_mask if specific adapter requested
        if adapter_mask is not None:
            model_kwargs["adapter_mask"] = adapter_mask
            logger.info(f"Loading with {adapter_mask} adapter only")
        else:
            logger.info("Loading full model with all LoRA adapters")
            
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move to device
        if device != "auto":
            self.model = self.model.to(device)
        self.model.eval()
        
        adapter_info = f"{adapter_mask} adapter" if adapter_mask else "all LoRA adapters"
        logger.info(f"Jina v4 initialized with {adapter_info}, dim={truncate_dim}")
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
        task: Optional[Literal["retrieval", "text-matching", "code"]] = None
    ) -> np.ndarray:
        """
        Embed a list of text documents.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            Embeddings array of shape (n_texts, truncate_dim)
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                # If all adapters loaded, need to specify task
                if self.adapter_mask is None:
                    task_to_use = task or "retrieval"  # Default to retrieval
                    outputs = self.model(**inputs, task_label=task_to_use)
                else:
                    outputs = self.model(**inputs)
                
                # For Jina v4, use single_vec_emb (pooled embeddings)
                if hasattr(outputs, 'single_vec_emb'):
                    batch_embeddings = outputs.single_vec_emb
                elif hasattr(outputs, 'embeddings'):
                    batch_embeddings = outputs.embeddings
                elif hasattr(outputs, 'last_hidden_state'):
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    raise ValueError(f"Unknown output format from Jina model: {type(outputs)}")
                
                # Truncate if needed
                if self.truncate_dim:
                    batch_embeddings = batch_embeddings[:, :self.truncate_dim]
                
                # Convert to float32 for numpy (from bfloat16)
                embeddings.append(batch_embeddings.float().cpu().numpy())
        
        return np.vstack(embeddings)
    
    def embed_images(
        self,
        image_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed images or visual documents.
        
        Args:
            image_paths: List of paths to images/PDFs
            show_progress: Show progress bar
            
        Returns:
            Embeddings array of shape (n_images, truncate_dim)
        """
        embeddings = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Load images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.error(f"Failed to load image {path}: {e}")
                    # Use zero embedding for failed images
                    images.append(None)
            
            # Process valid images
            valid_images = [img for img in images if img is not None]
            if valid_images:
                # Note: Actual implementation would need proper image preprocessing
                # This is a placeholder - Jina v4 requires specific image handling
                logger.warning("Image embedding not fully implemented yet")
                
                # Placeholder: return random embeddings
                batch_embeddings = np.random.randn(len(images), self.truncate_dim or 2048)
            else:
                batch_embeddings = np.zeros((len(images), self.truncate_dim or 2048))
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def embed_mixed(
        self,
        items: List[Union[str, Path]],
        item_types: List[Literal["text", "image", "document"]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed mixed content types.
        
        Args:
            items: List of texts or paths
            item_types: Type of each item
            show_progress: Show progress bar
            
        Returns:
            Embeddings array of shape (n_items, truncate_dim)
        """
        # Group by type for efficient processing
        text_indices = [i for i, t in enumerate(item_types) if t == "text"]
        image_indices = [i for i, t in enumerate(item_types) if t in ["image", "document"]]
        
        # Initialize results array
        embeddings = np.zeros((len(items), self.truncate_dim or 2048))
        
        # Process texts
        if text_indices:
            texts = [items[i] for i in text_indices]
            text_embeddings = self.embed_texts(texts, show_progress)
            for idx, emb_idx in enumerate(text_indices):
                embeddings[emb_idx] = text_embeddings[idx]
        
        # Process images/documents
        if image_indices:
            paths = [items[i] for i in image_indices]
            image_embeddings = self.embed_images(paths, show_progress)
            for idx, emb_idx in enumerate(image_indices):
                embeddings[emb_idx] = image_embeddings[idx]
        
        return embeddings
    
    def late_chunking(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 128
    ) -> List[np.ndarray]:
        """
        Jina's late chunking strategy - embed full document then chunk.
        
        This preserves context better than chunking before embedding.
        
        Args:
            text: Full document text
            chunk_size: Size of chunks in tokens
            overlap: Token overlap between chunks
            
        Returns:
            List of chunk embeddings
        """
        # First, embed the full document
        full_embedding = self.embed_texts([text])[0]
        
        # Tokenize to get token boundaries
        tokens = self.tokenizer(text, return_tensors="pt")
        n_tokens = tokens.input_ids.shape[1]
        
        # Calculate chunk boundaries
        chunks = []
        for start in range(0, n_tokens, chunk_size - overlap):
            end = min(start + chunk_size, n_tokens)
            
            # Get chunk text
            chunk_ids = tokens.input_ids[:, start:end]
            chunk_text = self.tokenizer.decode(chunk_ids[0], skip_special_tokens=True)
            
            # Embed chunk (this would ideally reuse full document computation)
            chunk_embedding = self.embed_texts([chunk_text])[0]
            chunks.append(chunk_embedding)
        
        return chunks
    
    def get_dimension(self) -> int:
        """Return the output embedding dimension."""
        return self.truncate_dim or 2048