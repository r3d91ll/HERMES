#!/usr/bin/env python3
"""Test script to verify JinaV4Embedder works independently."""

import sys
sys.path.insert(0, '/home/todd/olympus/HERMES')

# Import directly from the embedder file
from hermes.embedders.jina_v4 import JinaV4Embedder
import torch

# Test instantiation
try:
    embedder = JinaV4Embedder(
        adapter_mask="code",  # Use specific adapter for now
        truncate_dim=1024,
        max_length=12288,  # 12k context
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("✅ JinaV4Embedder instantiated successfully")
    
    # Test embedding generation
    test_text = "This is a test document for embedding generation."
    embeddings = embedder.embed_texts([test_text])  # Task already set by adapter_mask
    print(f"✅ Generated embedding with shape: {len(embeddings[0])}")
    print("✅ HERMES embedder is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)