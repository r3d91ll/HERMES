#!/usr/bin/env python3
"""Debug script to understand Jina v4 output format."""

import sys
sys.path.insert(0, '/home/todd/olympus/HERMES')

from transformers import AutoModel, AutoTokenizer
import torch

# Load model
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v4")

# Test text
text = ["This is a test document"]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs, task_label="retrieval")
    print(f"Output type: {type(outputs)}")
    print(f"Output attributes: {dir(outputs)}")
    
    # Try to find embeddings
    if hasattr(outputs, 'embeddings'):
        print(f"Has embeddings attribute, shape: {outputs.embeddings.shape}")
    
    # Check if it's a tuple/list
    if isinstance(outputs, (tuple, list)):
        print(f"Output is a {type(outputs).__name__} with {len(outputs)} elements")
        for i, item in enumerate(outputs):
            print(f"  Element {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
    
    # Try accessing as array
    try:
        if hasattr(outputs, 'logits'):
            print(f"Has logits, shape: {outputs.logits.shape}")
        if hasattr(outputs, 'pooler_output'):
            print(f"Has pooler_output, shape: {outputs.pooler_output.shape}")
    except:
        pass