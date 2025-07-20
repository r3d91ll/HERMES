#!/usr/bin/env python3
"""Simple test to understand Jina v4 loading."""

import sys
sys.path.insert(0, '/home/todd/olympus/HERMES')

from transformers import AutoModel, AutoTokenizer
import torch

# Try loading without any adapter_mask
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4",
    trust_remote_code=True,
)

# Check model info
print(f"Model type: {type(model)}")
print(f"Model config: {model.config}")

# Try to find adapter info
if hasattr(model.config, 'adapter_mask'):
    print(f"Config has adapter_mask: {model.config.adapter_mask}")
    
# Test simple encoding
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v4")
inputs = tokenizer(["test text"], return_tensors="pt", padding=True, truncation=True)

# Try different ways to pass task
print("\nTrying different task approaches:")

# Method 1: Direct call
try:
    outputs = model(**inputs)
    print("✅ Direct call works")
except Exception as e:
    print(f"❌ Direct call failed: {e}")

# Method 2: With task_label
try:
    outputs = model(**inputs, task_label="code")
    print("✅ task_label='code' works")
except Exception as e:
    print(f"❌ task_label failed: {e}")