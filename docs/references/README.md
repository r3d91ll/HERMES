# HERMES Reference Documentation

## Core Technologies

### Jina Embeddings v4

**Paper**: [Jina Embeddings v4: A Multimodal Multilingual Embedding Model (2024)](./jina-embeddings-v4-paper.pdf)

**Why Jina v4 for HERMES:**

1. **Multimodal Support**: Can embed text, images, and visual documents (PDFs with charts/tables)
2. **Matryoshka Embeddings**: Flexible dimensions from 128 to 2048
   - Perfect for HADES' dimensional allocation strategy
   - Can adjust embedding size based on use case
3. **Task-Specific Adapters**: 
   - Retrieval adapter for document search
   - Code understanding for technical documents
   - Text matching for similarity
4. **Long Context**: 32,768 token support for full documents
5. **Multilingual**: 30+ languages for diverse datasets

**Key Features for HERMES Pipeline:**
- **Unified Embeddings**: Single model for all content types
- **Late Chunking**: Process full documents then chunk (preserves context)
- **Dense + Sparse**: Supports both retrieval modes
- **3.75B Parameters**: Rich semantic understanding

**Integration Strategy:**
```python
# Basic usage in HERMES
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4",
    trust_remote_code=True,
    adapter_mask="retrieval"  # or "text-matching", "code"
)

# Generate embeddings with flexible dimensions
embeddings = model.encode(
    documents,
    truncate_dim=1024  # Match HADES' WHAT dimension
)
```

**Dimensional Flexibility for HADES:**
- Can generate 2048-dim embeddings
- Truncate to 1024 for HADES' WHAT dimension
- Or use full 2048 and let HADES extract dimensional components

This aligns perfectly with HERMES' role in **H**andling diverse formats, **E**xtracting meaning, and **E**mbedding content for downstream analysis.