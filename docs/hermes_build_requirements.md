# HERMES Build Requirements - Data Pipeline Infrastructure

## Overview

HERMES (Handling, Extracting, Restructuring, Metadata, Embedding, Storing) is the universal data pipeline that prepares documents for graph database storage. It operates independently from HADES, focusing solely on document processing and graph construction.

**Core Purpose**: Transform diverse document formats into graph-ready data with dimensional metadata (WHERE, WHAT, CONVEYANCE).

## System Architecture

### HERMES Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   HERMES Pipeline Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Documents ──► Loaders ──► Extractors ──► Embedders ──► Storage │
│      │            │            │                │             │           │
│      ▼           ▼            ▼                ▼            ▼      │
│  ┌─────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ PDFs      │ │DocLing  │ │Metadata    │ │Jina v4     │ │ArangoDB  │ │
│  │ Code      │ │PyPDF2   │ │DSPy        │ │Sentence    │ │Graph     │ │
│  │ Markdown  │ │AST      │ │Analysis    │ │Transform   │ │Storage   │ │
│  │ JSON      │ │Parser   │ │            │ │ & ISNE     │ │          │ │
│  └─────────┘ └────────┘ └──────────┘ └──────────┘ └─────────┘ │
│                                                                  │
│              ┌─────────────────────────────┐                    │
│              │   ISNE Initial Training          │                    │
│              │  (Learn graph structure)         │                    │
│              └─────────────────────────────┘                    │
│                                                                  │
│              ┌─────────────────────────────┐                    │
│              │    DSPy Optimization             │                    │
│              │  (Learn from Claude/LLM)         │                    │
│              └─────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘

## Theoretical Foundation

### Dimensional Model Implementation

```python
# HERMES implements dimensional storage, not computation
dimensions = {
    'WHERE': 'Filesystem location, graph position, structural context',
    'WHAT': 'Semantic content, embeddings, topic metadata',
    'CONVEYANCE': 'Implementation fidelity, actionability scores',
    'TIME': 'Creation, modification, version history'
}

# HERMES stores potential, not information
# Information emerges when System-Observer (query) interacts with data
```

## Core Components

### 1. Document Loaders

```python
class HERMESLoaderInterface:
    """Base interface for all document loaders"""
    
    supported_loaders = {
        '.pdf': 'DoclingLoader',      # Advanced PDF extraction
        '.py': 'PythonASTLoader',     # Code structure analysis
        '.md': 'MarkdownLoader',      # Structured text
        '.txt': 'TextLoader',         # Plain text
        '.json': 'JSONLoader',        # Structured data
        '.ipynb': 'NotebookLoader'    # Jupyter notebooks
    }
    
    def extract_content(self, file_path: Path) -> DocumentContent:
        """Extract raw content and structure"""
        pass
        
    def extract_metadata(self, file_path: Path) -> LocationMetadata:
        """Extract WHERE dimension metadata"""
        return {
            'absolute_path': str(file_path),
            'directory_chain': file_path.parts,
            'file_type': file_path.suffix,
            'size_bytes': file_path.stat().st_size,
            'permissions': oct(file_path.stat().st_mode),
            'created': file_path.stat().st_ctime,
            'modified': file_path.stat().st_mtime
        }
```

### 2. Metadata Extraction (DSPy Enhanced)

```python
class DSPyMetadataExtractor:
    """DSPy-optimized metadata extraction"""
    
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()  # DSPy module
        self.conveyance_analyzer = ConveyanceAnalyzer()  # DSPy module
        
    def extract_semantic_metadata(self, content: str) -> SemanticMetadata:
        """Extract WHAT dimension metadata using DSPy"""
        
        # DSPy learns optimal extraction from examples
        metadata = self.metadata_extractor(
            document_content=content,
            document_type=self.doc_type
        )
        
        return {
            'domain': metadata.domain,
            'complexity': metadata.complexity,
            'prerequisites': metadata.prerequisites,
            'concepts': metadata.concepts,
            'language': metadata.language
        }
    
    def assess_conveyance(self, content: str, metadata: dict) -> ConveyanceMetrics:
        """Assess CONVEYANCE dimension using DSPy"""
        
        scores = self.conveyance_analyzer(
            content=content,
            metadata=metadata
        )
        
        return {
            'implementation_fidelity': scores.implementation_fidelity,
            'actionability': scores.actionability,
            'bridge_potential': scores.bridge_potential,
            'has_algorithm': scores.has_algorithm,
            'has_equations': scores.has_equations,
            'has_examples': scores.has_examples
        }
```

### 3. Embedding Generation

```python
class HERMESEmbedder:
    """Multi-model embedding generation"""
    
    def __init__(self):
        self.embedders = {
            'jina_v4': JinaV4Embedder(),  # 1024-dim semantic
            'code_bert': CodeBERTEmbedder(),  # Code-specific
            'sentence_transformers': SentenceTransformer()  # Fallback
        }
        
    def generate_embeddings(self, content: str, doc_type: str) -> Embeddings:
        """Generate multi-modal embeddings"""
        
        embeddings = {}
        
        # Primary semantic embedding
        embeddings['semantic'] = self.embedders['jina_v4'].encode(
            content, 
            task="retrieval.passage"
        )
        
        # Code-specific if applicable
        if doc_type in ['.py', '.js', '.java']:
            embeddings['code'] = self.embedders['code_bert'].encode(content)
            
        return embeddings
```

### 4. ISNE Initial Training

```python
class ISNETrainer:
    """
    ISNE in HERMES learns the initial graph structure and embeddings.
    This creates the base topology that HADES will later make dynamic.
    """
    
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.encoder = InductiveShallowEncoder()
        
    def train_on_graph(self, nodes, edges):
        """
        Train ISNE on the initial graph structure.
        Unlike Node2Vec, ISNE can handle new nodes later.
        """
        # Initialize node positions
        for node in nodes:
            # ISNE's key: encode based on neighborhood, not lookup table
            neighborhood = self._get_neighborhood_structure(node, edges)
            initial_position = self.encoder.encode(neighborhood)
            node.isne_embedding = initial_position
            
        # Iterative refinement
        for iteration in range(self.training_iterations):
            # Update positions based on graph structure
            for node in nodes:
                # Capture local structure
                local_structure = self._capture_local_structure(node, edges)
                
                # Update embedding based on neighbors
                new_embedding = self.encoder.refine(
                    current=node.isne_embedding,
                    neighborhood=local_structure
                )
                
                node.isne_embedding = new_embedding
                
        return nodes
        
    def encode_new_node(self, new_node, existing_graph):
        """
        ISNE's advantage: can encode nodes not seen during training.
        This is crucial for dynamic document collections.
        """
        # Get neighborhood in existing graph
        neighbors = existing_graph.find_neighbors(new_node)
        neighborhood_structure = self._get_neighborhood_structure(new_node, neighbors)
        
        # Inductively generate embedding
        new_node.isne_embedding = self.encoder.encode(neighborhood_structure)
        
        return new_node
```

### 5. Graph Storage

```python
class ArangoGraphStorage:
    """Store documents as graph with dimensional edges"""
    
    def store_document(self, doc: Document) -> str:
        """Store document node and create edges"""
        
        # Create document node
        node_data = {
            '_key': doc.doc_id,
            'content': doc.content.cleaned_text,
            'embeddings': doc.embeddings.to_dict(),
            'metadata': {
                'location': doc.location.to_dict(),
                'semantic': doc.semantic.to_dict(),
                'conveyance': doc.conveyance.to_dict(),
                'temporal': doc.temporal.to_dict()
            }
        }
        
        self.nodes.insert(node_data)
        
        # Create dimensional edges
        self._create_where_edges(doc)  # Filesystem structure
        self._create_what_edges(doc)   # Semantic similarity
        self._create_conveyance_edges(doc)  # Implementation bridges
        
        return doc.doc_id
```

## Pipeline Configuration

### HERMES Configuration Schema

```yaml
hermes_config:
  # Pipeline settings
  pipeline:
    batch_size: 100
    max_workers: 8
    chunk_size: 1000
    chunk_overlap: 200
    
  # Loader settings
  loaders:
    pdf:
      use_docling: true
      extract_images: false
      ocr_enabled: false
    code:
      extract_ast: true
      include_comments: true
      
  # Embedding settings
  embeddings:
    models:
      - jina_v4
      - sentence_transformers
    batch_size: 32
    max_sequence_length: 8192
    
  # Storage settings
  storage:
    backend: arangodb
    connection:
      host: localhost
      port: 8529
      database: hermes
    graph:
      name: document_graph
      edge_collections:
        - edges_where
        - edges_what
        - edges_conveyance
        
  # DSPy optimization
  dspy:
    enabled: true
    training_mode: false
    model: "gpt-4"  # For optimization only
    examples_path: "./training_examples"
```

## Implementation Requirements

### Phase 1: Basic Pipeline (Week 1)

```python
# Minimal working pipeline
requirements = {
    'document_loading': {
        'pdf_support': 'PyPDF2 or pdfplumber',
        'code_parsing': 'AST module',
        'text_extraction': 'Basic regex cleaning'
    },
    'metadata_extraction': {
        'filesystem': 'os.path and pathlib',
        'basic_semantic': 'Keywords and patterns',
        'manual_conveyance': 'Rule-based scoring'
    },
    'embeddings': {
        'model': 'sentence-transformers (free)',
        'dimension': 384,
        'batch_processing': 'Simple loop'
    },
    'storage': {
        'database': 'ArangoDB Docker',
        'schema': 'Basic node/edge structure',
        'indexing': 'HNSW for vectors'
    }
}
```

### Phase 2: DSPy Integration (Week 2)

```python
# Add learning capabilities
dspy_integration = {
    'metadata_learning': {
        'training_data': 'Claude-annotated examples',
        'optimization': 'BootstrapFewShot',
        'metrics': 'Metadata richness score'
    },
    'conveyance_learning': {
        'training_pairs': 'Theory-practice bridges',
        'learned_formula': 'Context amplification α',
        'validation': 'Known good examples'
    },
    'pipeline_optimization': {
        'modules': ['MetadataExtractor', 'ConveyanceAnalyzer'],
        'budget': 'medium',
        'iterations': 100
    }
}
```

### Phase 3: Production Features (Week 3)

```python
# Production-ready features
production_features = {
    'scalability': {
        'async_processing': 'asyncio pipeline',
        'parallel_extraction': 'multiprocessing',
        'batch_embeddings': 'GPU optimization'
    },
    'reliability': {
        'error_handling': 'Graceful degradation',
        'retry_logic': 'Exponential backoff',
        'monitoring': 'Prometheus metrics'
    },
    'flexibility': {
        'plugin_loaders': 'Dynamic loading',
        'custom_extractors': 'User-defined',
        'embedding_models': 'Configurable'
    }
}
```

## Testing Requirements

### Unit Tests

```python
test_requirements = {
    'loaders': {
        'pdf_extraction': 'Test with various PDF types',
        'code_parsing': 'Multiple languages',
        'error_handling': 'Corrupted files'
    },
    'extractors': {
        'metadata_quality': 'Validate completeness',
        'conveyance_scoring': 'Known examples',
        'dspy_learning': 'Training convergence'
    },
    'storage': {
        'graph_integrity': 'Node-edge consistency',
        'query_performance': 'Index efficiency',
        'concurrent_access': 'Thread safety'
    }
}
```

### Integration Tests

```python
integration_tests = {
    'pipeline_flow': 'End-to-end document processing',
    'error_propagation': 'Failure handling',
    'performance': 'Throughput benchmarks',
    'memory_usage': 'Resource consumption'
}
```

## Performance Specifications

```yaml
performance_targets:
  # Document processing
  throughput:
    pdf_loading: 10 docs/minute
    code_parsing: 50 files/minute
    embedding_generation: 30 docs/minute
    
  # Latency targets
  latency_p95:
    single_document: < 5 seconds
    batch_100_docs: < 3 minutes
    
  # Resource usage
  resources:
    cpu_cores: 4-8
    ram_usage: < 16GB
    gpu_memory: < 8GB (if available)
    
  # Storage efficiency
  storage:
    compression: 'Document text compressed'
    indexing: 'HNSW with m=16'
    query_time: < 100ms for 1M docs
```

## Deployment Options

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install HERMES
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install

# Copy source
COPY hermes/ ./hermes/
COPY configs/ ./configs/

# Run pipeline
CMD ["python", "-m", "hermes.pipeline", "--config", "configs/default.yaml"]
```

### Standalone Script

```python
# Simple HERMES invocation
from hermes import process_directory

stats = process_directory(
    directory_path="/path/to/documents",
    config_path="hermes_config.yaml",
    progress_callback=lambda msg, current, total: print(f"{msg}: {current}/{total}")
)

print(f"Processed {stats['documents_processed']} documents")
print(f"Created {stats['edges_created']} relationships")
```

## ISNE Role in HERMES vs HADES

### HERMES: Static Graph Learning

- ISNE trains on initial document collection
- Learns base graph topology and embeddings
- Handles new documents as they're added (inductive)
- Creates stable foundation for graph structure

### HADES: Dynamic Graph Response

- ISNE enables nodes to shift with System-Observer presence
- Positions update based on query injection
- Creates gradients through local distortions
- Enables information emergence through movement

## Critical Design Principles

1. **Separation of Concerns**: HERMES only processes and stores. No query handling or emergence computation.

2. **Dimensional Storage**: Store WHERE, WHAT, CONVEYANCE as metadata and edges. Don't compute information.

3. **Learning Pipeline**: Use DSPy to learn from examples, not hand-craft rules.

4. **Modular Design**: Each component (loader, extractor, embedder) is independently replaceable.

5. **Graph-First**: Everything is a node or edge. No special cases.

6. **ISNE Integration**: ISNE provides inductive capability for both initial training (HERMES) and dynamic response (HADES).

This document defines HERMES as a pure data pipeline, separate from HADES's emergence engine. HERMES prepares the graph with ISNE embeddings; HADES uses ISNE to enable dynamic information emergence through System-Observer interaction.
