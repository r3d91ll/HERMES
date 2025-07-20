# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the HERMES repository.

## Project Overview

HERMES (Heterogeneous Extraction and Retrieval for Multi-dimensional Embedding Systems) is the critical data pipeline that prepares documents for the HADES multi-modal agentic RAG system. Without HERMES, HADES cannot function - it transforms raw documents into the multi-dimensional representation required by the information theory framework.

**HERMES's Mission**: Transform chaos into structure by extracting WHERE, WHAT, and CONVEYANCE dimensions from raw documents, enabling HADES to discover theory-practice bridges and validate the multiplicative information model.

## HERMES's Role in the Olympus Ecosystem

HERMES serves as the bridge between raw data and the HADES theoretical framework:

1. **Document Ingestion**: Accepts PDFs, text, markdown, code files
2. **Dimensional Extraction**: Computes WHERE, WHAT, CONVEYANCE vectors
3. **Graph Preparation**: Structures data for ArangoDB storage
4. **ISNE Training**: Prepares document graphs for epistemic topology learning
5. **Quality Assurance**: Ensures all dimensions are non-zero (per theory)

## Theoretical Alignment

HERMES implements the core axioms of Information Reconstructionism:

### Zero Propagation Enforcement
```python
# HERMES ensures no dimension is zero
if where_magnitude == 0 or what_magnitude == 0 or conveyance_magnitude == 0:
    logger.warning(f"Zero dimension detected: WHERE={where_magnitude}, WHAT={what_magnitude}, CONV={conveyance_magnitude}")
    # Document cannot contribute to information flow
```

### Context Amplification Measurement
HERMES measures base conveyance and prepares for context amplification:
- Extracts theory components from academic papers
- Identifies practice components from code/implementations  
- Calculates bridge potential for theory-practice connections

### Observer-Aware Processing
While HERMES operates from the A-Observer perspective (omnipresent), it prepares data for S-Observer queries:
- Encodes filesystem topology in WHERE vectors
- Preserves document relationships for FRAME calculations
- Maintains provenance for observer-dependent views

## Core Components

### 1. Document Loaders (`hermes/loaders/`)
- **PDFLoader**: Extracts text with fallback to OCR
- **TextLoader**: Plain text with encoding detection
- **MarkdownLoader**: Preserves structure and metadata
- **JSONLoader**: Handles structured data formats

### 2. Embedders (`hermes/embedders/`)
- **JinaV4Embedder**: Primary 1024-dim semantic embeddings (WHAT dimension)
- **SentenceTransformerEmbedder**: Local alternative for testing
- All embeddings preserved at full dimensionality

### 3. Extractors (`hermes/extractors/`)
- **VLLMConveyanceAnalyzer**: Uses Qwen3-30B to analyze actionability
- **MetadataExtractor**: Extracts publication dates, authors, venues
- **DSPyMetadataExtractor**: Learning-based metadata extraction

### 4. Gathering (`hermes/gathering/`)
- **ChronologicalGatherer**: Time-based paper collection (1998-2024)
- **PriorityGatherer**: Focused collection (ISNE, DSPy, quantum observer)
- **ArxivGatherer**: Configurable topic-based gathering
- YAML-based configuration for maintainability

### 5. Processing (`hermes/processing/`)
- **process_to_hades.py**: Main pipeline for HADES preparation
- **build_isne_graph.py**: Constructs graph for ISNE training
- **organize_papers_lcc.py**: Library of Congress organization

### 6. Analysis (`hermes/analysis/`)
- **BridgeDetector**: Identifies theory-practice connections
- **CitationNetworkAnalyzer**: Maps knowledge flow
- **SemanticCitationAnalyzer**: Semantic relationship discovery

## Dimensional Extraction Details

### WHERE Dimension (102 dimensions)
```python
# Filesystem topology encoding
where_vector[0] = len(path_parts)  # Depth
where_vector[1:9] = directory_depth_encoding
where_vector[10:22] = permission_structure
where_vector[22:30] = file_attributes
# LCC encoding for semantic organization
where_vector[30:70] = lcc_position_encoding
where_vector[70:102] = graph_proximity_features
```

### WHAT Dimension (1024 dimensions)
- Full Jina v4 embeddings, no dimensionality reduction
- Captures semantic meaning independent of location/format
- Zero only for encrypted/corrupted content

### CONVEYANCE Dimension (922 dimensions)
```python
# Actionability encoding
conveyance_vector[0] = implementation_fidelity
conveyance_vector[1] = actionability_score
conveyance_vector[2] = bridge_potential
conveyance_vector[3:53] = theory_components_one_hot
conveyance_vector[53:103] = practice_components_one_hot
conveyance_vector[103:922] = context_features
```

## Development Commands

```bash
# Setup
poetry install
poetry shell
cp config.example.yaml config.yaml

# Gather papers
./hermes-cli gather chronological --start-year 1998 --papers-per-year 50
./hermes-cli gather topics --topics quantum_observer node_embeddings

# Process documents
python -m hermes.processing.process_to_hades \
    --input-dir ./data/papers \
    --output-dir ./datasets/hades

# Build ISNE graph (requires 2000+ documents)
python -m hermes.processing.build_isne_graph \
    --paper-dirs ./data/papers \
    --min-docs 2000

# Run analysis
python -m hermes.analysis.bridge_detector \
    --dataset ./datasets/hades

# Tests
pytest tests/
black hermes/ tests/
ruff check .
mypy hermes/
```

## Configuration

### Rate Limiting (`hermes/gathering/config/gathering_config.yaml`)
```yaml
rate_limiting:
  download_delay: 5      # Seconds between downloads
  batch_delay: 30       # Seconds after batch
  batch_size: 10        # Papers per batch
```

### Search Topics (`hermes/gathering/config/search_topics.yaml`)
- Easily add new topics without code changes
- Configure categories, queries, and keywords
- Support for cross-domain bridge papers

## Critical Implementation Notes

### 1. Minimum Corpus Size
- ISNE requires minimum 2000 documents for meaningful topology
- Below this threshold, epistemic structure cannot be learned
- Current target: ~1600 from chronological + priority gathering

### 2. Conveyance Analysis
- Uses vLLM with Qwen3-30B-A3B-FP8 for efficiency
- Analyzes actionability, not just semantic content
- Generates training data for DSPy optimization

### 3. Bridge Detection
- Multiplicative model: WHERE × WHAT × CONVEYANCE
- Context amplification: Context^α where α ≈ 1.7
- Validates theory-practice connections

### 4. Graph Construction
- Edges created when combined similarity > 0.6
- Preserves multi-dimensional relationships
- Prepared for both ArangoDB and ISNE training

## Quality Metrics

HERMES ensures data quality for HADES:
- **Completeness**: All documents have non-zero dimensions
- **Consistency**: Standardized vector representations
- **Accuracy**: Validated conveyance analysis
- **Traceability**: Full provenance maintained

## Integration with HADES

HERMES output feeds directly into HADES:
1. Documents stored with multi-dimensional vectors
2. Graph structure enables FRAME calculations
3. ISNE embeddings capture epistemic topology
4. Bridge discoveries validate theoretical framework

## Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch size or gpu_memory_utilization
2. **Rate Limiting**: Respect arXiv delays to avoid bans
3. **Missing Dependencies**: Ensure Poetry environment active
4. **Zero Dimensions**: Check file accessibility and content

### Performance Optimization
- Process documents in batches of 10
- Use checkpoint saves every 100 documents
- Share analyzer instance across documents
- Unload models after processing

## Future Enhancements

1. **Real-time Processing**: Stream documents as downloaded
2. **Distributed Processing**: Multi-GPU support
3. **Enhanced Extractors**: More sophisticated conveyance analysis
4. **Dynamic Dimensions**: Adaptive vector sizing
5. **Temporal Analysis**: WHEN dimension implementation

Remember: HERMES is where theory meets implementation. Every document processed brings us closer to validating the multiplicative information model and discovering new theory-practice bridges.