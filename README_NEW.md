# HERMES - Universal Data Pipeline for HADES

HERMES (Heterogeneous Extraction and Retrieval for Multi-dimensional Embedding Systems) is a universal data pipeline designed to process documents for the HADES multi-modal agentic RAG system.

## Project Structure

```
HERMES/
├── hermes/              # Main package
│   ├── core/           # Core functionality
│   ├── loaders/        # Document loaders
│   ├── embedders/      # Embedding generators
│   ├── extractors/     # Metadata and conveyance extractors
│   ├── storage/        # Database interfaces
│   ├── gathering/      # Paper gathering modules
│   ├── analysis/       # Analysis tools (bridge detection, etc.)
│   ├── processing/     # Document processing pipelines
│   ├── cli/           # Command-line interfaces
│   └── utils/         # Utility functions
├── docs/              # Documentation
│   ├── strategies/    # Research strategies
│   └── evidence/      # Evidence and findings
├── scripts/           # Utility scripts
├── tests/            # Test suite
├── data/             # Downloaded papers (git-ignored)
└── datasets/         # Processed datasets (git-ignored)
```

## Installation

```bash
# Install dependencies
poetry install

# Activate environment
poetry shell
```

## Quick Start

### 1. Gather Papers

```bash
# Gather papers chronologically (1998-2024)
./hermes-cli gather chronological --start-year 1998 --papers-per-year 50

# Gather papers by topic
./hermes-cli gather topics --topics quantum_observer node_embeddings dspy_optimization

# List available topics
./hermes-cli gather list-topics
```

### 2. Process Documents

```bash
# Process papers through HERMES pipeline
./hermes-cli process --input-dir ./data/papers --output-dir ./datasets/hades

# Build ISNE graph (requires 2000+ documents)
./hermes-cli build --paper-dirs ./data/papers --min-docs 2000
```

### 3. Analyze Results

```bash
# Run bridge detection
python -m hermes.analysis.bridge_detector --dataset ./datasets/hades

# Analyze collection
./hermes-cli gather topics --analyze-only --output-dir ./data/papers
```

## Configuration

Edit configuration files in `hermes/gathering/config/`:
- `gathering_config.yaml` - Main configuration
- `search_topics.yaml` - Search topics and keywords

## Key Features

- **Multi-dimensional Analysis**: WHERE × WHAT × CONVEYANCE × TIME
- **Configurable Gathering**: YAML-based topic configuration
- **Bridge Detection**: Find theory-practice connections
- **ISNE Training**: Learn epistemic topology from document graphs
- **Respectful Rate Limiting**: Polite to arXiv and other sources
- **Checkpoint Support**: Resume interrupted processing

## Requirements

- Python 3.11+
- CUDA GPU (for Qwen conveyance analysis)
- 32GB+ RAM recommended
- Optional: ArangoDB for graph storage

## License

MIT License - See LICENSE file for details.
