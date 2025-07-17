# H.E.R.M.E.S

**Handling, Extracting, Restructuring, Metadata, Embedding, Storing**

HERMES is a universal data pipeline for preparing graph databases. Named after the Greek messenger god who could travel between all realms, HERMES bridges the gap between raw data and graph-based analysis systems.

The acronym perfectly captures the pipeline's role:

- **H**andling - Multiple document formats and sources
- **E**xtracting - Content and structural information
- **R**estructuring - Normalizing into common formats
- **M**etadata - Rich contextual annotations
- **E**mbedding - Semantic and dimensional representations
- **S**toring - Graph database persistence

## Purpose

Hermes handles the infrastructure concerns of:

1. **Normalizing** diverse data formats into a common structure
2. **Extracting** metadata and structural information
3. **Generating** embeddings and dimensional representations
4. **Loading** data into ArangoDB or other graph databases
5. **Training** graph embedding models (ISNE, Node2Vec, etc.)

This allows downstream analysis systems (like HADES) to focus on their core algorithms rather than data preparation.

## Architecture

```
Raw Data → Normalization → Metadata Extraction → Embedding → Graph Storage → Model Training
```

## Key Features

- **Format Agnostic**: Handles PDFs, markdown, code files, JSON, etc.
- **Metadata Rich**: Extracts structural, semantic, and contextual metadata
- **Embedding Ready**: Integrates with various embedding models (Jina, OpenAI, etc.)
- **Graph Native**: Designed for graph databases from the ground up
- **Reusable**: Configure once, use for multiple projects

## Separation of Concerns

Hermes is intentionally separate from analysis systems like HADES. This separation ensures:

- Clean architectural boundaries
- Reusable infrastructure
- Focused codebases
- Independent development and testing

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your pipeline
cp config.example.yaml config.yaml

# Run the pipeline
python -m hermes.pipeline --config config.yaml
```

## License

[To be determined]
