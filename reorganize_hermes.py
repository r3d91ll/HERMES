#!/usr/bin/env python3
"""
Reorganize HERMES files into proper module structure.
This script moves standalone files into appropriate subdirectories.
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directories():
    """Create the new directory structure."""
    dirs = [
        "hermes/gathering",
        "hermes/gathering/config", 
        "hermes/analysis",
        "hermes/processing",
        "hermes/cli",
        "hermes/utils",
        "docs/strategies",
        "docs/evidence",
        "scripts",
        "tests"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def move_files():
    """Move files to their appropriate locations."""
    
    # File move mappings
    moves = {
        # Gathering scripts -> hermes/gathering/
        "gather_arxiv_papers.py": "scripts/legacy/gather_arxiv_papers.py",
        "gather_priority_papers.py": "scripts/legacy/gather_priority_papers.py",
        "gather_quantum_observer_papers.py": "scripts/legacy/gather_quantum_observer_papers.py",
        "unified_source_finder.py": "scripts/legacy/unified_source_finder.py",
        "find_sources.py": "scripts/legacy/find_sources.py",
        "find_ml_papers_arxiv.py": "scripts/legacy/find_ml_papers_arxiv.py",
        "find_ml_papers_with_impact.py": "scripts/legacy/find_ml_papers_with_impact.py",
        "find_anthropology_semantics.py": "scripts/legacy/find_anthropology_semantics.py",
        
        # Analysis scripts -> hermes/analysis/
        "citation_network_analyzer.py": "hermes/analysis/citation_network_analyzer.py",
        "semantic_citation_analyzer.py": "hermes/analysis/semantic_citation_analyzer.py",
        
        # Processing scripts -> hermes/processing/
        "process_corpus.py": "hermes/processing/process_corpus.py",
        "process_to_hades.py": "hermes/processing/process_to_hades.py",
        "build_isne_graph.py": "hermes/processing/build_isne_graph.py",
        "organize_papers_lcc.py": "hermes/processing/organize_papers_lcc.py",
        
        # Documentation -> docs/
        "anthropology_source_strategy.md": "docs/strategies/anthropology_source_strategy.md",
        "strategic_paper_selection.md": "docs/strategies/strategic_paper_selection.md",
        "conveyance_lag_evidence.md": "docs/evidence/conveyance_lag_evidence.md",
        "hermes_build_requirements.md": "docs/hermes_build_requirements.md",
        
        # Test scripts -> tests/
        "test_conveyance.py": "tests/test_conveyance.py",
        "test_ocr_detection.py": "tests/test_ocr_detection.py",
        "test_ocr_pdf.py": "tests/test_ocr_pdf.py",
        "test_single_ocr.py": "tests/test_single_ocr.py",
        "test_vllm_analyzer.py": "tests/test_vllm_analyzer.py",
        "test_vllm_debug.py": "tests/legacy/test_vllm_debug.py",
        "test_vllm_direct.py": "tests/legacy/test_vllm_direct.py",
        "test_vllm_final.py": "tests/test_vllm_final.py",
        "test_vllm_simple.py": "tests/legacy/test_vllm_simple.py",
        
        # Scripts -> scripts/
        "download_qwen.py": "scripts/download_qwen.py",
        "configure_arangodb_network.sh": "scripts/configure_arangodb_network.sh",
        "gather_chronological.sh": "scripts/gather_chronological.sh",
        
        # Config files stay in root
        # "config.yaml": "config.yaml",
        # "config.example.yaml": "config.example.yaml",
        # "qwen_conveyance_config.py": "config/qwen_conveyance_config.py",
    }
    
    # Create legacy directories
    Path("scripts/legacy").mkdir(parents=True, exist_ok=True)
    Path("tests/legacy").mkdir(parents=True, exist_ok=True)
    
    for src, dst in moves.items():
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists():
            # Create parent directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(src_path), str(dst_path))
            logger.info(f"Moved {src} -> {dst}")
        else:
            logger.warning(f"Source file not found: {src}")


def create_main_cli():
    """Create main CLI entry point."""
    cli_content = '''#!/usr/bin/env python3
"""
HERMES Command Line Interface

Main entry point for all HERMES functionality.
"""

import argparse
import sys
from pathlib import Path

# Add HERMES to path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.cli import gather_papers
from hermes.processing import process_to_hades, build_isne_graph


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HERMES - Universal Data Pipeline for HADES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  gather    - Gather papers from various sources
  process   - Process documents to HADES format
  build     - Build ISNE graph from corpus
  
Examples:
  # Gather papers chronologically
  hermes-cli gather chronological --start-year 1998
  
  # Process documents
  hermes-cli process --input-dir ./data/papers --output-dir ./datasets/hades
  
  # Build ISNE graph
  hermes-cli build --paper-dirs ./data/papers --min-docs 2000
        """
    )
    
    parser.add_argument('--version', action='version', version='HERMES 0.1.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Gather command
    gather_parser = subparsers.add_parser('gather', help='Gather papers')
    gather_parser.set_defaults(func=gather_papers.main)
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('--input-dir', type=Path, required=True)
    process_parser.add_argument('--output-dir', type=Path, required=True)
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build ISNE graph')
    build_parser.add_argument('--paper-dirs', nargs='+', type=Path)
    build_parser.add_argument('--min-docs', type=int, default=2000)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if hasattr(args, 'func'):
        # Delegate to subcommand
        sys.argv = sys.argv[1:]  # Remove 'hermes-cli' from argv
        args.func()
    else:
        print(f"Command '{args.command}' not yet implemented")


if __name__ == "__main__":
    main()
'''
    
    with open("hermes-cli", "w") as f:
        f.write(cli_content)
    
    # Make executable
    Path("hermes-cli").chmod(0o755)
    logger.info("Created main CLI entry point: hermes-cli")


def create_readme():
    """Create updated README with new structure."""
    readme_content = '''# HERMES - Universal Data Pipeline for HADES

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
'''
    
    with open("README_NEW.md", "w") as f:
        f.write(readme_content)
    
    logger.info("Created new README: README_NEW.md")


def main():
    """Run the reorganization."""
    print("HERMES Reorganization Script")
    print("============================")
    print("This will reorganize HERMES files into a proper module structure.")
    print("\nNew structure:")
    print("- hermes/gathering/ - Paper gathering modules")
    print("- hermes/analysis/ - Analysis tools")
    print("- hermes/processing/ - Processing pipelines")
    print("- hermes/cli/ - Command-line interfaces")
    print("- scripts/ - Utility scripts")
    print("- tests/ - Test suite")
    print("\nOriginal files will be moved to appropriate locations.")
    
    if input("\nContinue? (y/n): ").lower() != 'y':
        return
    
    # Create directories
    create_directories()
    
    # Move files
    move_files()
    
    # Create main CLI
    create_main_cli()
    
    # Create new README
    create_readme()
    
    print("\n✓ Reorganization complete!")
    print("\nNext steps:")
    print("1. Review the new structure")
    print("2. Update imports in moved files")
    print("3. Test the new CLI with: ./hermes --help")
    print("4. Replace README.md with README_NEW.md when ready")


if __name__ == "__main__":
    main()