#!/usr/bin/env python3
"""
Process documents through HERMES pipeline and store in HADES-compatible format.
This creates a dataset aligned with ISNE and DSPy methodologies.
"""

import sys
from pathlib import Path
import logging
from typing import Dict, Any, List
import json
import numpy as np
from tqdm import tqdm

# Add HERMES to path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.core.config import load_config, setup_logging
from hermes.loaders import PDFLoader, OCRPDFLoader
from hermes.embedders import JinaV4Embedder, SentenceTransformerEmbedder
from hermes.extractors import ClaudeConveyanceAnalyzer, LocalConveyanceAnalyzer
from hermes.extractors.vllm_conveyance_analyzer import VLLMConveyanceAnalyzer
from hermes.core.adaptive_isne import AdaptiveISNE


def process_for_hades_with_analyzer(file_path: Path, config, analyzer) -> Dict[str, Any]:
    """
    Process a document for HADES RAG system with provided analyzer.
    
    Returns document in format compatible with:
    - ISNE embeddings for dynamic graph response
    - DSPy optimization for learning
    - Multi-dimensional information theory
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Load document
        loader = PDFLoader()
        doc_data = loader.load(file_path)
        text = doc_data.get('text', '')
        
        # Check for OCR need
        page_count = doc_data.get('page_count', 1)
        chars_per_page = len(text) / page_count if page_count > 0 else 0
        
        if chars_per_page < config.processing.ocr.min_chars_per_page:
            logger.info(f"Using OCR for {file_path.name}")
            ocr_loader = OCRPDFLoader()
            doc_data = ocr_loader.load(file_path)
            text = doc_data.get('content', '')
        
        # 2. Semantic analysis with provided analyzer
        conveyance_analysis = analyzer.analyze(text, doc_data.get('metadata'))
        
        # If reasoning chain is available, log it
        if hasattr(conveyance_analysis, 'reasoning_chain') and conveyance_analysis.reasoning_chain:
            logger.debug(f"Reasoning for {file_path.name}: {conveyance_analysis.reasoning_chain[:200]}...")
        
        # 3. Generate embeddings
        # Use local embedder for testing (switch to JinaV4 for production)
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2",
            truncate_dim=384,  # Smaller dimension for testing
            show_progress_bar=False
        )
        
        # Smart chunking for long documents
        chunks = []
        if len(text) > 8000:
            # Chunk at paragraph boundaries
            paragraphs = text.split('\n\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < 8000:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [text]
        
        # Embed chunks
        embeddings = embedder.embed_texts(chunks, show_progress=False)
        
        # 4. Calculate dimensional values
        
        # WHERE dimension (filesystem/structural)
        # ETHNOGRAPHIC NOTE: This is where Foucault meets filesystem
        # The WHERE vector doesn't encode "meaning" but rather position in an epistemic topology.
        # ISNE will learn that proximity in this space correlates with conceptual relationships,
        # citation patterns, and semantic similarity. The 102 dimensions compress an entire
        # archaeology of knowledge into a navigable manifold.
        where_vector = np.zeros(102)  # As specified in HADES
        
        # Encode hierarchical position (not "meaning" but relational topology)
        path_parts = str(file_path).split('/')
        where_vector[0] = len(path_parts)  # Depth in knowledge hierarchy
        
        # LCC-based position encoding (if using organized structure)
        if 'papers' in path_parts and len(path_parts) > path_parts.index('papers') + 2:
            # Extract LCC components from path like /Q/QA76.9.I52/filename.pdf
            idx = path_parts.index('papers')
            if idx + 1 < len(path_parts):
                main_class = path_parts[idx + 1]  # Q, P, G, etc.
                if main_class in 'QPBTHGNZ':  # Valid LCC main classes
                    where_vector[10 + ord(main_class) - ord('A')] = 1.0
            if idx + 2 < len(path_parts):
                subclass = path_parts[idx + 2]  # QA76.9, GN476, etc.
                # Hash subclass to distribute across dimensions 20-60
                subclass_hash = sum(ord(c) for c in subclass)
                where_vector[20 + (subclass_hash % 40)] = 1.0
        
        # Legacy encoding for backward compatibility
        where_vector[1] = 1.0 if 'anthropology' in str(file_path).lower() else 0.0
        where_vector[2] = 1.0 if 'information' in str(file_path).lower() else 0.0
        
        # Normalize to unit vector (preserves relationships, not magnitudes)
        where_magnitude = np.linalg.norm(where_vector)
        if where_magnitude > 0:
            where_vector = where_vector / where_magnitude
        
        # WHAT dimension (semantic - from embeddings)
        embedding_dim = embedder.get_dimension()
        what_vector = embeddings[0] if len(embeddings) > 0 else np.zeros(embedding_dim)
        
        # CONVEYANCE dimension (actionability)
        conveyance_vector = np.zeros(922)  # As specified in HADES
        # Encode conveyance analysis results
        conveyance_vector[0] = conveyance_analysis.implementation_fidelity
        conveyance_vector[1] = conveyance_analysis.actionability
        conveyance_vector[2] = conveyance_analysis.bridge_potential
        # Theory components (one-hot encoding)
        for i, component in enumerate(conveyance_analysis.theory_components[:50]):
            conveyance_vector[10 + i] = 1.0
        # Practice components
        for i, component in enumerate(conveyance_analysis.practice_components[:50]):
            conveyance_vector[60 + i] = 1.0
        # Normalize
        conv_magnitude = np.linalg.norm(conveyance_vector)
        if conv_magnitude > 0:
            conveyance_vector = conveyance_vector / conv_magnitude
        
        # 5. Create HADES-compatible document
        hades_doc = {
            # Core identifiers
            'doc_id': file_path.stem,
            'source_path': str(file_path),
            'doc_type': 'academic_paper',
            
            # Content
            'content': text,
            'chunks': chunks,
            'chunk_embeddings': embeddings.tolist(),
            
            # Dimensional vectors (for HADES multi-dimensional model)
            'dimensions': {
                'WHERE': {
                    'vector': where_vector.tolist(),
                    'magnitude': float(where_magnitude),
                    'metadata': {
                        'path': str(file_path),
                        'depth': len(path_parts),
                        'domain': 'anthropology' if 'anthropology' in str(file_path).lower() else 'information_theory'
                    }
                },
                'WHAT': {
                    'vector': what_vector.tolist(),
                    'magnitude': float(np.linalg.norm(what_vector)),
                    'metadata': {
                        'embedding_model': 'jina_v4',
                        'dimension': len(what_vector)
                    }
                },
                'CONVEYANCE': {
                    'vector': conveyance_vector.tolist(),
                    'magnitude': float(conv_magnitude),
                    'metadata': {
                        'implementation_fidelity': conveyance_analysis.implementation_fidelity,
                        'actionability': conveyance_analysis.actionability,
                        'bridge_potential': conveyance_analysis.bridge_potential,
                        'theory_components': conveyance_analysis.theory_components,
                        'practice_components': conveyance_analysis.practice_components,
                        'missing_links': conveyance_analysis.missing_links
                    }
                }
            },
            
            # For DSPy training
            'training_data': {
                'conveyance_reasoning': conveyance_analysis.reasoning,
                'implementation_suggestions': conveyance_analysis.implementation_suggestions,
                'confidence': conveyance_analysis.confidence
            },
            
            # For ISNE
            'graph_metadata': {
                'node_type': 'document',
                'ready_for_isne': True,
                'requires_ocr': 'OCR' in doc_data.get('loader', '')
            }
        }
        
        return hades_doc
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        raise


def process_for_hades(file_path: Path, config) -> Dict[str, Any]:
    """
    Process a document for HADES RAG system (backward compatible).
    Creates its own analyzer instance.
    """
    analyzer = VLLMConveyanceAnalyzer(
        model_name="Qwen/Qwen3-30B-A3B-FP8",
        max_model_len=20480,
        gpu_memory_utilization=0.75,
        enable_reasoning=True,
        reasoning_parser="deepseek_r1",
        lazy_load=True
    )
    try:
        return process_for_hades_with_analyzer(file_path, config, analyzer)
    finally:
        analyzer.unload_model()


def create_hades_dataset(papers_dir: Path, output_dir: Path, config, limit=None):
    """Create a HADES-compatible dataset from papers."""
    logger = logging.getLogger(__name__)
    
    # Get papers
    pdf_files = list(papers_dir.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]
    
    logger.info(f"Processing {len(pdf_files)} papers for HADES dataset")
    
    # Create single analyzer instance for all documents
    analyzer = VLLMConveyanceAnalyzer(
        model_name="Qwen/Qwen3-30B-A3B-FP8",
        max_model_len=20480,  # 20K context
        gpu_memory_utilization=0.75,
        enable_reasoning=True,
        reasoning_parser="deepseek_r1",
        lazy_load=True  # Load on first use
    )
    
    # Process each paper
    documents = []
    training_examples = []  # Collect for DSPy fine-tuning
    
    for pdf_file in tqdm(pdf_files, desc="Creating HADES dataset"):
        try:
            hades_doc = process_for_hades_with_analyzer(pdf_file, config, analyzer)
            documents.append(hades_doc)
            
            # Collect training example if using model
            if hades_doc.get('training_data', {}).get('confidence', 0) > 0.7:
                training_examples.append({
                    'document': pdf_file.stem,
                    'conveyance_analysis': hades_doc['training_data']
                })
        except Exception as e:
            logger.error(f"Skipping {pdf_file}: {e}")
    
    # Save training examples for DSPy
    if training_examples:
        analyzer._save_training_data(training_examples)
        logger.info(f"Saved {len(training_examples)} training examples")
    
    # Unload model to free GPU memory
    analyzer.unload_model()
    
    # Initialize ISNE for graph embeddings
    if documents and config.isne.enabled:
        logger.info("Training ISNE on document collection...")
        isne = AdaptiveISNE(
            embedding_dim=config.isne.embedding_dim,
            use_dspy=config.isne.use_adaptive
        )
        
        # Create nodes and edges for ISNE training
        nodes = []
        edges = []
        
        for i, doc in enumerate(documents):
            nodes.append({
                'id': f"doc_{i}",
                'doc_id': doc['doc_id'],
                'type': 'document',
                'features': doc['dimensions']
            })
        
        # Create edges based on similarity
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                # Calculate multi-dimensional similarity
                what_sim = np.dot(
                    documents[i]['dimensions']['WHAT']['vector'],
                    documents[j]['dimensions']['WHAT']['vector']
                )
                conv_sim = np.dot(
                    documents[i]['dimensions']['CONVEYANCE']['vector'],
                    documents[j]['dimensions']['CONVEYANCE']['vector']
                )
                
                if what_sim > 0.7 or conv_sim > 0.8:
                    edges.append({
                        'source': f"doc_{i}",
                        'target': f"doc_{j}",
                        'weight': (what_sim + conv_sim) / 2
                    })
        
        # Train ISNE
        isne.train_on_graph(nodes, edges)
        
        # Add ISNE embeddings to documents
        for i, doc in enumerate(documents):
            doc['isne_embedding'] = nodes[i].get('isne_embedding', []).tolist()
    
    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save documents
    docs_file = output_dir / "hades_documents.json"
    with open(docs_file, 'w') as f:
        json.dump(documents, f, indent=2)
    
    # Save metadata
    embedding_dim = 384 if documents else 1024  # Get actual dimension used
    metadata = {
        'total_documents': len(documents),
        'dimensions': {
            'WHERE': 102,
            'WHAT': embedding_dim,
            'CONVEYANCE': 922
        },
        'isne_enabled': config.isne.enabled,
        'domains': ['anthropology', 'information_theory'],
        'purpose': 'Cross-domain bridge detection',
        'methodology': 'Multi-dimensional information theory with ISNE and DSPy',
        'embedder': 'sentence-transformers/all-MiniLM-L6-v2'  # Track which model was used
    }
    
    meta_file = output_dir / "dataset_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"HADES dataset saved to {output_dir}")
    logger.info(f"Total documents: {len(documents)}")
    
    # Show summary
    high_bridge = [d for d in documents if d['dimensions']['CONVEYANCE']['metadata']['bridge_potential'] > 0.7]
    logger.info(f"High bridge potential papers: {len(high_bridge)}")
    
    return documents


def main():
    """Create HADES dataset from papers."""
    import sys
    
    config = load_config()
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    
    # Create anthropology-information theory dataset
    papers_dir = Path("/home/todd/olympus/data/papers")
    output_dir = Path("./datasets/anthropology_information_theory")
    
    logger.info("Creating HADES dataset for anthropology-information theory bridges...")
    
    # Check for command line argument for batch mode
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # Batch mode - process all without asking
        logger.info("Running in batch mode - processing all papers")
        documents = create_hades_dataset(papers_dir, output_dir, config)
    else:
        # Interactive mode - start with test batch
        documents = create_hades_dataset(papers_dir, output_dir, config, limit=10)
        
        if sys.stdin.isatty():
            if input("\nCreate full dataset? (y/n): ").lower() == 'y':
                documents = create_hades_dataset(papers_dir, output_dir, config)
        else:
            logger.info("Non-interactive mode detected. Run with --batch to process all papers")
    
    logger.info("\nDataset creation complete!")
    logger.info("This dataset can be used with HADES MCP tool for RAG queries")


if __name__ == "__main__":
    main()