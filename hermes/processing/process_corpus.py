#!/usr/bin/env python3
"""
Process academic papers through HERMES pipeline.
"""

import sys
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List, Dict, Any
import json

# Add HERMES to path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.core.config import load_config, setup_logging
from hermes.loaders import PDFLoader, OCRPDFLoader
from hermes.embedders import JinaV4Embedder
from hermes.extractors import ConveyanceAnalyzer
from hermes.storage import ArangoStorage


def process_single_document(file_path: Path, config, storage) -> Dict[str, Any]:
    """Process a single document through the pipeline."""
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Load document
        logger.info(f"Loading: {file_path.name}")
        
        # Try regular PDF loader first
        loader = PDFLoader()
        doc_data = loader.load(file_path)
        
        # Check if we need OCR
        text = doc_data.get('text', '')
        page_count = doc_data.get('page_count', 1)
        chars_per_page = len(text) / page_count if page_count > 0 else 0
        
        if chars_per_page < config.processing.ocr.min_chars_per_page:
            logger.info(f"Document appears to need OCR ({chars_per_page:.1f} chars/page)")
            if config.processing.ocr.enabled:
                ocr_loader = OCRPDFLoader(
                    dpi=config.processing.ocr.dpi,
                    language=config.processing.ocr.language
                )
                doc_data = ocr_loader.load(file_path)
                text = doc_data.get('content', '')
        
        # 2. Analyze conveyance
        analyzer = ConveyanceAnalyzer()
        conveyance_metrics = analyzer.analyze(text, doc_data.get('metadata'))
        
        # 3. Generate embeddings
        embedder = JinaV4Embedder(
            truncate_dim=config.embeddings.models['jina_v4'].truncate_dim,
            adapter_mask=config.embeddings.models['jina_v4'].adapter_mask
        )
        
        # Chunk text if needed
        if len(text) > 10000:  # Rough limit
            # Simple chunking for now
            chunks = []
            chunk_size = 8000
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i+chunk_size])
        else:
            chunks = [text]
        
        embeddings = embedder.embed_texts(chunks, show_progress=False)
        
        # 4. Prepare document for storage
        document = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'text': text,
            'chunks': chunks,
            'embeddings': embeddings.tolist(),
            'metadata': {
                'loader': doc_data.get('loader', 'PDFLoader'),
                'page_count': page_count,
                'ocr_performed': 'OCR' in doc_data.get('loader', ''),
                'ocr_confidence': doc_data.get('ocr_confidence', None)
            },
            'conveyance': {
                'implementation_fidelity': conveyance_metrics.implementation_fidelity,
                'actionability': conveyance_metrics.actionability,
                'bridge_potential': conveyance_metrics.bridge_potential,
                'has_algorithms': conveyance_metrics.has_algorithms,
                'has_equations': conveyance_metrics.has_equations,
                'has_examples': conveyance_metrics.has_examples,
                'has_code': conveyance_metrics.has_code,
                'complexity_score': conveyance_metrics.complexity_score
            },
            'dimensions': {
                'WHERE': {
                    'path': str(file_path),
                    'directory': str(file_path.parent),
                    'file_type': file_path.suffix
                },
                'WHAT': {
                    'embeddings': 'stored',
                    'semantic_dim': len(embeddings[0]) if len(embeddings) > 0 else 0
                },
                'CONVEYANCE': conveyance_metrics.implementation_fidelity
            }
        }
        
        # 5. Store in ArangoDB
        doc_id = storage.store_document(document)
        
        logger.info(f"Processed: {file_path.name} -> {doc_id}")
        
        return {
            'success': True,
            'doc_id': doc_id,
            'file': file_path.name,
            'conveyance': conveyance_metrics.implementation_fidelity
        }
        
    except Exception as e:
        logger.error(f"Failed to process {file_path.name}: {e}")
        return {
            'success': False,
            'file': file_path.name,
            'error': str(e)
        }


def process_corpus(papers_dir: Path, config, limit: int = None):
    """Process all papers in directory."""
    logger = logging.getLogger(__name__)
    
    # Initialize storage
    storage = ArangoStorage.from_config(config)
    storage.initialize()
    
    # Get all PDFs
    pdf_files = list(papers_dir.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]
    
    logger.info(f"Found {len(pdf_files)} PDFs to process")
    
    # Process each document
    results = []
    successful = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing papers"):
        result = process_single_document(pdf_file, config, storage)
        results.append(result)
        if result['success']:
            successful += 1
    
    # Summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Successfully processed: {successful}/{len(pdf_files)}")
    
    # Save results
    results_path = Path("processing_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Show conveyance distribution
    conveyance_scores = [r['conveyance'] for r in results if r.get('conveyance') is not None]
    if conveyance_scores:
        avg_conveyance = sum(conveyance_scores) / len(conveyance_scores)
        logger.info(f"\nAverage conveyance score: {avg_conveyance:.3f}")
        
        # Group by conveyance level
        high_conveyance = [r for r in results if r.get('conveyance', 0) > 0.7]
        medium_conveyance = [r for r in results if 0.3 < r.get('conveyance', 0) <= 0.7]
        low_conveyance = [r for r in results if r.get('conveyance', 0) <= 0.3]
        
        logger.info(f"High conveyance (>0.7): {len(high_conveyance)} papers")
        logger.info(f"Medium conveyance (0.3-0.7): {len(medium_conveyance)} papers")
        logger.info(f"Low conveyance (<=0.3): {len(low_conveyance)} papers")
    
    return results


def main():
    """Main entry point."""
    # Load configuration
    config = load_config(Path("config.yaml") if Path("config.yaml").exists() else None)
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting HERMES document processing pipeline")
    
    # Process papers
    papers_dir = Path("/home/todd/olympus/data/papers")
    
    # Start with a small batch
    logger.info("\nProcessing initial batch of 5 papers...")
    results = process_corpus(papers_dir, config, limit=5)
    
    # Check if user wants to continue
    if input("\nProcess all papers? (y/n): ").lower() == 'y':
        results = process_corpus(papers_dir, config)


if __name__ == "__main__":
    main()