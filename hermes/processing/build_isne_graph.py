#!/usr/bin/env python3
"""
Build ISNE Graph from Document Corpus

This script:
1. Processes all downloaded papers through HERMES pipeline
2. Stores them in ArangoDB with multi-dimensional vectors
3. Builds a graph structure based on similarities
4. Trains ISNE on the resulting graph

This is the critical step that enables everything else!
"""

import sys
from pathlib import Path
import logging
import json
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
from datetime import datetime

# Add HERMES to path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.core.config import load_config, setup_logging
from process_to_hades import process_for_hades_with_analyzer, create_hades_dataset
from hermes.extractors.vllm_conveyance_analyzer import VLLMConveyanceAnalyzer
from hermes.core.adaptive_isne import AdaptiveISNE

# Try to import ArangoDB
try:
    from arango import ArangoClient
    ARANGO_AVAILABLE = True
except ImportError:
    ARANGO_AVAILABLE = False
    logging.warning("ArangoDB not available. Will save to JSON files instead.")

logger = logging.getLogger(__name__)


class ISNEGraphBuilder:
    """Build and train ISNE on document corpus."""
    
    def __init__(self, config, db_config: Dict = None):
        self.config = config
        self.db_config = db_config or {
            'host': 'http://localhost:8529',
            'username': 'root',
            'password': 'openSesame',
            'database': 'hades_papers'
        }
        
        # Initialize database connection if available
        self.db = None
        self.graph = None
        if ARANGO_AVAILABLE and self.db_config:
            self._init_database()
    
    def _init_database(self):
        """Initialize ArangoDB connection and collections."""
        try:
            client = ArangoClient(hosts=self.db_config['host'])
            
            # Connect to system database first
            sys_db = client.db('_system', 
                             username=self.db_config['username'],
                             password=self.db_config['password'])
            
            # Create database if it doesn't exist
            if not sys_db.has_database(self.db_config['database']):
                sys_db.create_database(self.db_config['database'])
            
            # Connect to our database
            self.db = client.db(self.db_config['database'],
                              username=self.db_config['username'],
                              password=self.db_config['password'])
            
            # Create collections
            if not self.db.has_collection('papers'):
                self.db.create_collection('papers')
            
            if not self.db.has_collection('paper_edges'):
                self.db.create_collection('paper_edges', edge=True)
            
            # Create graph
            if not self.db.has_graph('paper_graph'):
                self.graph = self.db.create_graph(
                    'paper_graph',
                    edge_definitions=[{
                        'edge_collection': 'paper_edges',
                        'from_vertex_collections': ['papers'],
                        'to_vertex_collections': ['papers']
                    }]
                )
            else:
                self.graph = self.db.graph('paper_graph')
                
            logger.info("ArangoDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ArangoDB: {e}")
            self.db = None
    
    def process_corpus(self, 
                      paper_dirs: List[Path],
                      output_dir: Path,
                      limit: int = None) -> Tuple[List[Dict], int]:
        """
        Process all papers in the given directories.
        
        Returns:
            (documents, total_processed)
        """
        # Collect all PDF files
        all_pdfs = []
        for paper_dir in paper_dirs:
            if paper_dir.exists():
                pdfs = list(paper_dir.glob("*.pdf"))
                logger.info(f"Found {len(pdfs)} PDFs in {paper_dir}")
                all_pdfs.extend(pdfs)
        
        if limit:
            all_pdfs = all_pdfs[:limit]
            
        logger.info(f"Processing {len(all_pdfs)} papers total")
        
        # Create analyzer (shared for efficiency)
        analyzer = VLLMConveyanceAnalyzer(
            model_name="Qwen/Qwen3-30B-A3B-FP8",
            max_model_len=20480,
            gpu_memory_utilization=0.9,
            lazy_load=True
        )
        
        documents = []
        processed = 0
        failed = 0
        
        # Process in batches to manage memory
        batch_size = 10
        
        for i in tqdm(range(0, len(all_pdfs), batch_size), desc="Processing batches"):
            batch = all_pdfs[i:i+batch_size]
            
            for pdf_file in batch:
                try:
                    # Process document
                    hades_doc = process_for_hades_with_analyzer(pdf_file, self.config, analyzer)
                    
                    # Add metadata about source
                    hades_doc['source_file'] = str(pdf_file)
                    hades_doc['processed_at'] = datetime.now().isoformat()
                    
                    documents.append(hades_doc)
                    processed += 1
                    
                    # Store in database if available
                    if self.db:
                        self._store_document(hades_doc)
                        
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file}: {e}")
                    failed += 1
            
            # Save checkpoint every 100 documents
            if processed % 100 == 0:
                self._save_checkpoint(documents, output_dir, processed)
        
        # Unload model to free GPU
        analyzer.unload_model()
        
        logger.info(f"Processed {processed} documents successfully, {failed} failed")
        
        # Save final results
        self._save_documents(documents, output_dir)
        
        return documents, processed
    
    def _store_document(self, doc: Dict):
        """Store document in ArangoDB."""
        if not self.db:
            return
            
        try:
            papers_col = self.db.collection('papers')
            
            # Prepare document for storage
            arango_doc = {
                '_key': doc['doc_id'],
                'title': doc.get('metadata', {}).get('title', doc['doc_id']),
                'content': doc['content'][:1000],  # Truncate for storage
                'where_vector': doc['dimensions']['WHERE']['vector'],
                'what_vector': doc['dimensions']['WHAT']['vector'],
                'conveyance_vector': doc['dimensions']['CONVEYANCE']['vector'],
                'metadata': doc.get('metadata', {}),
                'conveyance_scores': doc['dimensions']['CONVEYANCE']['metadata'],
                'processed_at': doc['processed_at']
            }
            
            # Insert or update
            papers_col.insert(arango_doc, overwrite=True)
            
        except Exception as e:
            logger.warning(f"Failed to store document {doc['doc_id']}: {e}")
    
    def build_graph(self, documents: List[Dict], similarity_threshold: float = 0.6):
        """
        Build graph structure based on multi-dimensional similarities.
        
        Creates edges between papers when their combined similarity exceeds threshold.
        """
        logger.info(f"Building graph from {len(documents)} documents")
        
        edges_created = 0
        
        # Calculate similarities between all pairs
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                doc_i = documents[i]
                doc_j = documents[j]
                
                # Calculate dimensional similarities
                where_sim = self._cosine_similarity(
                    doc_i['dimensions']['WHERE']['vector'],
                    doc_j['dimensions']['WHERE']['vector']
                )
                
                what_sim = self._cosine_similarity(
                    doc_i['dimensions']['WHAT']['vector'],
                    doc_j['dimensions']['WHAT']['vector']
                )
                
                conv_sim = self._cosine_similarity(
                    doc_i['dimensions']['CONVEYANCE']['vector'],
                    doc_j['dimensions']['CONVEYANCE']['vector']
                )
                
                # Combined similarity (multiplicative model)
                combined_sim = (where_sim * what_sim * conv_sim) ** (1/3)  # Geometric mean
                
                if combined_sim >= similarity_threshold:
                    # Create edge
                    edge_data = {
                        'from': doc_i['doc_id'],
                        'to': doc_j['doc_id'],
                        'weight': combined_sim,
                        'where_similarity': where_sim,
                        'what_similarity': what_sim,
                        'conveyance_similarity': conv_sim
                    }
                    
                    if self.db:
                        self._create_edge(doc_i['doc_id'], doc_j['doc_id'], edge_data)
                    
                    edges_created += 1
        
        logger.info(f"Created {edges_created} edges with threshold {similarity_threshold}")
        
        return edges_created
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _create_edge(self, from_id: str, to_id: str, edge_data: Dict):
        """Create edge in ArangoDB."""
        if not self.db:
            return
            
        try:
            edges_col = self.db.collection('paper_edges')
            
            edge_doc = {
                '_from': f'papers/{from_id}',
                '_to': f'papers/{to_id}',
                **edge_data
            }
            
            edges_col.insert(edge_doc, overwrite=True)
            
        except Exception as e:
            logger.warning(f"Failed to create edge {from_id} -> {to_id}: {e}")
    
    def train_isne(self, documents: List[Dict], output_dir: Path):
        """Train ISNE on the document graph."""
        logger.info("Training ISNE on document collection...")
        
        # Initialize ISNE
        isne = AdaptiveISNE(
            embedding_dim=self.config.isne.embedding_dim,
            use_dspy=self.config.isne.use_adaptive
        )
        
        # Prepare nodes and edges for ISNE
        nodes = []
        edges = []
        
        # Create nodes
        for doc in documents:
            nodes.append({
                'id': doc['doc_id'],
                'type': 'paper',
                'features': {
                    'where': doc['dimensions']['WHERE']['vector'],
                    'what': doc['dimensions']['WHAT']['vector'],
                    'conveyance': doc['dimensions']['CONVEYANCE']['vector']
                }
            })
        
        # Create edges based on similarities
        edge_count = 0
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                # Calculate combined similarity
                where_sim = self._cosine_similarity(
                    documents[i]['dimensions']['WHERE']['vector'],
                    documents[j]['dimensions']['WHERE']['vector']
                )
                what_sim = self._cosine_similarity(
                    documents[i]['dimensions']['WHAT']['vector'],
                    documents[j]['dimensions']['WHAT']['vector']
                )
                conv_sim = self._cosine_similarity(
                    documents[i]['dimensions']['CONVEYANCE']['vector'],
                    documents[j]['dimensions']['CONVEYANCE']['vector']
                )
                
                combined_sim = (where_sim * what_sim * conv_sim) ** (1/3)
                
                if combined_sim > 0.5:  # Lower threshold for ISNE
                    edges.append({
                        'source': documents[i]['doc_id'],
                        'target': documents[j]['doc_id'],
                        'weight': combined_sim
                    })
                    edge_count += 1
        
        logger.info(f"Created {edge_count} edges for ISNE training")
        
        # Train ISNE
        isne.train_on_graph(nodes, edges)
        
        # Save ISNE model
        model_path = output_dir / "isne_model.pkl"
        isne.save_model(str(model_path))
        logger.info(f"ISNE model saved to {model_path}")
        
        # Save embeddings with documents
        for i, doc in enumerate(documents):
            if i < len(nodes) and 'isne_embedding' in nodes[i]:
                doc['isne_embedding'] = nodes[i]['isne_embedding'].tolist()
        
        return isne
    
    def _save_checkpoint(self, documents: List[Dict], output_dir: Path, processed: int):
        """Save checkpoint during processing."""
        checkpoint_file = output_dir / f"checkpoint_{processed}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed': processed,
                'documents': documents
            }, f)
        logger.info(f"Saved checkpoint at {processed} documents")
    
    def _save_documents(self, documents: List[Dict], output_dir: Path):
        """Save final document collection."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        docs_file = output_dir / "hades_documents_complete.json"
        with open(docs_file, 'w') as f:
            json.dump(documents, f, indent=2)
        
        # Save metadata summary
        metadata = {
            'total_documents': len(documents),
            'timestamp': datetime.now().isoformat(),
            'dimensions': {
                'WHERE': 102,
                'WHAT': documents[0]['dimensions']['WHAT']['vector'].__len__() if documents else 1024,
                'CONVEYANCE': 922
            },
            'has_isne_embeddings': any('isne_embedding' in d for d in documents)
        }
        
        meta_file = output_dir / "dataset_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(documents)} documents to {output_dir}")


def main():
    """Build ISNE graph from document corpus."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ISNE graph from papers")
    parser.add_argument("--paper-dirs", nargs="+", type=Path,
                       default=[Path("./data/ml_papers_chronological"),
                               Path("./data/priority_papers"),
                               Path("./data/quantum_observer_papers")],
                       help="Directories containing papers")
    parser.add_argument("--output-dir", type=Path, 
                       default=Path("./datasets/isne_baseline"),
                       help="Output directory for processed data")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of papers to process")
    parser.add_argument("--skip-db", action="store_true",
                       help="Skip ArangoDB and only save to files")
    parser.add_argument("--similarity-threshold", type=float, default=0.6,
                       help="Similarity threshold for graph edges")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    setup_logging(config)
    
    # Database config
    db_config = None if args.skip_db else {
        'host': 'http://localhost:8529',
        'username': 'root',
        'password': 'openSesame',
        'database': 'hades_papers'
    }
    
    # Count total papers
    total_papers = 0
    for paper_dir in args.paper_dirs:
        if paper_dir.exists():
            count = len(list(paper_dir.glob("*.pdf")))
            print(f"  {paper_dir}: {count} papers")
            total_papers += count
    
    print(f"""
ISNE Graph Builder
==================

This will:
1. Process papers through HERMES pipeline
2. Extract multi-dimensional vectors (WHERE, WHAT, CONVEYANCE)
3. Store in {'ArangoDB' if not args.skip_db else 'JSON files'}
4. Build similarity graph with threshold {args.similarity_threshold}
5. Train ISNE on the resulting graph

Paper sources:
{chr(10).join(f"  - {d}" for d in args.paper_dirs)}

Total papers found: {total_papers}
Processing limit: {args.limit or 'None (all papers)'}
Output directory: {args.output_dir}

NOTE: This will take several hours for 2000+ papers.
Each paper requires:
- PDF text extraction
- Conveyance analysis with Qwen3-30B
- Embedding generation
- Multi-dimensional vector calculation
""")
    
    if input("Continue? (y/n): ").lower() != 'y':
        return
    
    # Create builder
    builder = ISNEGraphBuilder(config, db_config)
    
    # Process corpus
    start_time = time.time()
    documents, processed = builder.process_corpus(
        args.paper_dirs,
        args.output_dir,
        limit=args.limit
    )
    
    if processed < 2000:
        logger.warning(f"Only processed {processed} documents. ISNE paper recommends minimum 2000.")
        if input(f"Continue with {processed} documents? (y/n): ").lower() != 'y':
            return
    
    # Build graph
    edges = builder.build_graph(documents, args.similarity_threshold)
    
    # Train ISNE
    if documents:
        isne = builder.train_isne(documents, args.output_dir)
    
    # Report results
    elapsed = time.time() - start_time
    print(f"""
=== Processing Complete ===
Documents processed: {processed}
Graph edges created: {edges}
Time elapsed: {elapsed/60:.1f} minutes
Output saved to: {args.output_dir}

Next steps:
1. Verify ISNE embeddings: check {args.output_dir}/isne_model.pkl
2. Test bridge detection on this corpus
3. Run validation queries
""")


if __name__ == "__main__":
    main()