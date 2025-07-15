"""
Main pipeline orchestrator for HERMES.

This module implements the complete HERMES pipeline that processes documents
from loading through embedding to storage, following the dimensional model
of WHERE × WHAT × CONVEYANCE.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from hermes.core.base import BaseLoader, BaseEmbedder, BaseStorage, BaseAnalyzer
from hermes.core.document_model import Document, DocumentFactory, DocumentType
from hermes.core.directory_traverser import DirectoryTraverser
from hermes.core.adaptive_isne import AdaptiveISNE
from hermes.loaders import TextLoader, PythonASTLoader, DoclingLoader
from hermes.embedders import JinaV4Embedder
from hermes.storage import ArangoStorage
from hermes.analyzers.conveyance_analyzer import ConveyanceAnalyzer
from hermes.optimization.dspy_optimizers import HERMESPipelineOptimizer


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for HERMES pipeline."""
    # Processing options
    batch_size: int = 100
    max_workers: int = mp.cpu_count()
    use_async: bool = True
    
    # Directory traversal
    respect_gitignore: bool = True
    create_directory_nodes: bool = True
    
    # Embedding options
    embedding_models: List[str] = None
    embedding_batch_size: int = 32
    
    # Storage options
    storage_backend: str = "arango"
    storage_config: Dict[str, Any] = None
    
    # Analysis options
    analyze_conveyance: bool = True
    use_dspy_optimization: bool = False
    
    # ISNE options
    use_adaptive_isne: bool = True
    isne_embedding_dim: int = 128
    isne_update_interval: int = 100  # Update positions every N documents
    
    # Chunking options
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    def __post_init__(self):
        if self.embedding_models is None:
            self.embedding_models = ["jina_v4"]
        if self.storage_config is None:
            self.storage_config = {}


class HERMESPipeline:
    """
    Main orchestrator for the HERMES document processing pipeline.
    
    The pipeline follows these stages:
    1. Load: Extract content from files
    2. Analyze: Extract metadata and assess conveyance
    3. Embed: Generate vector representations
    4. Store: Save to graph database with relationships
    
    Each stage can be customized with different implementations.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        loaders: Optional[Dict[str, BaseLoader]] = None,
        embedders: Optional[Dict[str, BaseEmbedder]] = None,
        analyzers: Optional[Dict[str, BaseAnalyzer]] = None,
        storage: Optional[BaseStorage] = None
    ):
        """
        Initialize HERMES pipeline.
        
        Args:
            config: Pipeline configuration
            loaders: Dictionary of file loaders by extension
            embedders: Dictionary of embedders by name
            analyzers: Dictionary of analyzers by name
            storage: Storage backend
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.loaders = loaders or self._default_loaders()
        self.embedders = embedders or self._default_embedders()
        self.analyzers = analyzers or self._default_analyzers()
        self.storage = storage or self._default_storage()
        
        # Initialize directory traverser
        self.traverser = DirectoryTraverser(
            storage=self.storage,
            loaders=self.loaders,
            respect_gitignore=self.config.respect_gitignore,
            create_directory_nodes=self.config.create_directory_nodes,
            batch_size=self.config.batch_size
        )
        
        # Initialize Adaptive ISNE if enabled
        self.isne = None
        if self.config.use_adaptive_isne:
            self.isne = AdaptiveISNE(
                embedding_dim=self.config.isne_embedding_dim,
                use_dspy=self.config.use_dspy_optimization
            )
            logger.info("Initialized Adaptive ISNE for dynamic graph embedding")
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "documents_failed": 0,
            "embeddings_generated": 0,
            "relationships_created": 0,
            "isne_updates": 0
        }
        
        # Thread/Process pools for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
    
    def _default_loaders(self) -> Dict[str, BaseLoader]:
        """Get default loaders."""
        return {
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".py": PythonASTLoader(),
            ".pdf": DoclingLoader(),
        }
    
    def _default_embedders(self) -> Dict[str, BaseEmbedder]:
        """Get default embedders."""
        return {
            "jina_v4": JinaV4Embedder()
        }
    
    def _default_analyzers(self) -> Dict[str, BaseAnalyzer]:
        """Get default analyzers."""
        return {
            "conveyance": ConveyanceAnalyzer()
        }
    
    def _default_storage(self) -> BaseStorage:
        """Get default storage backend."""
        return ArangoStorage(**self.config.storage_config)
    
    async def process_directory(
        self,
        directory_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Process an entire directory tree.
        
        This is the main entry point for batch processing. It uses the
        directory-first approach to build graph structure before processing files.
        
        Args:
            directory_path: Root directory to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing statistics
        """
        directory_path = Path(directory_path)
        logger.info(f"Starting HERMES pipeline for {directory_path}")
        
        # Phase 1: Build directory structure (synchronous)
        logger.info("Phase 1: Building directory structure")
        traversal_stats = self.traverser.traverse_and_build(
            directory_path,
            progress_callback=None  # We'll handle progress differently
        )
        
        # Phase 2: Process documents with full pipeline
        logger.info("Phase 2: Processing documents through pipeline")
        
        # Get all documents from storage that need processing
        documents_to_process = self._get_unprocessed_documents()
        total_docs = len(documents_to_process)
        
        # Process in batches
        for i in range(0, total_docs, self.config.batch_size):
            batch = documents_to_process[i:i + self.config.batch_size]
            
            if self.config.use_async:
                await self._process_batch_async(batch)
            else:
                self._process_batch_sync(batch)
            
            if progress_callback:
                progress_callback(
                    f"Processing batch {i//self.config.batch_size + 1}",
                    min(i + self.config.batch_size, total_docs),
                    total_docs
                )
        
        # Phase 3: Discover relationships
        logger.info("Phase 3: Discovering semantic relationships")
        self._discover_relationships()
        
        # Compile statistics
        final_stats = {
            **traversal_stats,
            **self.stats,
            "total_documents": total_docs
        }
        
        logger.info(f"Pipeline complete: {final_stats}")
        return final_stats
    
    async def process_document(
        self,
        file_path: Union[str, Path],
        force_reload: bool = False
    ) -> Optional[Document]:
        """
        Process a single document through the pipeline.
        
        Args:
            file_path: Path to document
            force_reload: Force reprocessing even if already in storage
            
        Returns:
            Processed document or None if failed
        """
        file_path = Path(file_path)
        
        try:
            # Check if already processed
            doc_id = f"file:{file_path}"
            if not force_reload:
                existing = self.storage.retrieve_document(doc_id)
                if existing and existing.get("pipeline_stage") == "stored":
                    logger.debug(f"Document {file_path} already processed")
                    return Document.from_dict(existing)
            
            # Load document
            document = await self._load_document(file_path)
            if not document:
                return None
            
            # Analyze document
            if self.config.analyze_conveyance:
                document = await self._analyze_document(document)
            
            # Generate embeddings
            document = await self._embed_document(document)
            
            # Store document
            self._store_document(document)
            
            self.stats["documents_processed"] += 1
            return document
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.stats["documents_failed"] += 1
            return None
    
    async def _load_document(self, file_path: Path) -> Optional[Document]:
        """Load document from file."""
        # Find appropriate loader
        loader = None
        for ext, l in self.loaders.items():
            if l.can_load(file_path):
                loader = l
                break
        
        if not loader:
            logger.warning(f"No loader for {file_path}")
            return None
        
        try:
            # Load content
            loader_output = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                loader.load,
                file_path
            )
            
            # Create document
            document = DocumentFactory.from_loader_output(
                file_path,
                loader_output
            )
            
            document.update_stage("loaded")
            return document
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    async def _analyze_document(self, document: Document) -> Document:
        """Analyze document for metadata and conveyance."""
        try:
            # Run analyzers
            for name, analyzer in self.analyzers.items():
                if name == "conveyance" and self.config.analyze_conveyance:
                    analysis = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        analyzer.analyze,
                        document
                    )
                    
                    # Update conveyance scores
                    document.conveyance.implementation_fidelity = analysis.get("implementation_fidelity", 0.0)
                    document.conveyance.actionability = analysis.get("actionability", 0.0)
                    document.conveyance.bridge_potential = analysis.get("bridge_potential", 0.0)
                    document.conveyance.amplification_score = analysis.get("amplification_score", 0.0)
            
            document.update_stage("analyzed")
            return document
            
        except Exception as e:
            logger.error(f"Failed to analyze document {document.doc_id}: {e}")
            document.add_error("analysis", e)
            return document
    
    async def _embed_document(self, document: Document) -> Document:
        """Generate embeddings for document."""
        try:
            # Initialize embeddings if needed
            if not document.embeddings:
                from hermes.core.document_model import Embeddings
                document.embeddings = Embeddings()
            
            # Generate embeddings with each model
            for model_name in self.config.embedding_models:
                if model_name in self.embedders:
                    embedder = self.embedders[model_name]
                    
                    # Generate embedding
                    embedding = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        embedder.embed,
                        document.content.cleaned_text or document.content.raw_text
                    )
                    
                    # Store based on model type
                    if model_name == "jina_v4":
                        document.embeddings.jina_semantic = embedding
                        document.embeddings.embedding_model_versions["jina_v4"] = "1.0"
                    
                    self.stats["embeddings_generated"] += 1
            
            document.update_stage("embedded")
            return document
            
        except Exception as e:
            logger.error(f"Failed to embed document {document.doc_id}: {e}")
            document.add_error("embedding", e)
            return document
    
    def _store_document(self, document: Document) -> bool:
        """Store document in database and update ISNE if enabled."""
        try:
            # Convert to storage format
            doc_data = document.to_dict()
            
            # Store in database
            success = self.storage.store_document(document.doc_id, doc_data)
            
            if success:
                document.update_stage("stored")
                
                # Add to ISNE graph if enabled
                if self.isne and document.embeddings:
                    self.isne.add_document(document)
                    
                    # Update positions periodically
                    if self.stats["documents_processed"] % self.config.isne_update_interval == 0:
                        logger.info("Updating ISNE positions...")
                        self.isne.update_positions(iterations=5)
                        self.stats["isne_updates"] += 1
                        
                        # Store updated positions in metadata
                        self._store_isne_positions()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to store document {document.doc_id}: {e}")
            document.add_error("storage", e)
            return False
    
    async def _process_batch_async(self, documents: List[Dict[str, Any]]):
        """Process a batch of documents asynchronously."""
        tasks = []
        
        for doc_data in documents:
            file_path = Path(doc_data.get("metadata", {}).get("absolute_path", ""))
            if file_path.exists():
                task = self.process_document(file_path)
                tasks.append(task)
        
        # Process all documents in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
    
    def _process_batch_sync(self, documents: List[Dict[str, Any]]):
        """Process a batch of documents synchronously."""
        for doc_data in documents:
            file_path = Path(doc_data.get("metadata", {}).get("absolute_path", ""))
            if file_path.exists():
                # Run async function in sync context
                asyncio.run(self.process_document(file_path))
    
    def _get_unprocessed_documents(self) -> List[Dict[str, Any]]:
        """Get documents that need processing from storage."""
        # This would query the storage for documents that are not fully processed
        # For now, return empty list as documents are processed during traversal
        return []
    
    def _discover_relationships(self):
        """Discover semantic relationships between documents using ISNE neighborhoods."""
        if not self.isne:
            logger.info("ISNE not enabled, skipping semantic relationship discovery")
            return
            
        logger.info("Discovering semantic relationships from ISNE neighborhoods")
        
        # Get final positions after all documents processed
        self.isne.update_positions(iterations=10)
        
        # Create edges based on ISNE neighborhoods
        edges_created = 0
        for node_id, node in self.isne.nodes.items():
            for neighbor_id in node.neighbors:
                # Create semantic edge
                edge_data = {
                    "from": node_id,
                    "to": neighbor_id,
                    "relationship": "semantic_similarity",
                    "edge_type": "semantic",
                    "weight": 1.0 - node.stress,  # Higher weight for lower stress
                    "metadata": {
                        "discovered_by": "adaptive_isne",
                        "conveyance_similarity": abs(
                            node.document.conveyance.implementation_fidelity -
                            self.isne.nodes[neighbor_id].document.conveyance.implementation_fidelity
                        )
                    }
                }
                
                if self.storage.create_edge(edge_data):
                    edges_created += 1
        
        logger.info(f"Created {edges_created} semantic edges from ISNE neighborhoods")
        self.stats["relationships_created"] = edges_created
        
        # Store final ISNE positions
        self._store_isne_positions()
    
    def _store_isne_positions(self):
        """Store ISNE positions as metadata in documents."""
        if not self.isne:
            return
            
        positions = self.isne.get_positions()
        stress_map = self.isne.get_stress_map()
        
        for doc_id, position in positions.items():
            # Update document metadata with ISNE position
            metadata_update = {
                "isne_position": position.tolist(),
                "isne_stress": stress_map.get(doc_id, 0.0),
                "isne_last_update": self.stats["isne_updates"]
            }
            
            # This would update just the metadata field
            # For now, we'll log it
            logger.debug(f"ISNE position for {doc_id}: {position[:3]}... (stress: {stress_map.get(doc_id, 0):.3f})")
    
    def optimize_with_dspy(self, training_data: List[Dict[str, Any]]):
        """
        Optimize pipeline components using DSPy.
        
        This allows the pipeline to learn optimal strategies from data
        rather than using hand-crafted rules.
        """
        if not self.config.use_dspy_optimization:
            logger.info("DSPy optimization not enabled")
            return
        
        logger.info("Optimizing HERMES pipeline with DSPy")
        
        # Initialize optimizer
        optimizer = HERMESPipelineOptimizer()
        
        # Optimize metadata extraction
        if training_data:
            optimizer.optimize_metadata_extraction(
                training_data=training_data,
                budget="medium"
            )
        
        logger.info("DSPy optimization complete")
    
    def close(self):
        """Clean up resources."""
        # Final ISNE update and position storage
        if self.isne:
            logger.info("Performing final ISNE optimization...")
            self.isne.update_positions(iterations=20)
            self._store_isne_positions()
            
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        if hasattr(self.storage, 'close'):
            self.storage.close()


# Convenience functions

async def process_directory(
    directory_path: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Process a directory with HERMES pipeline.
    
    This is the main entry point for most users.
    """
    pipeline = HERMESPipeline(config=config)
    
    try:
        stats = await pipeline.process_directory(
            directory_path,
            progress_callback=progress_callback
        )
        return stats
        
    finally:
        pipeline.close()


def process_directory_sync(
    directory_path: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Synchronous wrapper for process_directory."""
    return asyncio.run(
        process_directory(directory_path, config, progress_callback)
    )