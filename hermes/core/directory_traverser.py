"""
Directory traversal and graph construction for HERMES.

This module implements a top-down directory traversal that builds the graph
database with directory structure as the foundational edges, aligning with
reconstruction theory's emphasis on physical reality.

Methodology:

1. Directory structure = WHERE dimension reality
   - It's the actual, physical organization
   - It exists independent of our interpretation
   - It provides natural hierarchical relationships

2. Parent-child directory relationships = First edges
   - These are "given" relationships, not inferred
   - They're unambiguous and stable
   - They form the skeleton of our graph

This directory-first approach ensures that our graph is grounded in physical
reality before any semantic interpretation is applied. By establishing the
structural foundation first, we create a stable base upon which semantic
relationships can be discovered rather than imposed.

For academic reference: This methodology aligns with Actor-Network Theory's
emphasis on following the actors (files/directories) and letting them define
their own relationships, rather than imposing predetermined categories.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Set
import logging
from datetime import datetime
from collections import deque

from hermes.core.base import BaseLoader, BaseStorage


logger = logging.getLogger(__name__)


class DirectoryTraverser:
    """
    Traverse directories top-down, building graph with directory structure first.
    
    Philosophy:
    - Directory structure represents physical reality (WHERE)
    - Parent-child relationships are primary edges
    - Files inherit context from their directory ancestry
    - Top-down processing preserves hierarchical context
    """
    
    def __init__(
        self,
        storage: BaseStorage,
        loaders: Dict[str, BaseLoader],
        respect_gitignore: bool = True,
        create_directory_nodes: bool = True,
        batch_size: int = 100
    ):
        """
        Initialize directory traverser.
        
        Args:
            storage: Storage backend for graph database
            loaders: Dict mapping file extensions to loaders
            respect_gitignore: Whether to respect .gitignore files
            create_directory_nodes: Create nodes for directories
            batch_size: Number of items to process before committing
        """
        self.storage = storage
        self.loaders = loaders
        self.respect_gitignore = respect_gitignore
        self.create_directory_nodes = create_directory_nodes
        self.batch_size = batch_size
        
        # Track processing state
        self.processed_paths: Set[Path] = set()
        self.directory_stack: List[Dict[str, Any]] = []
        self.gitignore_patterns: List[str] = []
        
    def traverse_and_build(
        self,
        root_path: Path,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Traverse directory tree and build graph database.
        
        The key insight: Build directory relationships FIRST, then populate
        with file content. This ensures the graph structure mirrors reality.
        
        Args:
            root_path: Root directory to traverse
            progress_callback: Optional callback for progress updates
            
        Returns:
            Statistics about the traversal
        """
        root_path = Path(root_path).resolve()
        stats = {
            "directories_processed": 0,
            "files_processed": 0,
            "edges_created": 0,
            "errors": 0,
            "start_time": datetime.now()
        }
        
        logger.info(f"Starting directory traversal from {root_path}")
        
        # Phase 1: Build directory structure in graph
        logger.info("Phase 1: Building directory structure")
        dir_nodes = self._build_directory_structure(root_path, stats)
        
        # Phase 2: Process files within structure
        logger.info("Phase 2: Processing files within structure")
        self._process_files_in_structure(root_path, dir_nodes, stats, progress_callback)
        
        stats["end_time"] = datetime.now()
        stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()
        
        logger.info(f"Traversal complete: {stats['directories_processed']} directories, "
                   f"{stats['files_processed']} files in {stats['duration']:.2f}s")
        
        return stats
    
    def _build_directory_structure(
        self,
        root_path: Path,
        stats: Dict[str, Any]
    ) -> Dict[Path, str]:
        """
        Phase 1: Build directory nodes and relationships.
        
        This establishes the foundational WHERE dimension of our graph.
        """
        dir_nodes = {}  # Path -> node_id mapping
        
        # Create root node
        root_node_id = self._create_directory_node(root_path, parent_id=None)
        dir_nodes[root_path] = root_node_id
        stats["directories_processed"] += 1
        
        # BFS traversal to maintain hierarchical order
        queue = deque([root_path])
        
        while queue:
            current_dir = queue.popleft()
            current_node_id = dir_nodes[current_dir]
            
            try:
                # List subdirectories
                for item in current_dir.iterdir():
                    if item.is_dir() and not self._should_ignore(item):
                        # Create directory node
                        dir_node_id = self._create_directory_node(
                            item, 
                            parent_id=current_node_id
                        )
                        dir_nodes[item] = dir_node_id
                        
                        # Create parent-child edge
                        self._create_directory_edge(
                            parent_id=current_node_id,
                            child_id=dir_node_id,
                            relationship="contains"
                        )
                        
                        stats["directories_processed"] += 1
                        stats["edges_created"] += 1
                        
                        # Add to queue for further traversal
                        queue.append(item)
                        
            except PermissionError as e:
                logger.warning(f"Permission denied: {current_dir}")
                stats["errors"] += 1
        
        return dir_nodes
    
    def _process_files_in_structure(
        self,
        root_path: Path,
        dir_nodes: Dict[Path, str],
        stats: Dict[str, Any],
        progress_callback: Optional[Callable]
    ):
        """
        Phase 2: Process files within established directory structure.
        
        Files inherit context from their directory ancestry.
        """
        total_files = sum(1 for _ in self._iter_files(root_path))
        processed = 0
        
        for file_path in self._iter_files(root_path):
            if self._should_ignore(file_path):
                continue
                
            # Find appropriate loader
            loader = self._get_loader(file_path)
            if not loader:
                logger.debug(f"No loader for {file_path}")
                continue
            
            try:
                # Load file content
                file_data = loader.load(file_path)
                
                # Create file node with directory context
                parent_dir = file_path.parent
                parent_node_id = dir_nodes.get(parent_dir)
                
                file_node_id = self._create_file_node(
                    file_path,
                    file_data,
                    parent_directory_id=parent_node_id
                )
                
                # Create directory-contains-file edge
                if parent_node_id:
                    self._create_directory_edge(
                        parent_id=parent_node_id,
                        child_id=file_node_id,
                        relationship="contains_file"
                    )
                    stats["edges_created"] += 1
                
                stats["files_processed"] += 1
                processed += 1
                
                if progress_callback and processed % 10 == 0:
                    progress_callback(str(file_path), processed, total_files)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats["errors"] += 1
    
    def _create_directory_node(
        self,
        path: Path,
        parent_id: Optional[str] = None
    ) -> str:
        """Create a directory node in the graph."""
        # Directory metadata that establishes WHERE
        metadata = {
            "node_type": "directory",
            "name": path.name,
            "absolute_path": str(path),
            "relative_path": str(path.relative_to(path.anchor)),
            "depth": len(path.parts) - 1,
            "parent_id": parent_id,
            "created_at": datetime.now().isoformat()
        }
        
        # Calculate directory-specific embeddings
        # This could include:
        # - Path structure encoding
        # - Directory name semantics
        # - Hierarchical position encoding
        
        node_id = f"dir:{path}"
        self.storage.store_document(node_id, metadata)
        
        return node_id
    
    def _create_file_node(
        self,
        path: Path,
        file_data: Dict[str, Any],
        parent_directory_id: Optional[str] = None
    ) -> str:
        """Create a file node with inherited directory context."""
        # Combine file data with directory context
        metadata = {
            "node_type": "file",
            "name": path.name,
            "absolute_path": str(path),
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
            "parent_directory_id": parent_directory_id,
            "directory_depth": len(path.parts) - 1,
            "created_at": datetime.now().isoformat()
        }
        
        # Merge with loader-provided data
        metadata.update(file_data.get("metadata", {}))
        
        # The key insight: Directory context influences embeddings
        # Files inherit semantic context from their location
        
        node_id = f"file:{path}"
        full_data = {
            **file_data,
            "metadata": metadata,
            "inherited_context": self._get_inherited_context(path, parent_directory_id)
        }
        
        self.storage.store_document(node_id, full_data)
        
        return node_id
    
    def _create_directory_edge(
        self,
        parent_id: str,
        child_id: str,
        relationship: str
    ):
        """Create an edge representing directory relationships."""
        edge_data = {
            "from": parent_id,
            "to": child_id,
            "relationship": relationship,
            "edge_type": "structural",  # These are structural, not semantic
            "weight": 1.0,  # Directory relationships are certain
            "created_at": datetime.now().isoformat()
        }
        
        self.storage.create_edge(edge_data)
    
    def _get_inherited_context(
        self,
        path: Path,
        parent_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Extract context inherited from directory hierarchy.
        
        This is crucial for reconstruction - files don't exist in isolation,
        they exist within a directory structure that provides context.
        """
        context = {
            "path_components": list(path.parts[:-1]),
            "directory_chain": [],
            "structural_depth": len(path.parts) - 1
        }
        
        # Traverse up the directory tree
        current = path.parent
        while current != current.parent:
            context["directory_chain"].append({
                "name": current.name,
                "level": len(current.parts) - 1
            })
            current = current.parent
        
        # Reverse to get top-down order
        context["directory_chain"].reverse()
        
        # Infer context from directory names
        # e.g., "test" → testing context, "docs" → documentation
        context["inferred_purpose"] = self._infer_purpose_from_path(path)
        
        return context
    
    def _infer_purpose_from_path(self, path: Path) -> List[str]:
        """Infer file purpose from its path components."""
        purposes = []
        path_lower = str(path).lower()
        
        # Common directory patterns
        patterns = {
            "test": ["testing", "validation"],
            "doc": ["documentation", "reference"],
            "src": ["implementation", "source_code"],
            "lib": ["library", "dependency"],
            "example": ["example", "demonstration"],
            "tutorial": ["tutorial", "learning"],
            "benchmark": ["performance", "benchmark"],
            "script": ["automation", "tooling"],
            "config": ["configuration", "settings"],
            "data": ["data", "dataset"],
        }
        
        for pattern, tags in patterns.items():
            if pattern in path_lower:
                purposes.extend(tags)
        
        return list(set(purposes))
    
    def _iter_files(self, root_path: Path):
        """Iterate through all files in directory tree."""
        for dirpath, _, filenames in os.walk(root_path):
            dir_path = Path(dirpath)
            if self._should_ignore(dir_path):
                continue
                
            for filename in filenames:
                file_path = dir_path / filename
                if not self._should_ignore(file_path):
                    yield file_path
    
    def _get_loader(self, path: Path) -> Optional[BaseLoader]:
        """Get appropriate loader for file type."""
        # Check each loader's can_load method
        for loader in self.loaders.values():
            if loader.can_load(path):
                return loader
        return None
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        # Common ignore patterns
        ignore_names = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
        
        # Check if any parent directory should be ignored
        for part in path.parts:
            if part in ignore_names:
                return True
        
        # TODO: Implement .gitignore parsing if respect_gitignore is True
        
        return False