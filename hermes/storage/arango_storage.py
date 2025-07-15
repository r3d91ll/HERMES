"""
ArangoDB storage backend for HERMES.

This module implements graph storage using ArangoDB's multi-model capabilities,
storing documents as nodes and relationships as edges while supporting
high-dimensional vector similarity search.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import logging
from pathlib import Path

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.collection import StandardCollection
from arango.exceptions import DocumentInsertError, ArangoError

from hermes.core.base import BaseStorage
from hermes.storage.vector_search import VectorSearchEngine


logger = logging.getLogger(__name__)


class ArangoStorage(BaseStorage):
    """
    ArangoDB storage implementation for HERMES.
    
    Features:
    - Document (node) storage with 2048-dimensional vectors
    - Edge storage for relationships
    - Efficient vector similarity search
    - Directory-first graph construction support
    - Batch operations for performance
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8529,
        username: str = "root",
        password: str = "",
        database: str = "hermes",
        nodes_collection: str = "nodes",
        edges_collection: str = "edges",
        queries_collection: str = "queries",  # Separate collection for queries
        create_collections: bool = True,
        vector_index_type: str = "hash",  # ArangoDB 3.12+ will have native vector indexes
    ):
        """
        Initialize ArangoDB storage.
        
        Args:
            host: ArangoDB host
            port: ArangoDB port
            username: Database username
            password: Database password
            database: Database name
            nodes_collection: Name for nodes collection
            edges_collection: Name for edges collection
            create_collections: Whether to create collections if missing
            vector_index_type: Type of index for vector search
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.nodes_collection_name = nodes_collection
        self.edges_collection_name = edges_collection
        self.queries_collection_name = queries_collection
        
        # Connect to ArangoDB
        self._connect()
        
        # Ensure collections exist
        if create_collections:
            self._ensure_collections()
        
        # Initialize vector search engine
        self.vector_search = VectorSearchEngine(
            index_path=Path(f"./indexes/{self.database_name}_vectors.idx"),
            dimension=2048,  # Full HADES dimension
            index_type="flat"  # Start with exact search
        )
    
    def _connect(self):
        """Establish connection to ArangoDB."""
        try:
            # Create client
            self.client = ArangoClient(hosts=f"http://{self.host}:{self.port}")
            
            # Connect to system database first
            sys_db = self.client.db("_system", username=self.username, password=self.password)
            
            # Create database if it doesn't exist
            if not sys_db.has_database(self.database_name):
                logger.info(f"Creating database '{self.database_name}'")
                sys_db.create_database(self.database_name)
            
            # Connect to our database
            self.db: StandardDatabase = self.client.db(
                self.database_name,
                username=self.username,
                password=self.password
            )
            
            logger.info(f"Connected to ArangoDB database '{self.database_name}'")
            
        except ArangoError as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            raise
    
    def _ensure_collections(self):
        """Ensure required collections exist with proper configuration."""
        # Create nodes collection (document collection)
        if not self.db.has_collection(self.nodes_collection_name):
            logger.info(f"Creating nodes collection '{self.nodes_collection_name}'")
            self.nodes_collection = self.db.create_collection(
                self.nodes_collection_name,
                edge=False,
                user_keys=True  # Allow custom _key values
            )
        else:
            self.nodes_collection = self.db.collection(self.nodes_collection_name)
        
        # Create edges collection (edge collection)
        if not self.db.has_collection(self.edges_collection_name):
            logger.info(f"Creating edges collection '{self.edges_collection_name}'")
            self.edges_collection = self.db.create_collection(
                self.edges_collection_name,
                edge=True,
                user_keys=True
            )
        else:
            self.edges_collection = self.db.collection(self.edges_collection_name)
        
        # Create queries collection (special toggleable collection)
        if not self.db.has_collection(self.queries_collection_name):
            logger.info(f"Creating queries collection '{self.queries_collection_name}'")
            self.queries_collection = self.db.create_collection(
                self.queries_collection_name,
                edge=False,
                user_keys=True
            )
        else:
            self.queries_collection = self.db.collection(self.queries_collection_name)
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for efficient querying."""
        # Index on node_type for filtering
        self.nodes_collection.add_hash_index(fields=["node_type"], unique=False)
        
        # Index on path for file/directory lookups
        self.nodes_collection.add_hash_index(fields=["metadata.absolute_path"], unique=True, sparse=True)
        
        # Index on timestamps
        self.nodes_collection.add_skiplist_index(fields=["created_at"], unique=False)
        
        # For edges: index on relationship type
        self.edges_collection.add_hash_index(fields=["relationship"], unique=False)
        
        logger.info("Created indexes for collections")
    
    def store_document(self, doc_id: str, data: Dict[str, Any]) -> bool:
        """
        Store a document (node) in the graph.
        
        Args:
            doc_id: Unique document identifier
            data: Document data including embeddings and metadata
            
        Returns:
            Success status
        """
        try:
            # Prepare document for ArangoDB
            doc = self._prepare_document(doc_id, data)
            
            # Insert or update
            if self.nodes_collection.has(doc["_key"]):
                # Update existing
                self.nodes_collection.update(doc)
                logger.debug(f"Updated document {doc_id}")
            else:
                # Insert new
                self.nodes_collection.insert(doc)
                logger.debug(f"Inserted document {doc_id}")
            
            # Add to vector index if embeddings present
            if "embeddings" in data:
                self._add_to_vector_index(doc_id, data["embeddings"])
            
            return True
            
        except ArangoError as e:
            logger.error(f"Failed to store document {doc_id}: {e}")
            return False
    
    def store_documents_batch(self, documents: List[Tuple[str, Dict[str, Any]]]) -> int:
        """
        Store multiple documents in batch for efficiency.
        
        Args:
            documents: List of (doc_id, data) tuples
            
        Returns:
            Number of successfully stored documents
        """
        prepared_docs = []
        for doc_id, data in documents:
            try:
                doc = self._prepare_document(doc_id, data)
                prepared_docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to prepare document {doc_id}: {e}")
        
        if not prepared_docs:
            return 0
        
        try:
            # Batch insert with overwrite mode
            result = self.nodes_collection.insert_many(
                prepared_docs,
                overwrite=True,  # Update if exists
                silent=False
            )
            
            # Count successes
            success_count = sum(1 for r in result if not isinstance(r, ArangoError))
            logger.info(f"Batch stored {success_count}/{len(prepared_docs)} documents")
            
            return success_count
            
        except ArangoError as e:
            logger.error(f"Batch storage failed: {e}")
            return 0
    
    def retrieve_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        try:
            key = self._make_key(doc_id)
            doc = self.nodes_collection.get(key)
            
            if doc:
                return self._restore_document(doc)
            return None
            
        except ArangoError as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Uses Faiss for efficient similarity search with fallback to AQL.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        # Try vector search engine first
        try:
            # Get similar document IDs from vector index
            similar_docs = self.vector_search.search(query_embedding, k)
            
            if similar_docs:
                # Retrieve full documents from ArangoDB
                results = []
                for doc_id, score in similar_docs:
                    doc = self.retrieve_document(doc_id)
                    if doc:
                        doc["_score"] = score
                        results.append(doc)
                
                return results
        
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to AQL: {e}")
        
        # Fallback to AQL-based search for small datasets
        return self._search_aql_fallback(query_embedding, k)
    
    def _search_aql_fallback(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Fallback search using AQL (for small datasets only)."""
        # This is inefficient but works for small datasets
        aql = """
        FOR doc IN @@collection
            FILTER doc.embeddings.jina_v4_semantic != null
            LIMIT @limit
            RETURN doc
        """
        
        try:
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "@collection": self.nodes_collection_name,
                    "limit": k * 10  # Get more to sort client-side
                }
            )
            
            # Calculate similarities client-side
            results = []
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            for doc in cursor:
                restored = self._restore_document(doc)
                
                # Get document embedding
                doc_embedding = restored.get("embeddings", {}).get("jina_v4_semantic")
                if doc_embedding is not None:
                    # Calculate cosine similarity
                    doc_norm = doc_embedding / np.linalg.norm(doc_embedding)
                    similarity = np.dot(query_norm, doc_norm)
                    
                    restored["_score"] = float(similarity)
                    results.append(restored)
            
            # Sort by score and limit
            results.sort(key=lambda x: x.get("_score", 0), reverse=True)
            return results[:k]
            
        except ArangoError as e:
            logger.error(f"AQL search failed: {e}")
            return []
    
    def create_edge(self, edge_data: Dict[str, Any]) -> bool:
        """
        Create an edge between two nodes.
        
        Args:
            edge_data: Edge data with 'from', 'to', and properties
            
        Returns:
            Success status
        """
        try:
            # Prepare edge document
            edge = {
                "_from": f"{self.nodes_collection_name}/{self._make_key(edge_data['from'])}",
                "_to": f"{self.nodes_collection_name}/{self._make_key(edge_data['to'])}",
                "relationship": edge_data.get("relationship", "related"),
                "edge_type": edge_data.get("edge_type", "semantic"),
                "weight": edge_data.get("weight", 1.0),
                "created_at": edge_data.get("created_at", datetime.now().isoformat()),
                "metadata": edge_data.get("metadata", {})
            }
            
            # Generate edge key
            edge_key = f"{self._make_key(edge_data['from'])}_{self._make_key(edge_data['to'])}"
            edge["_key"] = edge_key
            
            # Insert or update edge
            if self.edges_collection.has(edge_key):
                self.edges_collection.update(edge)
            else:
                self.edges_collection.insert(edge)
            
            return True
            
        except ArangoError as e:
            logger.error(f"Failed to create edge: {e}")
            return False
    
    def get_neighbors(self, doc_id: str, relationship: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all neighbors of a document.
        
        Args:
            doc_id: Document identifier
            relationship: Optional relationship type filter
            
        Returns:
            List of neighboring documents
        """
        key = self._make_key(doc_id)
        node_id = f"{self.nodes_collection_name}/{key}"
        
        # Build AQL query
        aql = """
        FOR v, e IN 1..1 ANY @node_id @@edge_collection
        """
        
        bind_vars = {
            "node_id": node_id,
            "@edge_collection": self.edges_collection_name
        }
        
        if relationship:
            aql += " FILTER e.relationship == @relationship"
            bind_vars["relationship"] = relationship
        
        aql += " RETURN {node: v, edge: e}"
        
        try:
            cursor = self.db.aql.execute(aql, bind_vars=bind_vars)
            
            neighbors = []
            for result in cursor:
                neighbor = self._restore_document(result["node"])
                neighbor["_edge"] = result["edge"]
                neighbors.append(neighbor)
            
            return neighbors
            
        except ArangoError as e:
            logger.error(f"Failed to get neighbors: {e}")
            return []
    
    def _prepare_document(self, doc_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare document for ArangoDB storage.
        
        Handles:
        - Converting numpy arrays to lists
        - Setting _key field
        - Adding timestamps
        """
        doc = {
            "_key": self._make_key(doc_id),
            "doc_id": doc_id,
            "created_at": datetime.now().isoformat()
        }
        
        # Handle embeddings
        if "embeddings" in data:
            embeddings = data["embeddings"]
            
            # Convert numpy arrays to lists for JSON storage
            if "hades_dimensional" in embeddings:
                for key, vector in embeddings["hades_dimensional"].items():
                    if isinstance(vector, np.ndarray):
                        embeddings["hades_dimensional"][key] = vector.tolist()
            
            if "jina_v4_semantic" in embeddings and isinstance(embeddings["jina_v4_semantic"], np.ndarray):
                embeddings["jina_v4_semantic"] = embeddings["jina_v4_semantic"].tolist()
            
            doc["embeddings"] = embeddings
        
        # Copy other fields
        for key, value in data.items():
            if key not in ["embeddings", "_key", "_id"]:
                doc[key] = value
        
        return doc
    
    def _restore_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore document from ArangoDB format.
        
        Converts lists back to numpy arrays for embeddings.
        """
        restored = doc.copy()
        
        # Restore embeddings as numpy arrays
        if "embeddings" in restored:
            embeddings = restored["embeddings"]
            
            if "hades_dimensional" in embeddings:
                for key, vector in embeddings["hades_dimensional"].items():
                    if isinstance(vector, list):
                        embeddings["hades_dimensional"][key] = np.array(vector)
            
            if "jina_v4_semantic" in embeddings and isinstance(embeddings["jina_v4_semantic"], list):
                embeddings["jina_v4_semantic"] = np.array(embeddings["jina_v4_semantic"])
        
        # Remove ArangoDB internal fields if requested
        for field in ["_key", "_id", "_rev"]:
            restored.pop(field, None)
        
        return restored
    
    def _make_key(self, doc_id: str) -> str:
        """
        Convert document ID to valid ArangoDB key.
        
        ArangoDB keys must match pattern: ^[a-zA-Z0-9_-]+$
        """
        # Replace invalid characters
        key = doc_id.replace("/", "_").replace(":", "_").replace(".", "_")
        key = key.replace(" ", "_").replace("\\", "_")
        
        # Ensure it starts with letter or underscore
        if key and key[0].isdigit():
            key = f"_{key}"
        
        return key
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            return {
                "database": self.database_name,
                "nodes_count": self.nodes_collection.count(),
                "edges_count": self.edges_collection.count(),
                "nodes_collection": self.nodes_collection_name,
                "edges_collection": self.edges_collection_name,
                "indexes": {
                    "nodes": len(self.nodes_collection.indexes()),
                    "edges": len(self.edges_collection.indexes())
                }
            }
        except ArangoError as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def clear_all(self):
        """Clear all data (use with caution!)."""
        try:
            self.nodes_collection.truncate()
            self.edges_collection.truncate()
            logger.warning("Cleared all data from storage")
        except ArangoError as e:
            logger.error(f"Failed to clear data: {e}")
    
    def close(self):
        """Close database connection."""
        # Save vector index
        self.vector_search.save_index()
        # ArangoDB client doesn't require explicit closing
        pass
    
    def _add_to_vector_index(self, doc_id: str, embeddings: Dict[str, Any]):
        """Add document embeddings to vector search index."""
        vectors_to_add = {}
        
        # Check for different embedding types
        if "jina_v4_semantic" in embeddings:
            vec = embeddings["jina_v4_semantic"]
            if isinstance(vec, list):
                vec = np.array(vec)
            vectors_to_add[doc_id] = vec
        
        # Could also add HADES full vector
        elif "hades_dimensional" in embeddings:
            dims = embeddings["hades_dimensional"]
            if all(k in dims for k in ["where_vector", "what_vector", "conveyance_vector"]):
                # Concatenate HADES dimensions
                where = dims["where_vector"]
                what = dims["what_vector"] 
                conv = dims["conveyance_vector"]
                
                # Convert lists to arrays if needed
                if isinstance(where, list):
                    where = np.array(where)
                if isinstance(what, list):
                    what = np.array(what)
                if isinstance(conv, list):
                    conv = np.array(conv)
                
                full_vec = np.concatenate([where, what, conv])
                vectors_to_add[doc_id] = full_vec
        
        if vectors_to_add:
            self.vector_search.add_vectors(vectors_to_add, rebuild=False)
    
    def store_query(self, query_id: str, query_data: Dict[str, Any]) -> bool:
        """
        Store a query document in the special queries collection.
        
        Args:
            query_id: Unique query identifier
            query_data: Query document data
            
        Returns:
            Success status
        """
        try:
            # Prepare query document
            doc = self._prepare_document(query_id, query_data)
            doc["_toggleable"] = True  # Mark as toggleable
            
            # Store in queries collection
            if self.queries_collection.has(doc["_key"]):
                self.queries_collection.update(doc)
                logger.debug(f"Updated query {query_id}")
            else:
                self.queries_collection.insert(doc)
                logger.debug(f"Inserted query {query_id}")
            
            # Also store in main nodes collection if active
            if query_data.get("is_active", True):
                self.store_document(query_id, query_data)
            
            return True
            
        except ArangoError as e:
            logger.error(f"Failed to store query {query_id}: {e}")
            return False
    
    def toggle_query_collection(self, collection_tags: List[str], active: bool) -> int:
        """
        Toggle visibility of queries with specific collection tags.
        
        Args:
            collection_tags: Tags identifying the collection
            active: Whether to make queries active or inactive
            
        Returns:
            Number of queries affected
        """
        try:
            # Find all queries with matching tags
            aql = """
            FOR query IN @@queries_collection
                FILTER query.metadata.collection_tags != null
                FILTER LENGTH(
                    FOR tag IN @tags
                        FILTER tag IN query.metadata.collection_tags
                        RETURN 1
                ) > 0
                RETURN query
            """
            
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "@queries_collection": self.queries_collection_name,
                    "tags": collection_tags
                }
            )
            
            affected_count = 0
            for query in cursor:
                query_id = query["doc_id"]
                
                if active and not query.get("is_active", True):
                    # Activate: add to main nodes collection
                    self.store_document(query_id, self._restore_document(query))
                    # Update query active status
                    self.queries_collection.update({"_key": query["_key"], "is_active": True})
                    affected_count += 1
                    
                elif not active and query.get("is_active", True):
                    # Deactivate: remove from main nodes collection
                    try:
                        self.nodes_collection.delete(self._make_key(query_id))
                    except:
                        pass  # May not exist in nodes
                    # Update query active status
                    self.queries_collection.update({"_key": query["_key"], "is_active": False})
                    affected_count += 1
            
            logger.info(f"Toggled {affected_count} queries to active={active}")
            return affected_count
            
        except ArangoError as e:
            logger.error(f"Failed to toggle query collection: {e}")
            return 0
    
    def get_query_history(self, researcher_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get query history, optionally filtered by researcher.
        
        Args:
            researcher_id: Optional researcher ID filter
            
        Returns:
            List of query documents with execution history
        """
        try:
            if researcher_id:
                aql = """
                FOR query IN @@queries_collection
                    FILTER query.query_context.researcher_id == @researcher_id
                    SORT query.created_at DESC
                    RETURN query
                """
                bind_vars = {
                    "@queries_collection": self.queries_collection_name,
                    "researcher_id": researcher_id
                }
            else:
                aql = """
                FOR query IN @@queries_collection
                    SORT query.created_at DESC
                    RETURN query
                """
                bind_vars = {"@queries_collection": self.queries_collection_name}
            
            cursor = self.db.aql.execute(aql, bind_vars=bind_vars)
            
            return [self._restore_document(doc) for doc in cursor]
            
        except ArangoError as e:
            logger.error(f"Failed to get query history: {e}")
            return []
    
    def analyze_conveyance_evolution(self, query_id: str) -> Dict[str, Any]:
        """
        Analyze how a query's results have evolved over time.
        
        Args:
            query_id: Query to analyze
            
        Returns:
            Evolution analysis including conveyance trends
        """
        try:
            # Get query document
            key = self._make_key(query_id)
            query = self.queries_collection.get(key)
            
            if not query or not query.get("execution_history"):
                return {"error": "No execution history found"}
            
            history = query["execution_history"]
            
            # Analyze trends
            conveyance_values = [h["avg_conveyance"] for h in history]
            result_counts = [h["total_results"] for h in history]
            
            # Calculate deltas
            if len(history) > 1:
                conveyance_delta = conveyance_values[-1] - conveyance_values[0]
                result_delta = result_counts[-1] - result_counts[0]
                
                # Find new high-conveyance results
                latest_results = set(history[-1]["result_doc_ids"])
                initial_results = set(history[0]["result_doc_ids"])
                new_results = latest_results - initial_results
                
                return {
                    "query_id": query_id,
                    "execution_count": len(history),
                    "initial_conveyance": conveyance_values[0],
                    "current_conveyance": conveyance_values[-1],
                    "conveyance_delta": conveyance_delta,
                    "conveyance_trend": "increasing" if conveyance_delta > 0 else "decreasing",
                    "result_count_delta": result_delta,
                    "new_high_value_results": len(new_results),
                    "peak_conveyance": max(conveyance_values),
                    "timestamps": [h.get("timestamp") for h in history]
                }
            else:
                return {
                    "query_id": query_id,
                    "execution_count": 1,
                    "message": "Need multiple executions for evolution analysis"
                }
            
        except ArangoError as e:
            logger.error(f"Failed to analyze conveyance evolution: {e}")
            return {"error": str(e)}