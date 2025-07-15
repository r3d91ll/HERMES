"""
Adaptive ISNE (Incremental Stochastic Neighborhood Embedding) with DSPy optimization.

This module implements a DSPy-enhanced version of ISNE that learns optimal
distance metrics, movement policies, and neighborhood structures from data
rather than using fixed rules.

The key insight: Let the data teach us how different types of documents
should cluster and move, rather than imposing rigid distance metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from sklearn.metrics import pairwise_distances
import dspy

from hermes.core.document_model import Document, DocumentType
from hermes.optimization.dspy_optimizers import HERMESPipelineOptimizer

logger = logging.getLogger(__name__)


# DSPy Signatures for ISNE optimization

class DistanceCalculation(dspy.Signature):
    """Calculate semantic distance between two documents."""
    doc1_type = dspy.InputField(desc="Type of first document")
    doc1_metadata = dspy.InputField(desc="Metadata of first document")
    doc2_type = dspy.InputField(desc="Type of second document") 
    doc2_metadata = dspy.InputField(desc="Metadata of second document")
    neighbor_context = dspy.InputField(desc="Types and distances of nearby documents")
    
    dimension_weights = dspy.OutputField(desc="JSON weights for WHERE, WHAT, CONVEYANCE dimensions")
    distance_metric = dspy.OutputField(desc="Metric to use: cosine, euclidean, or manhattan")
    confidence = dspy.OutputField(desc="Confidence in this distance calculation (0-1)")


class MovementPolicy(dspy.Signature):
    """Decide if and how much a node should move."""
    node_type = dspy.InputField(desc="Type of document")
    current_stress = dspy.InputField(desc="Current embedding stress/error")
    neighbor_distances = dspy.InputField(desc="Distances to k nearest neighbors")
    time_since_last_move = dspy.InputField(desc="Iterations since last position update")
    global_stability = dspy.InputField(desc="Overall graph stability metric")
    
    should_move = dspy.OutputField(desc="Whether node should reposition (true/false)")
    movement_scale = dspy.OutputField(desc="How much to move (0-1)")
    movement_reason = dspy.OutputField(desc="Explanation for movement decision")


class NeighborhoodStructure(dspy.Signature):
    """Determine optimal neighborhood size and structure."""
    node_type = dspy.InputField(desc="Type of document")
    node_metadata = dspy.InputField(desc="Document metadata")
    local_density = dspy.InputField(desc="Number of documents within standard distance")
    conveyance_level = dspy.InputField(desc="Document's conveyance score")
    
    k_neighbors = dspy.OutputField(desc="Optimal number of neighbors to consider")
    neighborhood_type = dspy.OutputField(desc="Type: fixed, adaptive, or hierarchical")
    inclusion_criteria = dspy.OutputField(desc="What makes a good neighbor for this node")


@dataclass
class ISNENode:
    """Node in the ISNE graph with position and metadata."""
    doc_id: str
    position: np.ndarray  # Current position in embedded space
    velocity: np.ndarray  # For momentum-based updates
    document: Document
    neighbors: Set[str]  # IDs of neighboring nodes
    stress: float = 0.0  # Local embedding stress
    last_move_iteration: int = 0
    is_anchor: bool = False  # Anchor nodes don't move


class AdaptiveISNE:
    """
    DSPy-enhanced ISNE for dynamic graph embedding.
    
    This combines traditional ISNE with learned policies for:
    - Distance calculation (context-aware metrics)
    - Movement decisions (when and how much to move)
    - Neighborhood structure (adaptive k-NN)
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,  # Lower dim for visualization/computation
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        min_gain: float = 0.01,
        use_dspy: bool = True
    ):
        """
        Initialize Adaptive ISNE.
        
        Args:
            embedding_dim: Dimensionality of ISNE embedding space
            learning_rate: Base learning rate for position updates
            momentum: Momentum factor for updates
            min_gain: Minimum gain for adaptive learning rate
            use_dspy: Whether to use DSPy optimization
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_gain = min_gain
        self.use_dspy = use_dspy
        
        # Node storage
        self.nodes: Dict[str, ISNENode] = {}
        self.iteration = 0
        
        # DSPy modules
        if use_dspy:
            self.distance_calculator = dspy.ChainOfThought(DistanceCalculation)
            self.movement_policy = dspy.ChainOfThought(MovementPolicy)
            self.neighborhood_structure = dspy.ChainOfThought(NeighborhoodStructure)
        
        # Learning history
        self.distance_history: List[Dict[str, Any]] = []
        self.movement_history: List[Dict[str, Any]] = []
    
    def add_document(self, document: Document, position: Optional[np.ndarray] = None):
        """
        Add a new document to the ISNE graph.
        
        Args:
            document: Document to add
            position: Initial position (random if not provided)
        """
        if position is None:
            # Initialize with random position
            position = np.random.randn(self.embedding_dim) * 0.0001
        
        node = ISNENode(
            doc_id=document.doc_id,
            position=position,
            velocity=np.zeros(self.embedding_dim),
            document=document,
            neighbors=set(),
            last_move_iteration=self.iteration
        )
        
        # Determine if this should be an anchor node
        # High conveyance + physical grounding = anchor
        if document.conveyance.implementation_fidelity > 0.8:
            node.is_anchor = True
            logger.info(f"Node {document.doc_id} set as anchor (high conveyance)")
        
        self.nodes[document.doc_id] = node
        
        # Update neighborhoods
        self._update_neighborhoods(document.doc_id)
    
    def remove_document(self, doc_id: str):
        """Remove a document from the graph."""
        if doc_id in self.nodes:
            # Remove from all neighbor lists
            for node in self.nodes.values():
                node.neighbors.discard(doc_id)
            
            # Remove node
            del self.nodes[doc_id]
            
            logger.info(f"Removed node {doc_id} from ISNE graph")
    
    def update_positions(self, iterations: int = 1):
        """
        Update node positions for given iterations.
        
        This is where DSPy optimization shines - learning when and how
        to move nodes based on their type and context.
        """
        for _ in range(iterations):
            self.iteration += 1
            
            # Calculate global stability
            global_stability = self._calculate_global_stability()
            
            # Update each non-anchor node
            for node_id, node in self.nodes.items():
                if node.is_anchor:
                    continue
                
                # Decide if node should move
                if self.use_dspy:
                    should_move, scale = self._dspy_movement_decision(
                        node, global_stability
                    )
                else:
                    should_move, scale = self._fixed_movement_decision(
                        node, global_stability
                    )
                
                if should_move:
                    # Calculate forces and update position
                    force = self._calculate_forces(node)
                    self._update_node_position(node, force, scale)
                    node.last_move_iteration = self.iteration
    
    def _calculate_forces(self, node: ISNENode) -> np.ndarray:
        """
        Calculate attractive and repulsive forces on a node.
        
        Uses DSPy-optimized distance calculations for context-aware forces.
        """
        forces = np.zeros(self.embedding_dim)
        
        # Attractive forces from neighbors
        for neighbor_id in node.neighbors:
            if neighbor_id not in self.nodes:
                continue
                
            neighbor = self.nodes[neighbor_id]
            
            # Calculate distance with DSPy optimization
            if self.use_dspy:
                distance, weights = self._dspy_distance(node, neighbor)
            else:
                distance, weights = self._fixed_distance(node, neighbor)
            
            # Apply attractive force
            direction = neighbor.position - node.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
            # Force proportional to distance and dimension weights
            attractive_force = direction * distance * 0.1
            forces += attractive_force
        
        # Repulsive forces from all nodes
        for other_id, other in self.nodes.items():
            if other_id == node.doc_id:
                continue
                
            direction = node.position - other.position
            dist = np.linalg.norm(direction)
            
            if dist > 0 and dist < 1.0:  # Only repel nearby nodes
                direction = direction / dist
                repulsive_force = direction * (1.0 / (dist + 0.1))
                forces += repulsive_force * 0.05
        
        return forces
    
    def _dspy_distance(
        self, 
        node1: ISNENode, 
        node2: ISNENode
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate distance using DSPy-optimized metric.
        
        Returns:
            Tuple of (distance, dimension_weights)
        """
        # Prepare context
        neighbor_context = self._get_neighbor_context(node1)
        
        # DSPy prediction
        prediction = self.distance_calculator(
            doc1_type=node1.document.doc_type.value,
            doc1_metadata=str(node1.document.metadata),
            doc2_type=node2.document.doc_type.value,
            doc2_metadata=str(node2.document.metadata),
            neighbor_context=str(neighbor_context)
        )
        
        # Parse weights
        try:
            import json
            weights = json.loads(prediction.dimension_weights)
        except:
            weights = {"where": 0.05, "what": 0.50, "conveyance": 0.45}
        
        # Calculate weighted distance
        distance = self._calculate_weighted_distance(
            node1.document,
            node2.document,
            weights,
            prediction.distance_metric
        )
        
        # Store for learning
        self.distance_history.append({
            "node1": node1.doc_id,
            "node2": node2.doc_id,
            "distance": distance,
            "weights": weights,
            "confidence": float(prediction.confidence)
        })
        
        return distance, weights
    
    def _fixed_distance(
        self,
        node1: ISNENode,
        node2: ISNENode
    ) -> Tuple[float, Dict[str, float]]:
        """Fallback fixed distance calculation."""
        weights = {"where": 0.05, "what": 0.50, "conveyance": 0.45}
        
        distance = self._calculate_weighted_distance(
            node1.document,
            node2.document,
            weights,
            "cosine"
        )
        
        return distance, weights
    
    def _calculate_weighted_distance(
        self,
        doc1: Document,
        doc2: Document,
        weights: Dict[str, float],
        metric: str = "cosine"
    ) -> float:
        """
        Calculate weighted distance between documents.
        
        Combines WHERE, WHAT, and CONVEYANCE dimensions with learned weights.
        """
        distances = {}
        
        # WHERE distance (structural/path similarity)
        if doc1.location and doc2.location:
            path1 = doc1.location.directory_chain
            path2 = doc2.location.directory_chain
            
            # Jaccard similarity of path components
            if path1 and path2:
                intersection = len(set(path1) & set(path2))
                union = len(set(path1) | set(path2))
                distances["where"] = 1.0 - (intersection / union if union > 0 else 0)
            else:
                distances["where"] = 1.0
        else:
            distances["where"] = 0.5  # Unknown
        
        # WHAT distance (semantic similarity)
        if (doc1.embeddings and doc1.embeddings.jina_semantic is not None and
            doc2.embeddings and doc2.embeddings.jina_semantic is not None):
            
            vec1 = doc1.embeddings.jina_semantic.reshape(1, -1)
            vec2 = doc2.embeddings.jina_semantic.reshape(1, -1)
            
            if metric == "cosine":
                dist = pairwise_distances(vec1, vec2, metric="cosine")[0, 0]
            elif metric == "euclidean":
                dist = pairwise_distances(vec1, vec2, metric="euclidean")[0, 0]
            else:  # manhattan
                dist = pairwise_distances(vec1, vec2, metric="manhattan")[0, 0]
                
            distances["what"] = dist
        else:
            distances["what"] = 0.5
        
        # CONVEYANCE distance (implementation quality difference)
        conv1 = doc1.conveyance.implementation_fidelity
        conv2 = doc2.conveyance.implementation_fidelity
        distances["conveyance"] = abs(conv1 - conv2)
        
        # Weighted combination
        total_distance = 0.0
        for dim, weight in weights.items():
            if dim in distances:
                total_distance += distances[dim] * weight
        
        return total_distance
    
    def _dspy_movement_decision(
        self,
        node: ISNENode,
        global_stability: float
    ) -> Tuple[bool, float]:
        """Use DSPy to decide if and how much a node should move."""
        # Get neighbor distances
        neighbor_distances = []
        for neighbor_id in node.neighbors:
            if neighbor_id in self.nodes:
                dist, _ = self._dspy_distance(node, self.nodes[neighbor_id])
                neighbor_distances.append(dist)
        
        # DSPy prediction
        prediction = self.movement_policy(
            node_type=node.document.doc_type.value,
            current_stress=str(node.stress),
            neighbor_distances=str(neighbor_distances),
            time_since_last_move=str(self.iteration - node.last_move_iteration),
            global_stability=str(global_stability)
        )
        
        # Parse results
        should_move = prediction.should_move.lower() == "true"
        try:
            scale = float(prediction.movement_scale)
        except:
            scale = 0.5
        
        # Store for learning
        self.movement_history.append({
            "node": node.doc_id,
            "moved": should_move,
            "scale": scale,
            "reason": prediction.movement_reason
        })
        
        return should_move, scale
    
    def _fixed_movement_decision(
        self,
        node: ISNENode,
        global_stability: float
    ) -> Tuple[bool, float]:
        """Fallback fixed movement decision."""
        # Always move if stress is high
        if node.stress > 0.1:
            return True, 0.5
        
        # Move occasionally even with low stress
        if self.iteration - node.last_move_iteration > 10:
            return True, 0.1
            
        return False, 0.0
    
    def _update_node_position(self, node: ISNENode, force: np.ndarray, scale: float):
        """Update node position with momentum."""
        # Update velocity with momentum
        node.velocity = self.momentum * node.velocity + self.learning_rate * scale * force
        
        # Update position
        node.position += node.velocity
        
        # Recalculate stress
        node.stress = np.linalg.norm(force)
    
    def _update_neighborhoods(self, node_id: str):
        """
        Update neighborhood structure for a node.
        
        Uses DSPy to determine optimal k and neighbor selection.
        """
        node = self.nodes[node_id]
        
        if self.use_dspy:
            # Get optimal k from DSPy
            k = self._dspy_optimal_k(node)
        else:
            k = min(10, len(self.nodes) - 1)  # Default k=10
        
        # Find k nearest neighbors
        distances = []
        for other_id, other in self.nodes.items():
            if other_id != node_id:
                dist, _ = (self._dspy_distance(node, other) if self.use_dspy 
                          else self._fixed_distance(node, other))
                distances.append((other_id, dist))
        
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        node.neighbors = {neighbor_id for neighbor_id, _ in distances[:k]}
        
        # Make neighborhoods symmetric
        for neighbor_id in node.neighbors:
            if neighbor_id in self.nodes:
                self.nodes[neighbor_id].neighbors.add(node_id)
    
    def _dspy_optimal_k(self, node: ISNENode) -> int:
        """Use DSPy to determine optimal neighborhood size."""
        # Calculate local density
        distances = []
        for other in self.nodes.values():
            if other.doc_id != node.doc_id:
                dist, _ = self._dspy_distance(node, other)
                distances.append(dist)
        
        if distances:
            median_dist = np.median(distances)
            local_density = sum(1 for d in distances if d < median_dist)
        else:
            local_density = 0
        
        # DSPy prediction
        prediction = self.neighborhood_structure(
            node_type=node.document.doc_type.value,
            node_metadata=str(node.document.metadata),
            local_density=str(local_density),
            conveyance_level=str(node.document.conveyance.implementation_fidelity)
        )
        
        # Parse k
        try:
            k = int(prediction.k_neighbors)
            k = max(3, min(k, len(self.nodes) - 1))  # Bounds
        except:
            k = 10  # Default
            
        return k
    
    def _calculate_global_stability(self) -> float:
        """Calculate overall graph stability metric."""
        if not self.nodes:
            return 1.0
            
        total_stress = sum(node.stress for node in self.nodes.values())
        avg_stress = total_stress / len(self.nodes)
        
        # Stability is inverse of average stress
        stability = 1.0 / (1.0 + avg_stress)
        return stability
    
    def _get_neighbor_context(self, node: ISNENode) -> List[Dict[str, Any]]:
        """Get context about neighboring nodes."""
        context = []
        
        for neighbor_id in list(node.neighbors)[:5]:  # Top 5 neighbors
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                dist, _ = self._fixed_distance(node, neighbor)  # Avoid recursion
                
                context.append({
                    "type": neighbor.document.doc_type.value,
                    "distance": dist,
                    "conveyance": neighbor.document.conveyance.implementation_fidelity
                })
        
        return context
    
    def get_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all nodes."""
        return {
            node_id: node.position.copy()
            for node_id, node in self.nodes.items()
        }
    
    def get_stress_map(self) -> Dict[str, float]:
        """Get stress values for all nodes."""
        return {
            node_id: node.stress
            for node_id, node in self.nodes.items()
        }
    
    def optimize_with_dspy(self, training_examples: List[Dict[str, Any]]):
        """
        Optimize ISNE policies using DSPy.
        
        Training examples should include:
        - Known good document clusters
        - Verified distance relationships
        - Successful movement patterns
        """
        logger.info("Optimizing Adaptive ISNE with DSPy")
        
        # This would train the DSPy modules on examples
        # Implementation depends on the specific optimization strategy
        pass