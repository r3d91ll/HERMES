"""
Bridge Detection Algorithm for HADES

A "bridge" is a connection between theoretical knowledge and practical implementation.
The algorithm identifies these bridges by analyzing the multi-dimensional relationships
between documents, using the WHERE × WHAT × CONVEYANCE model.

Key insights:
1. Bridges manifest when all three dimensions are strong (multiplicative model)
2. Context amplifies bridge strength exponentially
3. Temporal patterns reveal how theory becomes practice
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document with multi-dimensional vectors."""
    doc_id: str
    content: str
    where_vector: np.ndarray  # 102 dimensions
    what_vector: np.ndarray   # 1024 dimensions (or embedding size)
    conveyance_vector: np.ndarray  # 922 dimensions
    metadata: Dict
    
    @property
    def is_theory(self) -> bool:
        """Check if document is primarily theoretical."""
        # High semantic content but lower actionability
        return self.metadata.get('conveyance_scores', {}).get('actionability', 0) < 0.3
    
    @property
    def is_practice(self) -> bool:
        """Check if document is primarily practical."""
        # High actionability and implementation fidelity
        scores = self.metadata.get('conveyance_scores', {})
        return (scores.get('actionability', 0) > 0.7 and 
                scores.get('implementation_fidelity', 0) > 0.6)


@dataclass
class Bridge:
    """A theory-practice bridge between documents."""
    theory_doc: Document
    practice_doc: Document
    
    # Dimensional similarities
    where_similarity: float
    what_similarity: float
    conveyance_similarity: float
    
    # Composite scores
    bridge_strength: float
    context_amplification: float
    
    # Evidence
    shared_concepts: List[str]
    transformation_path: List[str]
    confidence: float


class BridgeDetector:
    """
    Detects theory-practice bridges in document collections.
    
    The algorithm works by:
    1. Identifying theory and practice documents
    2. Computing multi-dimensional similarities
    3. Applying context amplification
    4. Validating bridges through evidence
    """
    
    def __init__(self, 
                 context_alpha: float = 1.7,
                 min_bridge_strength: float = 0.6):
        """
        Args:
            context_alpha: Exponential factor for context amplification
            min_bridge_strength: Minimum strength to consider a bridge
        """
        self.context_alpha = context_alpha
        self.min_bridge_strength = min_bridge_strength
        
    def detect_bridges(self, documents: List[Document]) -> List[Bridge]:
        """
        Find all theory-practice bridges in a document collection.
        
        This implements the core HADES hypothesis:
        - Theory papers have high WHAT but low CONVEYANCE
        - Practice papers have high CONVEYANCE
        - Bridges occur when WHERE and WHAT align between theory and practice
        """
        logger.info(f"Detecting bridges among {len(documents)} documents")
        
        # Separate theory and practice documents
        theory_docs = [d for d in documents if d.is_theory]
        practice_docs = [d for d in documents if d.is_practice]
        
        logger.info(f"Found {len(theory_docs)} theory and {len(practice_docs)} practice documents")
        
        bridges = []
        
        # Check all theory-practice pairs
        for theory in theory_docs:
            for practice in practice_docs:
                bridge = self._evaluate_bridge(theory, practice)
                if bridge and bridge.bridge_strength >= self.min_bridge_strength:
                    bridges.append(bridge)
        
        # Sort by strength
        bridges.sort(key=lambda b: b.bridge_strength, reverse=True)
        
        logger.info(f"Detected {len(bridges)} bridges above threshold {self.min_bridge_strength}")
        return bridges
    
    def _evaluate_bridge(self, theory: Document, practice: Document) -> Optional[Bridge]:
        """
        Evaluate if two documents form a theory-practice bridge.
        
        Implements: Bridge_Strength = WHERE × WHAT × CONVEYANCE × Context^α
        """
        # Calculate dimensional similarities
        where_sim = self._calculate_where_similarity(theory, practice)
        what_sim = cosine_similarity([theory.what_vector], [practice.what_vector])[0, 0]
        conv_sim = self._calculate_conveyance_complementarity(theory, practice)
        
        # Skip if any dimension is too weak (multiplicative model)
        if where_sim < 0.1 or what_sim < 0.1 or conv_sim < 0.1:
            return None
        
        # Calculate context amplification
        context_score = self._calculate_context(theory, practice)
        context_amp = context_score ** self.context_alpha
        
        # Calculate bridge strength
        base_strength = where_sim * what_sim * conv_sim
        bridge_strength = base_strength * context_amp
        
        # Extract evidence
        shared_concepts = self._extract_shared_concepts(theory, practice)
        transformation_path = self._trace_transformation_path(theory, practice)
        
        # Calculate confidence based on evidence quality
        confidence = self._calculate_confidence(
            shared_concepts, transformation_path, bridge_strength
        )
        
        return Bridge(
            theory_doc=theory,
            practice_doc=practice,
            where_similarity=where_sim,
            what_similarity=what_sim,
            conveyance_similarity=conv_sim,
            bridge_strength=bridge_strength,
            context_amplification=context_amp,
            shared_concepts=shared_concepts,
            transformation_path=transformation_path,
            confidence=confidence
        )
    
    def _calculate_where_similarity(self, doc1: Document, doc2: Document) -> float:
        """
        Calculate WHERE dimension similarity.
        
        This captures:
        - Filesystem proximity (are files organized together?)
        - Temporal proximity (published around same time?)
        - Structural similarity (similar depth in hierarchy?)
        """
        # Basic cosine similarity
        base_sim = cosine_similarity([doc1.where_vector], [doc2.where_vector])[0, 0]
        
        # Boost for temporal proximity
        time_boost = 0.0
        if 'published' in doc1.metadata and 'published' in doc2.metadata:
            year1 = int(doc1.metadata['published'][:4])
            year2 = int(doc2.metadata['published'][:4])
            years_apart = abs(year1 - year2)
            # Exponential decay: papers within 2 years get boost
            time_boost = 0.2 * np.exp(-years_apart / 2)
        
        # Boost for being in related directories
        path_boost = 0.0
        if 'path' in doc1.metadata and 'path' in doc2.metadata:
            path1_parts = doc1.metadata['path'].split('/')
            path2_parts = doc2.metadata['path'].split('/')
            # Check for shared parent directories
            shared_parents = len(set(path1_parts[:-1]) & set(path2_parts[:-1]))
            path_boost = 0.1 * min(shared_parents / 3, 1.0)
        
        return min(1.0, base_sim + time_boost + path_boost)
    
    def _calculate_conveyance_complementarity(self, theory: Document, practice: Document) -> float:
        """
        Calculate how well theory and practice complement each other.
        
        Good bridges have:
        - Theory with clear concepts but low actionability
        - Practice with high actionability that implements those concepts
        """
        theory_scores = theory.metadata.get('conveyance_scores', {})
        practice_scores = practice.metadata.get('conveyance_scores', {})
        
        # Theory should have low actionability but clear concepts
        theory_quality = (1 - theory_scores.get('actionability', 1)) * \
                        theory_scores.get('clarity', 0.5)
        
        # Practice should have high actionability and implementation
        practice_quality = practice_scores.get('actionability', 0) * \
                          practice_scores.get('implementation_fidelity', 0)
        
        # They complement if theory is theoretical and practice is practical
        complementarity = theory_quality * practice_quality
        
        # Also check if practice explicitly references theory concepts
        vector_similarity = cosine_similarity([theory.conveyance_vector], 
                                            [practice.conveyance_vector])[0, 0]
        
        return (complementarity + vector_similarity) / 2
    
    def _calculate_context(self, theory: Document, practice: Document) -> float:
        """
        Calculate shared context between documents.
        
        Context includes:
        - Shared citations
        - Common authors
        - Same conference/journal
        - Overlapping keywords
        """
        context_score = 0.0
        
        # Check for shared authors
        theory_authors = set(theory.metadata.get('authors', []))
        practice_authors = set(practice.metadata.get('authors', []))
        if theory_authors and practice_authors:
            author_overlap = len(theory_authors & practice_authors) / \
                           min(len(theory_authors), len(practice_authors))
            context_score += author_overlap * 0.3
        
        # Check for shared venue (conference/journal)
        if theory.metadata.get('venue') == practice.metadata.get('venue'):
            context_score += 0.2
        
        # Check for keyword overlap
        theory_keywords = set(theory.metadata.get('keywords', []))
        practice_keywords = set(practice.metadata.get('keywords', []))
        if theory_keywords and practice_keywords:
            keyword_overlap = len(theory_keywords & practice_keywords) / \
                            min(len(theory_keywords), len(practice_keywords))
            context_score += keyword_overlap * 0.3
        
        # Check for citation relationship (if available)
        theory_cites = set(theory.metadata.get('references', []))
        practice_cites = set(practice.metadata.get('references', []))
        
        # Does practice cite theory?
        if theory.doc_id in practice_cites:
            context_score += 0.5
        
        # Do they share citations?
        elif theory_cites and practice_cites:
            citation_overlap = len(theory_cites & practice_cites) / \
                             min(len(theory_cites), len(practice_cites))
            context_score += citation_overlap * 0.2
        
        return min(1.0, context_score)
    
    def _extract_shared_concepts(self, theory: Document, practice: Document) -> List[str]:
        """Extract concepts that appear in both documents."""
        # This would ideally use NER or concept extraction
        # For now, extract from metadata
        theory_concepts = theory.metadata.get('theory_components', [])
        practice_concepts = practice.metadata.get('practice_components', [])
        
        shared = []
        for t_concept in theory_concepts:
            for p_concept in practice_concepts:
                if self._concepts_match(t_concept, p_concept):
                    shared.append(f"{t_concept} → {p_concept}")
        
        return shared
    
    def _concepts_match(self, concept1: str, concept2: str) -> bool:
        """Check if two concepts are related."""
        # Simple string matching for now
        c1_lower = concept1.lower()
        c2_lower = concept2.lower()
        
        # Exact match
        if c1_lower == c2_lower:
            return True
        
        # Substring match
        if c1_lower in c2_lower or c2_lower in c1_lower:
            return True
        
        # Common algorithmic terms
        algo_mappings = {
            'pagerank': ['page rank', 'ranking', 'link analysis'],
            'attention': ['transformer', 'self-attention', 'attention mechanism'],
            'embedding': ['representation', 'vector', 'encoding'],
            'neural': ['network', 'deep learning', 'nn']
        }
        
        for key, values in algo_mappings.items():
            if key in c1_lower:
                return any(v in c2_lower for v in values)
            if key in c2_lower:
                return any(v in c1_lower for v in values)
        
        return False
    
    def _trace_transformation_path(self, theory: Document, practice: Document) -> List[str]:
        """
        Trace how theoretical concepts transform into practice.
        
        This identifies the key transformations:
        - Abstract algorithm → Concrete implementation
        - Mathematical formula → Code function
        - Theoretical property → Benchmark result
        """
        transformations = []
        
        # Check for algorithm to code transformation
        if 'algorithm' in theory.content.lower() and \
           any(kw in practice.content.lower() for kw in ['def ', 'function', 'class ']):
            transformations.append("Algorithm → Implementation")
        
        # Check for formula to computation
        if any(sym in theory.content for sym in ['∑', '∏', '∫', 'Σ']) and \
           any(kw in practice.content.lower() for kw in ['compute', 'calculate']):
            transformations.append("Formula → Computation")
        
        # Check for theory to benchmark
        if 'theorem' in theory.content.lower() and \
           any(kw in practice.content.lower() for kw in ['accuracy', 'performance', 'results']):
            transformations.append("Theory → Benchmark")
        
        # Check for concept to application
        theory_scores = theory.metadata.get('conveyance_scores', {})
        practice_scores = practice.metadata.get('conveyance_scores', {})
        if theory_scores.get('abstraction_level', 0) > 0.7 and \
           practice_scores.get('implementation_fidelity', 0) > 0.7:
            transformations.append("Abstract → Concrete")
        
        return transformations
    
    def _calculate_confidence(self, 
                            shared_concepts: List[str],
                            transformation_path: List[str],
                            bridge_strength: float) -> float:
        """Calculate confidence in the bridge detection."""
        confidence = bridge_strength  # Start with bridge strength
        
        # Boost for evidence
        concept_boost = min(0.2, len(shared_concepts) * 0.05)
        path_boost = min(0.2, len(transformation_path) * 0.05)
        
        confidence = min(1.0, confidence + concept_boost + path_boost)
        
        return confidence
    
    def create_bridge_graph(self, bridges: List[Bridge]) -> nx.DiGraph:
        """
        Create a directed graph of theory-practice bridges.
        
        This graph can be used for:
        - Visualizing knowledge flow
        - Finding multi-hop paths
        - Identifying bridge clusters
        """
        G = nx.DiGraph()
        
        # Add nodes
        all_docs = set()
        for bridge in bridges:
            all_docs.add(bridge.theory_doc)
            all_docs.add(bridge.practice_doc)
        
        for doc in all_docs:
            G.add_node(doc.doc_id, 
                      type='theory' if doc.is_theory else 'practice',
                      title=doc.metadata.get('title', doc.doc_id))
        
        # Add edges
        for bridge in bridges:
            G.add_edge(bridge.theory_doc.doc_id,
                      bridge.practice_doc.doc_id,
                      weight=bridge.bridge_strength,
                      where_sim=bridge.where_similarity,
                      what_sim=bridge.what_similarity,
                      conv_sim=bridge.conveyance_similarity,
                      shared_concepts=bridge.shared_concepts,
                      transformation=bridge.transformation_path)
        
        return G
    
    def find_bridge_chains(self, bridges: List[Bridge], max_length: int = 4) -> List[List[Bridge]]:
        """
        Find chains of bridges showing knowledge evolution.
        
        Example: Pure Math → Applied Math → Algorithm → Implementation
        """
        G = self.create_bridge_graph(bridges)
        chains = []
        
        # Find all simple paths up to max_length
        theory_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'theory']
        practice_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'practice']
        
        for theory in theory_nodes:
            for practice in practice_nodes:
                try:
                    paths = list(nx.all_simple_paths(G, theory, practice, cutoff=max_length))
                    for path in paths:
                        if len(path) > 2:  # Multi-hop paths
                            chain = []
                            for i in range(len(path) - 1):
                                # Find the bridge for this edge
                                for bridge in bridges:
                                    if (bridge.theory_doc.doc_id == path[i] and 
                                        bridge.practice_doc.doc_id == path[i+1]):
                                        chain.append(bridge)
                                        break
                            if len(chain) == len(path) - 1:
                                chains.append(chain)
                except nx.NetworkXNoPath:
                    continue
        
        return chains


def demo_bridge_detection():
    """Demonstrate bridge detection with example documents."""
    
    # Create example documents
    pagerank_paper = Document(
        doc_id="pagerank_1998",
        content="The PageRank citation ranking algorithm...",
        where_vector=np.random.rand(102),
        what_vector=np.random.rand(1024),
        conveyance_vector=np.random.rand(922),
        metadata={
            'title': 'The PageRank Citation Ranking',
            'published': '1998-01-29',
            'authors': ['Page', 'Brin'],
            'conveyance_scores': {
                'actionability': 0.2,  # Theory paper
                'clarity': 0.9,
                'abstraction_level': 0.8
            },
            'theory_components': ['link analysis', 'random walk', 'eigenvector']
        }
    )
    
    pagerank_impl = Document(
        doc_id="pagerank_impl_2001",
        content="def pagerank(graph, damping=0.85)...",
        where_vector=np.random.rand(102),
        what_vector=np.random.rand(1024),
        conveyance_vector=np.random.rand(922),
        metadata={
            'title': 'Efficient PageRank Implementation',
            'published': '2001-06-15',
            'authors': ['Developer'],
            'conveyance_scores': {
                'actionability': 0.9,  # Implementation
                'implementation_fidelity': 0.95,
                'clarity': 0.8
            },
            'practice_components': ['pagerank function', 'damping factor', 'power iteration'],
            'references': ['pagerank_1998']
        }
    )
    
    # Make vectors somewhat similar
    pagerank_impl.what_vector = pagerank_paper.what_vector + np.random.normal(0, 0.1, 1024)
    pagerank_impl.where_vector[0] = pagerank_paper.where_vector[0]  # Same domain
    
    # Detect bridge
    detector = BridgeDetector()
    documents = [pagerank_paper, pagerank_impl]
    bridges = detector.detect_bridges(documents)
    
    if bridges:
        bridge = bridges[0]
        print(f"Found bridge: {bridge.theory_doc.doc_id} → {bridge.practice_doc.doc_id}")
        print(f"Bridge strength: {bridge.bridge_strength:.3f}")
        print(f"Dimensional similarities: WHERE={bridge.where_similarity:.3f}, "
              f"WHAT={bridge.what_similarity:.3f}, CONV={bridge.conveyance_similarity:.3f}")
        print(f"Shared concepts: {bridge.shared_concepts}")
        print(f"Transformation: {bridge.transformation_path}")


if __name__ == "__main__":
    demo_bridge_detection()