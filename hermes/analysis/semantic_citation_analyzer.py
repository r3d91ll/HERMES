#!/usr/bin/env python3
"""
Semantic Citation Analyzer - Measures true conveyance through citation context.
Analyzes not just that A cites B, but HOW A builds on B semantically.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field
import re
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CitationContext:
    """A citation with its surrounding context and semantic analysis."""
    citing_paper_id: str
    cited_paper_id: str
    citation_text: str  # The sentence with citation
    context_before: str  # Paragraph before
    context_after: str  # Paragraph after
    
    # Semantic analysis
    semantic_similarity: float = 0.0  # How similar is context to cited paper
    citation_type: str = ""  # 'builds_on', 'mentions', 'critiques', etc.
    influence_score: float = 0.0  # Combined metric
    
    # Extracted features
    action_verbs: List[str] = field(default_factory=list)
    is_methodological: bool = False
    is_foundational: bool = False
    section_type: str = ""  # 'introduction', 'methods', 'related_work'


@dataclass
class SemanticCitation:
    """Enhanced citation with semantic influence measurement."""
    citation_id: str
    citing_paper: Dict
    cited_paper: Dict
    
    # Context windows
    contexts: List[CitationContext] = field(default_factory=list)
    
    # Aggregate metrics
    total_influence: float = 0.0
    semantic_similarity: float = 0.0
    conveyance_type: str = ""  # 'implementation', 'extension', 'application', etc.
    
    # Semantic features
    shared_concepts: List[str] = field(default_factory=list)
    concept_evolution: Dict[str, str] = field(default_factory=dict)  # concept -> how it evolved


class SemanticCitationAnalyzer:
    """
    Analyze HOW papers cite each other, not just that they do.
    Measures semantic similarity between citation context and cited work.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model."""
        self.encoder = SentenceTransformer(model_name)
        
        # Citation type patterns
        self.citation_patterns = {
            "builds_on": [
                r"we (extend|build on|improve|enhance)",
                r"based on .{0,20} we",
                r"following .{0,20} we",
                r"inspired by",
                r"adapting .{0,20} approach"
            ],
            "implements": [
                r"we (implement|use|apply|employ)",
                r"using .{0,20} method",
                r"following .{0,20} algorithm",
                r"as described in"
            ],
            "replaces": [
                r"instead of",
                r"rather than",
                r"replaces",
                r"supersedes",
                r"improves upon",
                r"better than"
            ],
            "negates": [
                r"contrary to",
                r"does not .{0,20} as claimed",
                r"incorrect",
                r"we show that .{0,20} not",
                r"disproves",
                r"contradicts"
            ],
            "redefines": [
                r"we redefine",
                r"means .{0,20} not",
                r"should be understood as",
                r"we propose .{0,20} instead",
                r"reinterpret"
            ],
            "compares": [
                r"compared to",
                r"outperforms",
                r"in contrast to",
                r"unlike",
                r"whereas"
            ],
            "mentions": [
                r"see also",
                r"for example",
                r"such as",
                r"including",
                r"e\.g\."
            ],
            "critiques": [
                r"however",
                r"limitation",
                r"fails to",
                r"does not",
                r"problem with"
            ]
        }
        
        # High conveyance action verbs
        self.action_verbs = {
            "high_conveyance": [
                "implement", "extend", "build", "adapt", "improve",
                "enhance", "apply", "utilize", "incorporate", "integrate"
            ],
            "medium_conveyance": [
                "follow", "use", "employ", "consider", "examine",
                "analyze", "evaluate", "test", "verify"
            ],
            "low_conveyance": [
                "mention", "cite", "reference", "note", "acknowledge",
                "list", "include", "review"
            ]
        }
        
    def extract_citation_contexts(self, paper_text: str, citations: List[str]) -> List[CitationContext]:
        """Extract citation contexts from paper text."""
        contexts = []
        
        # Split into paragraphs
        paragraphs = paper_text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            # Look for citations in this paragraph
            for citation in citations:
                citation_pattern = re.escape(citation)
                
                if re.search(citation_pattern, paragraph):
                    # Get surrounding context
                    context_before = paragraphs[i-1] if i > 0 else ""
                    context_after = paragraphs[i+1] if i < len(paragraphs)-1 else ""
                    
                    # Extract sentence with citation
                    sentences = re.split(r'[.!?]+', paragraph)
                    citation_sentence = ""
                    
                    for sent in sentences:
                        if citation in sent:
                            citation_sentence = sent.strip()
                            break
                            
                    context = CitationContext(
                        citing_paper_id="",  # To be filled
                        cited_paper_id="",   # To be filled
                        citation_text=citation_sentence,
                        context_before=context_before,
                        context_after=context_after
                    )
                    
                    # Analyze context
                    self._analyze_citation_context(context)
                    contexts.append(context)
                    
        return contexts
    
    def _analyze_citation_context(self, context: CitationContext):
        """Analyze a single citation context."""
        text = context.citation_text.lower()
        
        # Determine citation type
        for ctype, patterns in self.citation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    context.citation_type = ctype
                    break
                    
        # Extract action verbs
        for category, verbs in self.action_verbs.items():
            for verb in verbs:
                if verb in text:
                    context.action_verbs.append(verb)
                    
        # Check if methodological
        method_keywords = ["method", "algorithm", "approach", "technique", "procedure"]
        context.is_methodological = any(kw in text for kw in method_keywords)
        
        # Check if foundational
        foundation_keywords = ["based on", "builds on", "extends", "foundation", "fundamental"]
        context.is_foundational = any(kw in text for kw in foundation_keywords)
        
        # Guess section type
        if any(kw in context.context_before.lower() for kw in ["related work", "background", "prior work"]):
            context.section_type = "related_work"
        elif any(kw in context.context_before.lower() for kw in ["method", "approach", "algorithm"]):
            context.section_type = "methods"
        elif any(kw in context.context_before.lower() for kw in ["introduction", "we propose", "in this"]):
            context.section_type = "introduction"
        else:
            context.section_type = "other"
    
    def measure_semantic_influence(self, 
                                 citing_paper: Dict,
                                 cited_paper: Dict,
                                 contexts: List[CitationContext]) -> SemanticCitation:
        """Measure semantic influence between papers through citation contexts."""
        
        # Get paper abstracts for overall similarity
        citing_abstract = citing_paper.get("abstract", "")
        cited_abstract = cited_paper.get("abstract", "")
        
        # Encode abstracts
        citing_embedding = self.encoder.encode(citing_abstract)
        cited_embedding = self.encoder.encode(cited_abstract)
        
        # Overall semantic similarity
        abstract_similarity = cosine_similarity(
            citing_embedding.reshape(1, -1),
            cited_embedding.reshape(1, -1)
        )[0][0]
        
        # Analyze each context
        context_influences = []
        
        for context in contexts:
            # Combine context window
            full_context = f"{context.context_before} {context.citation_text} {context.context_after}"
            
            # Encode context
            context_embedding = self.encoder.encode(full_context)
            
            # Similarity between context and cited paper
            context_similarity = cosine_similarity(
                context_embedding.reshape(1, -1),
                cited_embedding.reshape(1, -1)
            )[0][0]
            
            context.semantic_similarity = float(context_similarity)
            
            # Calculate influence score
            influence = self._calculate_influence_score(context)
            context.influence_score = influence
            context_influences.append(influence)
            
        # Create semantic citation
        semantic_citation = SemanticCitation(
            citation_id=f"{citing_paper['id']}_{cited_paper['id']}",
            citing_paper=citing_paper,
            cited_paper=cited_paper,
            contexts=contexts,
            semantic_similarity=float(abstract_similarity)
        )
        
        # Calculate total influence
        if context_influences:
            semantic_citation.total_influence = sum(context_influences) / len(context_influences)
            
        # Determine conveyance type
        semantic_citation.conveyance_type = self._determine_conveyance_type(contexts)
        
        # Extract shared concepts
        semantic_citation.shared_concepts = self._extract_shared_concepts(
            citing_abstract, cited_abstract
        )
        
        return semantic_citation
    
    def _calculate_influence_score(self, context: CitationContext) -> float:
        """Calculate influence score for a citation context."""
        score = 0.0
        
        # Base score from semantic similarity
        score += context.semantic_similarity * 0.4
        
        # Citation type weights
        type_weights = {
            "builds_on": 0.9,      # High positive conveyance
            "implements": 0.8,     # Direct application
            "replaces": 0.7,       # Supersedes previous work
            "redefines": 0.6,      # Semantic evolution
            "compares": 0.5,       # Moderate conveyance
            "critiques": 0.3,      # Low positive (still engaging)
            "negates": 0.4,        # Negative but engaged
            "mentions": 0.2        # Minimal conveyance
        }
        score += type_weights.get(context.citation_type, 0.1) * 0.3
        
        # Action verb score
        if any(verb in context.action_verbs for verb in self.action_verbs["high_conveyance"]):
            score += 0.2
        elif any(verb in context.action_verbs for verb in self.action_verbs["medium_conveyance"]):
            score += 0.1
            
        # Methodological citation bonus
        if context.is_methodological:
            score += 0.1
            
        # Foundational citation bonus
        if context.is_foundational:
            score += 0.15
            
        # Section type weights
        section_weights = {
            "methods": 0.8,
            "introduction": 0.6,
            "related_work": 0.3,
            "other": 0.4
        }
        section_multiplier = section_weights.get(context.section_type, 0.5)
        
        return min(1.0, score * section_multiplier)
    
    def _determine_conveyance_type(self, contexts: List[CitationContext]) -> str:
        """Determine overall conveyance type from contexts."""
        if not contexts:
            return "unknown"
            
        # Count citation types
        type_counts = {}
        for context in contexts:
            ctype = context.citation_type
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            
        # Determine dominant type
        if type_counts.get("implements", 0) > 0:
            return "implementation"
        elif type_counts.get("builds_on", 0) > len(contexts) / 2:
            return "extension"
        elif type_counts.get("compares", 0) > len(contexts) / 2:
            return "comparison"
        elif any(c.is_methodological for c in contexts):
            return "methodological"
        else:
            return "reference"
    
    def _extract_shared_concepts(self, text1: str, text2: str) -> List[str]:
        """Extract concepts shared between two texts."""
        # Simple keyword extraction - in production, use better NLP
        keywords1 = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text1))
        keywords2 = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text2))
        
        shared = keywords1.intersection(keywords2)
        
        # Filter out common words
        common_words = {"The", "This", "These", "That", "Those", "We", "Our", "In", "For"}
        shared = [w for w in shared if w not in common_words]
        
        return sorted(shared)[:10]  # Top 10 shared concepts
    
    def analyze_citation_network_semantically(self, 
                                            papers: List[Dict],
                                            citations: List[Tuple[str, str]]) -> Dict[str, SemanticCitation]:
        """Analyze entire citation network with semantic context."""
        semantic_citations = {}
        
        # Create paper lookup
        paper_lookup = {p["id"]: p for p in papers}
        
        for citing_id, cited_id in citations:
            if citing_id not in paper_lookup or cited_id not in paper_lookup:
                continue
                
            citing_paper = paper_lookup[citing_id]
            cited_paper = paper_lookup[cited_id]
            
            # Extract contexts (would need full text in practice)
            # For now, simulate with abstract
            mock_context = CitationContext(
                citing_paper_id=citing_id,
                cited_paper_id=cited_id,
                citation_text=f"We build on {cited_paper['title']}",
                context_before=citing_paper.get("abstract", "")[:200],
                context_after=citing_paper.get("abstract", "")[200:400]
            )
            self._analyze_citation_context(mock_context)
            
            # Measure influence
            semantic_citation = self.measure_semantic_influence(
                citing_paper,
                cited_paper,
                [mock_context]
            )
            
            semantic_citations[semantic_citation.citation_id] = semantic_citation
            
        return semantic_citations
    
    def find_high_conveyance_citations(self, 
                                     semantic_citations: Dict[str, SemanticCitation],
                                     min_influence: float = 0.7) -> List[SemanticCitation]:
        """Find citations with high semantic influence."""
        high_conveyance = [
            citation for citation in semantic_citations.values()
            if citation.total_influence >= min_influence
        ]
        
        # Sort by influence
        high_conveyance.sort(key=lambda x: x.total_influence, reverse=True)
        
        return high_conveyance
    
    def trace_concept_evolution(self, 
                              semantic_citations: Dict[str, SemanticCitation],
                              concept: str) -> List[Dict]:
        """Trace how a concept evolves through citations."""
        evolution_path = []
        
        for citation in semantic_citations.values():
            if concept in citation.shared_concepts:
                # Look for how the concept is used differently
                citing_usage = self._extract_concept_usage(
                    citation.citing_paper["abstract"], concept
                )
                cited_usage = self._extract_concept_usage(
                    citation.cited_paper["abstract"], concept
                )
                
                if citing_usage != cited_usage:
                    evolution_path.append({
                        "from_paper": citation.cited_paper["title"],
                        "to_paper": citation.citing_paper["title"],
                        "original_usage": cited_usage,
                        "evolved_usage": citing_usage,
                        "influence": citation.total_influence
                    })
                    
        # Sort by influence
        evolution_path.sort(key=lambda x: x["influence"], reverse=True)
        
        return evolution_path
    
    def _extract_concept_usage(self, text: str, concept: str) -> str:
        """Extract how a concept is used in text."""
        # Find sentences containing the concept
        sentences = re.split(r'[.!?]+', text)
        
        for sent in sentences:
            if concept.lower() in sent.lower():
                # Extract key phrases around concept
                words = sent.split()
                concept_idx = -1
                
                for i, word in enumerate(words):
                    if concept.lower() in word.lower():
                        concept_idx = i
                        break
                        
                if concept_idx >= 0:
                    # Get surrounding words
                    start = max(0, concept_idx - 3)
                    end = min(len(words), concept_idx + 4)
                    context_phrase = " ".join(words[start:end])
                    return context_phrase
                    
        return ""
    
    def generate_semantic_citation_report(self,
                                        semantic_citations: Dict[str, SemanticCitation],
                                        output_file: str = "semantic_citation_analysis.md"):
        """Generate report on semantic citation patterns."""
        
        report = f"""# Semantic Citation Analysis: How Ideas Actually Spread

Generated: {datetime.now().isoformat()}

## Overview

Analyzed {len(semantic_citations)} citations for semantic influence beyond simple citation counts.

## High-Conveyance Citations

Citations where the citing paper deeply builds on the cited work:

"""
        
        high_conveyance = self.find_high_conveyance_citations(semantic_citations)
        
        for i, citation in enumerate(high_conveyance[:20], 1):
            report += f"""{i}. **{citation.citing_paper['title']}** 
   → builds on → **{citation.cited_paper['title']}**
   
   - Total Influence: {citation.total_influence:.3f}
   - Semantic Similarity: {citation.semantic_similarity:.3f}
   - Conveyance Type: {citation.conveyance_type}
   - Shared Concepts: {', '.join(citation.shared_concepts[:5])}

"""
            
            # Show best context
            if citation.contexts:
                best_context = max(citation.contexts, key=lambda c: c.influence_score)
                report += f"""   **Key Citation Context**: "{best_context.citation_text}"
   - Context Influence: {best_context.influence_score:.3f}
   - Citation Type: {best_context.citation_type}
   - Action Verbs: {', '.join(best_context.action_verbs)}

"""
        
        # Conveyance type distribution
        report += "\n## Conveyance Type Distribution\n\n"
        
        type_counts = {}
        for citation in semantic_citations.values():
            ctype = citation.conveyance_type
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            
        for ctype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(semantic_citations)) * 100
            report += f"- **{ctype}**: {count} ({percentage:.1f}%)\n"
            
        # Concept evolution
        report += "\n## Concept Evolution Examples\n\n"
        
        # Track some key concepts
        key_concepts = ["Transformer", "Attention", "BERT", "Embedding"]
        
        for concept in key_concepts:
            evolution = self.trace_concept_evolution(semantic_citations, concept)
            if evolution:
                report += f"### {concept} Evolution\n\n"
                for evo in evolution[:3]:
                    report += f"- **{evo['from_paper'][:50]}...** → **{evo['to_paper'][:50]}...**\n"
                    report += f"  - Original: \"{evo['original_usage']}\"\n"
                    report += f"  - Evolved: \"{evo['evolved_usage']}\"\n"
                    report += f"  - Influence: {evo['influence']:.3f}\n\n"
                    
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"Semantic citation report saved to {output_file}")


def main():
    """Demo semantic citation analysis."""
    analyzer = SemanticCitationAnalyzer()
    
    print("""
Semantic Citation Analyzer Ready!

This measures TRUE conveyance by analyzing:

1. Citation Context Similarity
   - Not just "A cites B"
   - But "How similar is the text around the citation to B's content?"
   
2. Citation Types
   - "builds_on" → High conveyance (0.9)
   - "implements" → High conveyance (0.8)
   - "mentions" → Low conveyance (0.2)
   
3. Action Verbs
   - "We extend..." → High conveyance
   - "See also..." → Low conveyance
   
4. Concept Evolution
   - How "Transformer" in 2017 becomes "BERT" in 2018
   - Semantic drift through citation chains

This proves conveyance isn't just citation count, but semantic influence!
""")


if __name__ == "__main__":
    main()