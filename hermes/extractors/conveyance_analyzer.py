"""
Conveyance Analyzer for HERMES pipeline.
Measures the actionability and implementation potential of content.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import re
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ConveyanceMetrics:
    """Metrics for measuring content conveyance."""
    implementation_fidelity: float  # How clearly implementable
    actionability: float           # How directly applicable
    bridge_potential: float        # Theory to practice translation
    has_algorithms: bool
    has_equations: bool
    has_examples: bool
    has_code: bool
    has_procedures: bool
    complexity_score: float
    raw_scores: Dict[str, float]


class ConveyanceAnalyzer:
    """
    Analyze content for its conveyance potential.
    
    CONVEYANCE measures how readily knowledge becomes executable,
    not whether you can access the file.
    """
    
    def __init__(self):
        """Initialize the conveyance analyzer."""
        self.algorithm_patterns = [
            r'algorithm\s+\d+',
            r'procedure\s+\d+',
            r'step\s+\d+:',
            r'^\d+\.\s+',  # Numbered steps
            r'function\s+\w+\s*\(',
            r'def\s+\w+\s*\(',
            r'algorithm:',
            r'pseudocode:'
        ]
        
        self.equation_patterns = [
            r'[A-Za-z]+\s*=\s*[^=]',  # Simple equations
            r'\\[A-Za-z]+{',  # LaTeX commands
            r'\$[^$]+\$',     # Inline math
            r'∑|∏|∫|∂|∇|∆',   # Math symbols
            r'\b(?:sum|product|integral)\b',
            r'Σ|Π|√'          # Greek letters
        ]
        
        self.example_patterns = [
            r'for example',
            r'e\.g\.',
            r'such as',
            r'example \d+:',
            r'consider the',
            r'suppose that',
            r'let us',
            r'imagine'
        ]
        
        self.procedure_patterns = [
            r'first,',
            r'then,',
            r'finally,',
            r'step by step',
            r'follow these',
            r'to do this',
            r'begin by',
            r'next,'
        ]
    
    def analyze(self, content: str, metadata: Dict[str, Any] = None) -> ConveyanceMetrics:
        """
        Analyze content for conveyance metrics.
        
        Args:
            content: Text content to analyze
            metadata: Optional metadata about the document
            
        Returns:
            ConveyanceMetrics with scores and indicators
        """
        if not content:
            return self._empty_metrics()
        
        # Detect various actionability indicators
        has_algorithms = self._detect_algorithms(content)
        has_equations = self._detect_equations(content)
        has_examples = self._detect_examples(content)
        has_code = self._detect_code(content, metadata)
        has_procedures = self._detect_procedures(content)
        
        # Calculate component scores
        algorithm_score = self._score_algorithms(content, has_algorithms)
        equation_score = self._score_equations(content, has_equations)
        example_score = self._score_examples(content, has_examples)
        code_score = self._score_code(content, has_code, metadata)
        procedure_score = self._score_procedures(content, has_procedures)
        structure_score = self._score_structure(content, metadata)
        
        # Calculate complexity
        complexity = self._calculate_complexity(content, metadata)
        
        # Calculate main metrics
        implementation_fidelity = self._calculate_implementation_fidelity(
            algorithm_score, equation_score, code_score, procedure_score, complexity
        )
        
        actionability = self._calculate_actionability(
            code_score, example_score, procedure_score, structure_score
        )
        
        bridge_potential = self._calculate_bridge_potential(
            has_algorithms, has_equations, has_examples, has_code, 
            algorithm_score, code_score
        )
        
        return ConveyanceMetrics(
            implementation_fidelity=implementation_fidelity,
            actionability=actionability,
            bridge_potential=bridge_potential,
            has_algorithms=has_algorithms,
            has_equations=has_equations,
            has_examples=has_examples,
            has_code=has_code,
            has_procedures=has_procedures,
            complexity_score=complexity,
            raw_scores={
                'algorithm_score': algorithm_score,
                'equation_score': equation_score,
                'example_score': example_score,
                'code_score': code_score,
                'procedure_score': procedure_score,
                'structure_score': structure_score
            }
        )
    
    def _detect_algorithms(self, content: str) -> bool:
        """Detect presence of algorithms."""
        lower_content = content.lower()
        
        # Check patterns
        for pattern in self.algorithm_patterns:
            if re.search(pattern, lower_content, re.MULTILINE | re.IGNORECASE):
                return True
        
        # Check keywords
        algorithm_keywords = ['algorithm', 'procedure', 'method', 'technique', 'approach']
        return any(keyword in lower_content for keyword in algorithm_keywords)
    
    def _detect_equations(self, content: str) -> bool:
        """Detect presence of equations."""
        for pattern in self.equation_patterns:
            if re.search(pattern, content):
                return True
        
        # Check for common equation indicators
        equation_indicators = ['equation', 'formula', 'theorem', 'lemma', 'proof']
        lower_content = content.lower()
        return any(indicator in lower_content for indicator in equation_indicators)
    
    def _detect_examples(self, content: str) -> bool:
        """Detect presence of examples."""
        lower_content = content.lower()
        
        for pattern in self.example_patterns:
            if re.search(pattern, lower_content, re.IGNORECASE):
                return True
        
        return 'example' in lower_content
    
    def _detect_code(self, content: str, metadata: Dict = None) -> bool:
        """Detect presence of code."""
        # Check metadata first (from loaders)
        if metadata:
            if metadata.get('code_blocks'):
                return True
            if metadata.get('loader') == 'PythonASTLoader':
                return True
        
        # Check for code patterns
        code_patterns = [
            r'```[\w]*\n',  # Code fence
            r'^\s{4,}\S',   # Indented code
            r'def\s+\w+\s*\(',
            r'function\s+\w+\s*\(',
            r'class\s+\w+',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'if\s+__name__\s*==',
            r'for\s+\w+\s+in\s+',
            r'while\s+.*:',
            r'return\s+\w+'
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        
        return False
    
    def _detect_procedures(self, content: str) -> bool:
        """Detect presence of procedures."""
        lower_content = content.lower()
        
        for pattern in self.procedure_patterns:
            if re.search(pattern, lower_content, re.IGNORECASE):
                return True
        
        # Check for numbered lists suggesting steps
        if re.search(r'^\d+[\.)]\s+', content, re.MULTILINE):
            return True
        
        return False
    
    def _score_algorithms(self, content: str, has_algorithms: bool) -> float:
        """Score the quality of algorithms."""
        if not has_algorithms:
            return 0.0
        
        score = 0.3  # Base score for having algorithms
        
        # Check for structured algorithms
        if re.search(r'algorithm\s+\d+', content.lower()):
            score += 0.2
        
        # Check for step-by-step presentation
        numbered_steps = len(re.findall(r'^\d+[\.)]\s+', content, re.MULTILINE))
        if numbered_steps > 3:
            score += min(0.3, numbered_steps * 0.05)
        
        # Check for input/output specification
        if 'input:' in content.lower() and 'output:' in content.lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_equations(self, content: str, has_equations: bool) -> float:
        """Score the quality of equations."""
        if not has_equations:
            return 0.0
        
        score = 0.2  # Base score
        
        # Count equation instances
        equation_count = 0
        for pattern in self.equation_patterns[:4]:  # Use main patterns
            equation_count += len(re.findall(pattern, content))
        
        # More equations = higher score
        score += min(0.5, equation_count * 0.05)
        
        # Check for equation explanations
        if re.search(r'where\s+\w+\s*=', content, re.IGNORECASE):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_examples(self, content: str, has_examples: bool) -> float:
        """Score the quality of examples."""
        if not has_examples:
            return 0.0
        
        score = 0.2  # Base score
        
        # Count example instances
        example_count = sum(1 for pattern in self.example_patterns 
                           if re.search(pattern, content.lower()))
        
        score += min(0.5, example_count * 0.1)
        
        # Check for concrete vs abstract examples
        if re.search(r'\d+', content):  # Contains numbers
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_code(self, content: str, has_code: bool, metadata: Dict = None) -> float:
        """Score the quality of code."""
        if not has_code:
            return 0.0
        
        score = 0.5  # Base score for having code
        
        # Check metadata for code blocks
        if metadata and metadata.get('code_blocks'):
            code_blocks = metadata['code_blocks']
            
            # Executable code scores higher
            executable_count = sum(1 for block in code_blocks if block.get('executable'))
            if executable_count > 0:
                score += min(0.3, executable_count * 0.1)
            
            # Longer code blocks score higher
            total_lines = sum(block.get('lines', 0) for block in code_blocks)
            if total_lines > 10:
                score += 0.2
        
        # Check for complete code (imports, main, etc.)
        if 'import' in content and ('if __name__' in content or 'def main' in content):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_procedures(self, content: str, has_procedures: bool) -> float:
        """Score the quality of procedures."""
        if not has_procedures:
            return 0.0
        
        score = 0.3  # Base score
        
        # Count procedural indicators
        procedure_count = sum(1 for pattern in self.procedure_patterns 
                            if re.search(pattern, content.lower()))
        
        score += min(0.4, procedure_count * 0.1)
        
        # Check for clear sequencing
        sequence_words = ['first', 'second', 'then', 'next', 'finally', 'lastly']
        sequence_count = sum(1 for word in sequence_words if word in content.lower())
        if sequence_count >= 3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_structure(self, content: str, metadata: Dict = None) -> float:
        """Score the document structure."""
        score = 0.0
        
        # Check for headers/sections
        if metadata and metadata.get('structure'):
            structure = metadata['structure']
            if structure.get('headers'):
                score += min(0.3, len(structure['headers']) * 0.05)
            if structure.get('sections'):
                score += 0.2
        
        # Check for lists
        if re.search(r'^[\*\-\+]\s+', content, re.MULTILINE):
            score += 0.2
        
        # Check for clear paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_complexity(self, content: str, metadata: Dict = None) -> float:
        """Calculate content complexity."""
        complexity = 0.0
        
        # Word count factor
        words = content.split()
        word_count = len(words)
        if word_count > 5000:
            complexity += 0.3
        elif word_count > 2000:
            complexity += 0.2
        elif word_count > 500:
            complexity += 0.1
        
        # Vocabulary complexity
        unique_words = len(set(words))
        vocab_ratio = unique_words / word_count if word_count > 0 else 0
        complexity += min(0.3, vocab_ratio)
        
        # Technical terminology
        tech_terms = ['algorithm', 'theorem', 'proof', 'lemma', 'corollary', 
                     'implementation', 'optimization', 'complexity', 'polynomial']
        tech_count = sum(1 for term in tech_terms if term in content.lower())
        complexity += min(0.4, tech_count * 0.05)
        
        return min(complexity, 1.0)
    
    def _calculate_implementation_fidelity(self, algorithm_score: float, 
                                         equation_score: float, code_score: float,
                                         procedure_score: float, complexity: float) -> float:
        """Calculate how clearly content can be implemented."""
        # High code and procedure scores = high fidelity
        # High complexity reduces fidelity
        
        positive_factors = (
            code_score * 0.4 +
            procedure_score * 0.3 +
            algorithm_score * 0.2 +
            equation_score * 0.1
        )
        
        # Complexity acts as a dampener
        fidelity = positive_factors * (1 - complexity * 0.3)
        
        return min(max(fidelity, 0.0), 1.0)
    
    def _calculate_actionability(self, code_score: float, example_score: float,
                               procedure_score: float, structure_score: float) -> float:
        """Calculate how directly content can be applied."""
        actionability = (
            code_score * 0.4 +
            example_score * 0.3 +
            procedure_score * 0.2 +
            structure_score * 0.1
        )
        
        return min(max(actionability, 0.0), 1.0)
    
    def _calculate_bridge_potential(self, has_algorithms: bool, has_equations: bool,
                                  has_examples: bool, has_code: bool,
                                  algorithm_score: float, code_score: float) -> float:
        """Calculate theory to practice bridge potential."""
        # Theory indicators
        theory_score = 0.0
        if has_equations:
            theory_score += 0.3
        if has_algorithms:
            theory_score += 0.2
            
        # Practice indicators
        practice_score = 0.0
        if has_code:
            practice_score += 0.3
        if has_examples:
            practice_score += 0.2
            
        # Bridge exists when we have both
        if theory_score > 0 and practice_score > 0:
            bridge = (theory_score + practice_score) * 0.8
            
            # High quality algorithms and code strengthen the bridge
            bridge += (algorithm_score * code_score) * 0.2
            
            return min(bridge, 1.0)
        
        # No bridge if only theory or only practice
        return 0.0
    
    def _empty_metrics(self) -> ConveyanceMetrics:
        """Return empty metrics for no content."""
        return ConveyanceMetrics(
            implementation_fidelity=0.0,
            actionability=0.0,
            bridge_potential=0.0,
            has_algorithms=False,
            has_equations=False,
            has_examples=False,
            has_code=False,
            has_procedures=False,
            complexity_score=0.0,
            raw_scores={}
        )