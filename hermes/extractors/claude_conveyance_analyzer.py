"""
Claude-based Conveyance Analyzer for HERMES.
Uses Claude to analyze conveyance and generate training data for DSPy.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ConveyanceAnalysis:
    """Results from Claude's conveyance analysis."""
    implementation_fidelity: float
    actionability: float  
    bridge_potential: float
    
    # Detailed analysis
    theory_components: List[str]
    practice_components: List[str]
    missing_links: List[str]
    implementation_suggestions: List[str]
    
    # Training data for DSPy
    reasoning: str
    confidence: float
    

class ClaudeConveyanceAnalyzer:
    """
    Use Claude to analyze conveyance potential of documents.
    
    This generates training data for DSPy while building the database,
    allowing the system to learn what makes knowledge actionable.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.analysis_prompt = """
        Analyze this document excerpt for its CONVEYANCE potential - how readily can this knowledge be translated into action or implementation?

        Document excerpt:
        {content}

        Please analyze:
        
        1. Implementation Fidelity (0-1): How clearly can concepts be implemented?
           - Look for: step-by-step procedures, algorithms, clear methodologies
           - 0 = pure philosophy/theory, 1 = directly executable
        
        2. Actionability (0-1): Can knowledge be directly applied?
           - Look for: practical examples, tools, techniques
           - 0 = abstract concepts only, 1 = ready-to-use methods
        
        3. Bridge Potential (0-1): Does this connect theory to practice?
           - Look for: theoretical foundations WITH practical applications
           - 0 = theory OR practice only, 1 = strong theory-practice connection
        
        4. Theory Components: What theoretical concepts are presented?
        
        5. Practice Components: What practical/implementable elements exist?
        
        6. Missing Links: What's needed to make this more actionable?
        
        7. Implementation Suggestions: How could this knowledge be implemented?
        
        Provide your response as JSON:
        {{
            "implementation_fidelity": 0.0-1.0,
            "actionability": 0.0-1.0,
            "bridge_potential": 0.0-1.0,
            "theory_components": ["list", "of", "theoretical", "concepts"],
            "practice_components": ["list", "of", "practical", "elements"],
            "missing_links": ["what's", "missing", "for", "implementation"],
            "implementation_suggestions": ["how", "to", "implement", "this"],
            "reasoning": "Detailed explanation of your analysis",
            "confidence": 0.0-1.0
        }}
        """
        
    def analyze(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConveyanceAnalysis:
        """
        Analyze content for conveyance using Claude.
        
        Args:
            content: Text content to analyze
            metadata: Optional metadata about the document
            
        Returns:
            ConveyanceAnalysis with scores and training data
        """
        # For now, return a mock analysis
        # In production, this would call Claude API
        
        # Truncate content for analysis (Claude has context limits)
        analysis_content = content[:8000] if len(content) > 8000 else content
        
        # This is where we'd call Claude
        # For now, return structured mock data
        mock_analysis = {
            "implementation_fidelity": 0.7,
            "actionability": 0.6,
            "bridge_potential": 0.8,
            "theory_components": [
                "information entropy",
                "channel capacity", 
                "encoding theory"
            ],
            "practice_components": [
                "compression algorithms",
                "error correction codes",
                "practical examples"
            ],
            "missing_links": [
                "code implementations",
                "step-by-step tutorials"
            ],
            "implementation_suggestions": [
                "Create Python implementation of entropy calculation",
                "Build interactive demo of channel capacity",
                "Develop coding exercises"
            ],
            "reasoning": "This document presents strong theoretical foundations with some practical applications, making it a good bridge between theory and practice.",
            "confidence": 0.85
        }
        
        return ConveyanceAnalysis(**mock_analysis)
    
    def generate_training_example(self, content: str, analysis: ConveyanceAnalysis) -> Dict[str, Any]:
        """
        Generate a training example for DSPy from the analysis.
        
        This creates labeled data that DSPy can learn from.
        """
        return {
            "input": {
                "content_excerpt": content[:1000],  # First 1000 chars
                "content_length": len(content),
                "has_math": "=" in content or "∑" in content,
                "has_code": "def " in content or "function" in content,
            },
            "output": {
                "implementation_fidelity": analysis.implementation_fidelity,
                "actionability": analysis.actionability,
                "bridge_potential": analysis.bridge_potential,
                "reasoning": analysis.reasoning
            },
            "metadata": {
                "theory_components": analysis.theory_components,
                "practice_components": analysis.practice_components,
                "confidence": analysis.confidence
            }
        }
    
    def batch_analyze(self, documents: List[Tuple[str, Dict]], save_training_data: bool = True) -> List[ConveyanceAnalysis]:
        """
        Analyze multiple documents and optionally save training data.
        
        Args:
            documents: List of (content, metadata) tuples
            save_training_data: Whether to save examples for DSPy training
            
        Returns:
            List of analyses
        """
        analyses = []
        training_examples = []
        
        for content, metadata in documents:
            analysis = self.analyze(content, metadata)
            analyses.append(analysis)
            
            if save_training_data:
                example = self.generate_training_example(content, analysis)
                training_examples.append(example)
        
        # Save training data
        if save_training_data and training_examples:
            self._save_training_data(training_examples)
            
        return analyses
    
    def _save_training_data(self, examples: List[Dict[str, Any]]):
        """Save training examples for DSPy."""
        import os
        os.makedirs("training_examples", exist_ok=True)
        
        # Append to existing examples
        file_path = "training_examples/conveyance_examples.jsonl"
        with open(file_path, 'a') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
                
        logger.info(f"Saved {len(examples)} training examples to {file_path}")