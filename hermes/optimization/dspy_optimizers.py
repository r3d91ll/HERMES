"""
DSPy optimization for HERMES pipeline components.

This module implements DSPy's reconstruction approach to optimize
each stage of the HERMES pipeline. Rather than hand-crafting prompts,
we let the system discover optimal instructions through interaction
with actual data - a perfect embodiment of reconstruction theory.
"""

import dspy
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


# HERMES-specific DSPy Signatures

class DocumentMetadataExtraction(dspy.Signature):
    """Extract structured metadata from document content."""
    document_content = dspy.InputField(desc="Raw document text content")
    document_path = dspy.InputField(desc="File path for context")
    document_type = dspy.InputField(desc="File type (pdf, py, md, etc)")
    
    metadata = dspy.OutputField(desc="JSON metadata with domain, complexity, prerequisites, concepts")


class CodeSymbolExtraction(dspy.Signature):
    """Extract symbols and relationships from code."""
    code_content = dspy.InputField(desc="Python source code")
    file_path = dspy.InputField(desc="Code file path")
    
    symbols = dspy.OutputField(desc="JSON with functions, classes, imports, and relationships")
    complexity_assessment = dspy.OutputField(desc="Code complexity analysis")


class ConveyanceAssessment(dspy.Signature):
    """Assess conveyance characteristics of content."""
    content = dspy.InputField(desc="Document or code content")
    metadata = dspy.InputField(desc="Extracted metadata")
    
    implementation_fidelity = dspy.OutputField(desc="How well does this implement its concepts? (0-1)")
    actionability = dspy.OutputField(desc="How actionable is this content? (0-1)")
    bridge_potential = dspy.OutputField(desc="Potential to bridge theory and practice (0-1)")


class DirectoryPurposeInference(dspy.Signature):
    """Infer purpose from directory structure."""
    directory_path = dspy.InputField(desc="Directory path")
    file_names = dspy.InputField(desc="List of files in directory")
    subdirectories = dspy.InputField(desc="List of subdirectories")
    
    inferred_purpose = dspy.OutputField(desc="Primary purpose (docs, src, tests, etc)")
    semantic_tags = dspy.OutputField(desc="Semantic tags for directory content")


# DSPy Modules for HERMES

class MetadataExtractor(dspy.Module):
    """DSPy module for metadata extraction."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(DocumentMetadataExtraction)
    
    def forward(self, document_content, document_path, document_type):
        prediction = self.extract(
            document_content=document_content[:2000],  # Limit context
            document_path=document_path,
            document_type=document_type
        )
        
        # Parse and validate metadata
        try:
            metadata = json.loads(prediction.metadata)
            # Ensure required fields
            required_fields = ["domain", "complexity", "prerequisites", "concepts"]
            for field in required_fields:
                if field not in metadata:
                    metadata[field] = []
            return metadata
        except:
            # Fallback structure
            return {
                "domain": [],
                "complexity": ["unknown"],
                "prerequisites": [],
                "concepts": []
            }


class ConveyanceAnalyzer(dspy.Module):
    """DSPy module for conveyance analysis."""
    
    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(ConveyanceAssessment)
    
    def forward(self, content, metadata):
        prediction = self.assess(
            content=content[:2000],
            metadata=json.dumps(metadata)
        )
        
        # Convert to float scores
        try:
            return {
                "implementation_fidelity": float(prediction.implementation_fidelity),
                "actionability": float(prediction.actionability),
                "bridge_potential": float(prediction.bridge_potential)
            }
        except:
            return {
                "implementation_fidelity": 0.5,
                "actionability": 0.5,
                "bridge_potential": 0.5
            }


# HERMES Pipeline Optimizer

class HERMESPipelineOptimizer:
    """
    Orchestrates DSPy optimization across HERMES pipeline stages.
    
    This embodies reconstruction theory - we don't prescribe how to
    extract metadata or assess conveyance. Instead, we let the system
    discover optimal strategies through interaction with real data.
    """
    
    def __init__(self, lm: Optional[dspy.LM] = None):
        """Initialize with language model."""
        if lm:
            dspy.settings.configure(lm=lm)
        
        # Initialize modules
        self.metadata_extractor = MetadataExtractor()
        self.conveyance_analyzer = ConveyanceAnalyzer()
        
        # Optimization state
        self.optimized_modules = {}
    
    def optimize_metadata_extraction(
        self,
        training_data: List[Dict[str, Any]],
        metric: Optional[Callable] = None,
        budget: str = "medium"
    ) -> MetadataExtractor:
        """
        Optimize metadata extraction using DSPy.
        
        Args:
            training_data: List of {document, expected_metadata} pairs
            metric: Custom metric function
            budget: Optimization budget (low/medium/high)
        
        Returns:
            Optimized metadata extractor
        """
        if metric is None:
            metric = self._default_metadata_metric
        
        # Choose optimizer based on budget and data size
        optimizer = self._select_optimizer(
            stage="metadata",
            data_size=len(training_data),
            budget=budget
        )
        
        # Compile optimized module
        optimized = optimizer.compile(
            self.metadata_extractor,
            trainset=training_data,
            metric=metric
        )
        
        self.optimized_modules["metadata"] = optimized
        return optimized
    
    def optimize_conveyance_analysis(
        self,
        training_data: List[Dict[str, Any]],
        metric: Optional[Callable] = None,
        budget: str = "medium"
    ) -> ConveyanceAnalyzer:
        """Optimize conveyance analysis."""
        if metric is None:
            metric = self._default_conveyance_metric
        
        optimizer = self._select_optimizer(
            stage="conveyance",
            data_size=len(training_data),
            budget=budget
        )
        
        optimized = optimizer.compile(
            self.conveyance_analyzer,
            trainset=training_data,
            metric=metric
        )
        
        self.optimized_modules["conveyance"] = optimized
        return optimized
    
    def _select_optimizer(self, stage: str, data_size: int, budget: str):
        """Select appropriate DSPy optimizer."""
        
        # Budget mappings
        budget_limits = {
            "low": 50,
            "medium": 200,
            "high": 1000
        }
        
        if data_size < 50 or budget == "low":
            # Fast iteration with few examples
            return dspy.BootstrapFewShot(
                max_bootstrapped_demos=4,
                max_labeled_demos=4
            )
        
        elif data_size < 200 or budget == "medium":
            # Better exploration with random search
            return dspy.BootstrapFewShotWithRandomSearch(
                max_bootstrapped_demos=4,
                num_candidate_programs=10,
                num_threads=4
            )
        
        else:  # Large data or high budget
            # Full optimization with MIPROv2
            return dspy.MIPROv2(
                metric=None,  # Will be set by compile
                auto="medium",
                num_trials=20
            )
    
    def _default_metadata_metric(self, example, prediction, trace=None):
        """Default metric for metadata quality."""
        try:
            # Check if all required fields are present
            required = ["domain", "complexity", "prerequisites", "concepts"]
            for field in required:
                if field not in prediction or not prediction[field]:
                    return 0.0
            
            # Score based on richness
            score = 0.0
            score += min(len(prediction["domain"]) / 5, 1.0) * 0.25
            score += min(len(prediction["concepts"]) / 10, 1.0) * 0.25
            score += 0.25 if prediction["complexity"] else 0.0
            score += min(len(prediction["prerequisites"]) / 3, 1.0) * 0.25
            
            return score
        except:
            return 0.0
    
    def _default_conveyance_metric(self, example, prediction, trace=None):
        """Default metric for conveyance assessment."""
        try:
            # All scores should be between 0 and 1
            scores = [
                prediction["implementation_fidelity"],
                prediction["actionability"],
                prediction["bridge_potential"]
            ]
            
            if all(0 <= s <= 1 for s in scores):
                # Reward differentiation (not all 0.5)
                variance = np.var(scores)
                return 0.7 + (0.3 * min(variance * 10, 1.0))
            return 0.0
        except:
            return 0.0
    
    def create_ensemble(self, modules: List[dspy.Module]) -> dspy.Module:
        """Create ensemble from multiple optimized modules."""
        return dspy.Ensemble(reduce_fn=dspy.majority).compile(modules)
    
    def save_optimized_modules(self, path: str):
        """Save optimized modules for reuse."""
        # DSPy handles serialization
        for name, module in self.optimized_modules.items():
            module.save(f"{path}/hermes_{name}_optimized.json")
    
    def load_optimized_modules(self, path: str):
        """Load previously optimized modules."""
        # This demonstrates reconstruction - the optimized prompts
        # are discovered patterns, not hand-coded instructions
        pass


# Metrics for HERMES optimization

def comprehensive_metadata_metric(example, prediction, trace=None):
    """
    Comprehensive metric for metadata extraction quality.
    
    This metric embodies HERMES' goals:
    - Rich domain classification
    - Accurate complexity assessment
    - Clear prerequisite identification
    - Comprehensive concept extraction
    """
    score = 0.0
    
    # Domain richness (WHERE dimension)
    domains = prediction.get("domain", [])
    if domains:
        score += 0.25 * min(len(domains) / 5, 1.0)
    
    # Complexity accuracy (CONVEYANCE dimension)
    complexity = prediction.get("complexity", [])
    if complexity and complexity[0] in ["beginner", "intermediate", "advanced"]:
        score += 0.25
    
    # Prerequisites (CONVEYANCE dimension)
    prereqs = prediction.get("prerequisites", [])
    score += 0.25 * min(len(prereqs) / 3, 1.0)
    
    # Concepts (WHAT dimension)
    concepts = prediction.get("concepts", [])
    score += 0.25 * min(len(concepts) / 10, 1.0)
    
    return score


def bridge_detection_metric(example, prediction, trace=None):
    """
    Metric for detecting theory-practice bridges.
    
    High scores for content that:
    - Shows high implementation fidelity
    - Is actionable
    - Has bridge potential
    """
    try:
        fidelity = float(prediction.get("implementation_fidelity", 0))
        actionability = float(prediction.get("actionability", 0))
        bridge = float(prediction.get("bridge_potential", 0))
        
        # Multiplicative model - all dimensions matter
        score = fidelity * actionability * bridge
        
        # Boost for high bridge potential
        if bridge > 0.8:
            score *= 1.5
        
        return min(score, 1.0)
    except:
        return 0.0


# Example usage for HERMES

def optimize_hermes_pipeline(training_documents: List[Dict[str, Any]]):
    """
    Example of optimizing HERMES pipeline with DSPy.
    
    This shows how DSPy's reconstruction approach aligns with HERMES:
    - No hand-crafted prompts
    - Optimization emerges from data
    - Composable, reusable modules
    """
    # Configure DSPy with LM
    lm = dspy.OpenAI(model="gpt-4", max_tokens=1000)
    dspy.settings.configure(lm=lm)
    
    # Initialize optimizer
    optimizer = HERMESPipelineOptimizer(lm=lm)
    
    # Prepare training data
    metadata_train = [
        {
            "document_content": doc["content"],
            "document_path": doc["path"],
            "document_type": doc["type"],
            "expected_metadata": doc["metadata"]
        }
        for doc in training_documents
    ]
    
    # Optimize metadata extraction
    logger.info("Optimizing metadata extraction...")
    optimized_metadata = optimizer.optimize_metadata_extraction(
        training_data=metadata_train,
        metric=comprehensive_metadata_metric,
        budget="medium"
    )
    
    # Optimize conveyance analysis
    logger.info("Optimizing conveyance analysis...")
    conveyance_train = [
        {
            "content": doc["content"],
            "metadata": doc["metadata"],
            "expected_scores": doc["conveyance_scores"]
        }
        for doc in training_documents
        if "conveyance_scores" in doc
    ]
    
    optimized_conveyance = optimizer.optimize_conveyance_analysis(
        training_data=conveyance_train,
        metric=bridge_detection_metric,
        budget="medium"
    )
    
    return optimizer