"""
vLLM-based Conveyance Analyzer using Qwen3-30B.
Loads model on demand and manages GPU resources efficiently.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass, asdict
import torch
import os
import multiprocessing

# Set multiprocessing start method for CUDA
if __name__ != "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

logger = logging.getLogger(__name__)

# Only import vLLM when needed
vllm = None
SamplingParams = None


@dataclass
class ConveyanceAnalysis:
    """Results from conveyance analysis."""
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
    
    # Raw reasoning chain if available
    reasoning_chain: Optional[str] = None


class VLLMConveyanceAnalyzer:
    """
    Use vLLM with Qwen3-30B for conveyance analysis.
    Loads model on first use to manage resources.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-FP8",
        max_model_len: int = 20480,
        gpu_memory_utilization: float = 0.75,
        enable_reasoning: bool = True,
        reasoning_parser: str = "deepseek_r1",
        lazy_load: bool = True
    ):
        """
        Initialize analyzer with vLLM configuration.
        
        Args:
            model_name: HuggingFace model ID
            max_model_len: Maximum context length
            gpu_memory_utilization: Fraction of GPU to use
            enable_reasoning: Enable reasoning chains
            reasoning_parser: Parser for reasoning format
            lazy_load: Load model on first use (default True)
        """
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_reasoning = enable_reasoning
        self.reasoning_parser = reasoning_parser
        
        self.llm = None
        self.sampling_params = None
        
        if not lazy_load:
            self._load_model()
            
        self._setup_prompts()
    
    def _load_model(self):
        """Load vLLM model with configuration."""
        global vllm, SamplingParams
        
        if self.llm is not None:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading {self.model_name} with vLLM...")
        
        try:
            # Import vLLM
            from vllm import LLM, SamplingParams
            
            # Set environment variable to use legacy engine
            import os
            os.environ["VLLM_USE_V1"] = "0"  # Disable V1 engine
            
            # Create LLM instance
            self.llm = LLM(
                model=self.model_name,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                dtype="auto",  # Let it handle FP8
                enforce_eager=True  # Disable CUDA graphs for compatibility
            )
            
            # Default sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            
            logger.info(f"Model loaded successfully on GPU")
            
        except ImportError:
            logger.error("vLLM not installed. Install with: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_prompts(self):
        """Setup analysis prompts."""
        self.prompts = {
            "full_analysis": """<|im_start|>system
You are an expert at analyzing academic papers for their conveyance potential - how readily knowledge can be translated into action or implementation.

Analyze using these criteria:
1. Implementation Fidelity (0-1): How clearly can concepts be implemented?
2. Actionability (0-1): Can knowledge be directly applied?
3. Bridge Potential (0-1): Does this connect theory to practice?

Provide detailed reasoning and output valid JSON.
<|im_end|>

<|im_start|>user
Analyze this document excerpt for conveyance:

{content}

Output JSON with scores and detailed analysis.
<|im_end|>

<|im_start|>assistant
I'll analyze this document's conveyance potential systematically.

""",
            
            "simple_analysis": """Analyze this text for conveyance potential (how actionable/implementable it is):

{content}

Provide scores 0-1 for:
- implementation_fidelity
- actionability  
- bridge_potential

Output JSON with reasoning."""
        }
    
    def analyze(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConveyanceAnalysis:
        """
        Analyze content for conveyance using Qwen.
        
        Args:
            content: Text content to analyze
            metadata: Optional metadata about the document
            
        Returns:
            ConveyanceAnalysis with scores and training data
        """
        # Ensure model is loaded
        if self.llm is None:
            self._load_model()
        
        # Truncate content if needed
        max_chars = (self.max_model_len - 1000) * 4  # Reserve tokens for prompt/response
        if len(content) > max_chars:
            logger.warning(f"Truncating content from {len(content)} to {max_chars} chars")
            content = content[:max_chars]
        
        # Format prompt
        prompt = self.prompts["full_analysis"].format(content=content)
        
        try:
            # Run inference
            logger.info("Running conveyance analysis with Qwen...")
            outputs = self.llm.generate([prompt], self.sampling_params)
            
            if outputs and len(outputs) > 0:
                response_text = outputs[0].outputs[0].text
                logger.info(f"Model response length: {len(response_text)} chars")
                
                # Extract reasoning if present
                reasoning_chain = None
                if "<reasoning>" in response_text and "</reasoning>" in response_text:
                    start = response_text.find("<reasoning>") + len("<reasoning>")
                    end = response_text.find("</reasoning>")
                    reasoning_chain = response_text[start:end].strip()
                elif "<think>" in response_text and "</think>" in response_text:
                    # Qwen sometimes uses <think> tags
                    start = response_text.find("<think>") + len("<think>")
                    end = response_text.find("</think>")
                    reasoning_chain = response_text[start:end].strip()
                
                # Parse JSON from response
                analysis_dict = self._parse_response(response_text)
                
                # Add reasoning chain if found
                if reasoning_chain:
                    analysis_dict["reasoning_chain"] = reasoning_chain
                
                return ConveyanceAnalysis(**analysis_dict)
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Fallback to pattern-based analysis
            return self._pattern_based_fallback(content, metadata)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from model response."""
        try:
            # Try to find all JSON blocks
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = list(re.finditer(json_pattern, response, re.DOTALL))
            
            # Try each JSON block to find one with the required fields
            for match in json_matches:
                json_str = match.group(0)
                try:
                    parsed = json.loads(json_str)
                    
                    # Check if this JSON has the score fields we need
                    if all(key in parsed for key in ["implementation_fidelity", "actionability", "bridge_potential"]):
                        # Extract analysis narratives if present
                        if "analysis" in parsed and isinstance(parsed["analysis"], dict):
                            analysis = parsed["analysis"]
                            # Use the detailed analysis as components
                            impl_text = analysis.get("implementation_fidelity_analysis", "")
                            action_text = analysis.get("actionability_analysis", "")
                            bridge_text = analysis.get("bridge_potential_analysis", "")
                            
                            parsed["theory_components"] = [s.strip() for s in impl_text.split(".")[:2] if s.strip()]
                            parsed["practice_components"] = [s.strip() for s in action_text.split(".")[:2] if s.strip()]
                            parsed["missing_links"] = [s.strip() for s in bridge_text.split(".")[:2] if s.strip()]
                            parsed["reasoning"] = f"Implementation: {impl_text[:100]}..."
                        break
                except json.JSONDecodeError:
                    continue
            else:
                # No valid JSON found with required fields
                logger.warning("No valid JSON with required fields found in response")
                parsed = {}
                
                # Ensure all required fields
                required_fields = {
                    "implementation_fidelity": 0.5,
                    "actionability": 0.5,
                    "bridge_potential": 0.5,
                    "theory_components": [],
                    "practice_components": [],
                    "missing_links": [],
                    "implementation_suggestions": [],
                    "reasoning": "Analysis based on model output",
                    "confidence": 0.8
                }
                
                for field, default in required_fields.items():
                    if field not in parsed:
                        parsed[field] = default
                        
                return parsed
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            
        # Return defaults if parsing fails
        return self._get_default_analysis()
    
    def _pattern_based_fallback(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConveyanceAnalysis:
        """Fallback to pattern-based analysis if model fails."""
        logger.info("Using pattern-based fallback analysis")
        
        content_lower = content.lower()
        
        # Simple heuristics
        implementation_keywords = ["algorithm", "method", "step", "procedure", "implement"]
        theory_keywords = ["theory", "hypothesis", "concept", "principle", "abstract"]
        practice_keywords = ["example", "application", "code", "results", "experiment"]
        
        impl_score = sum(1 for kw in implementation_keywords if kw in content_lower) / len(implementation_keywords)
        theory_score = sum(1 for kw in theory_keywords if kw in content_lower) / len(theory_keywords)
        practice_score = sum(1 for kw in practice_keywords if kw in content_lower) / len(practice_keywords)
        
        # Calculate scores
        implementation_fidelity = min(1.0, impl_score * 2)
        actionability = min(1.0, practice_score * 2)
        bridge_potential = min(1.0, (theory_score + practice_score) / 2)
        
        return ConveyanceAnalysis(
            implementation_fidelity=implementation_fidelity,
            actionability=actionability,
            bridge_potential=bridge_potential,
            theory_components=["general theoretical concepts"],
            practice_components=["general practical elements"],
            missing_links=["detailed implementation steps"],
            implementation_suggestions=["Provide concrete examples"],
            reasoning="Pattern-based analysis (model unavailable)",
            confidence=0.6
        )
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis structure."""
        return {
            "implementation_fidelity": 0.5,
            "actionability": 0.5,
            "bridge_potential": 0.5,
            "theory_components": ["concepts identified"],
            "practice_components": ["methods described"],
            "missing_links": ["specific details"],
            "implementation_suggestions": ["clarify approach"],
            "reasoning": "Default analysis",
            "confidence": 0.5
        }
    
    def batch_analyze(self, documents: List[Tuple[str, Dict]], save_training_data: bool = True) -> List[ConveyanceAnalysis]:
        """
        Analyze multiple documents.
        
        Args:
            documents: List of (content, metadata) tuples
            save_training_data: Whether to save examples for training
            
        Returns:
            List of analyses
        """
        analyses = []
        training_examples = []
        
        for i, (content, metadata) in enumerate(documents):
            logger.info(f"Analyzing document {i+1}/{len(documents)}")
            
            analysis = self.analyze(content, metadata)
            analyses.append(analysis)
            
            if save_training_data:
                example = {
                    "instruction": f"Analyze this document for conveyance potential:\n{content[:1000]}...",
                    "response": asdict(analysis),
                    "metadata": metadata
                }
                training_examples.append(example)
        
        if save_training_data and training_examples:
            self._save_training_data(training_examples)
            
        return analyses
    
    def _save_training_data(self, examples: List[Dict[str, Any]]):
        """Save training examples for fine-tuning."""
        os.makedirs("training_examples", exist_ok=True)
        
        file_path = "training_examples/qwen_conveyance_examples.jsonl"
        with open(file_path, 'a') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
                
        logger.info(f"Saved {len(examples)} training examples to {file_path}")
    
    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.llm is not None:
            logger.info("Unloading model...")
            del self.llm
            self.llm = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Model unloaded, GPU memory freed")


def main():
    """Test the analyzer."""
    print("""
vLLM Conveyance Analyzer Ready!

This will:
1. Load Qwen3-30B-A3B-FP8 on first use
2. Analyze documents with 20K context
3. Generate reasoning chains
4. Create training data for fine-tuning

The model will load when you first call analyze().
""")


if __name__ == "__main__":
    main()