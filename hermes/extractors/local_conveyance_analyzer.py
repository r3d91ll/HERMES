"""
Local LLM-based Conveyance Analyzer for HERMES.
Uses local models (Qwen, Llama, etc.) for development and testing.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass, asdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .claude_conveyance_analyzer import ConveyanceAnalysis

logger = logging.getLogger(__name__)


class LocalConveyanceAnalyzer:
    """
    Use local LLMs to analyze conveyance potential of documents.
    
    This allows development without API costs and creates training
    data for fine-tuning models specifically for HADES.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-FP8",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,  # FP8 model already quantized
        max_length: int = 20480,  # 20K context as configured
        use_vllm: bool = True,  # Use vLLM for inference
    ):
        """
        Initialize local model analyzer.
        
        Args:
            model_name: HuggingFace model ID
            device: Device to run on
            load_in_8bit: Use 8-bit quantization to save memory
            max_length: Maximum context length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading local model: {model_name}")
        
        # Load model and tokenizer
        model_kwargs = {
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": True,  # Required for Qwen models
        }
        
        # FP8 models are pre-quantized, use appropriate dtype
        if "FP8" in model_name:
            # FP8 models handle their own quantization
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
            if load_in_8bit and device == "cuda":
                model_kwargs["load_in_8bit"] = True
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loaded {model_name} successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to mock analyzer")
            self.model = None
            self.tokenizer = None
            
        self.analysis_prompt_template = """Analyze this document excerpt for its CONVEYANCE potential - how readily can this knowledge be translated into action or implementation?

Document excerpt:
{content}

Please analyze and provide scores from 0.0 to 1.0:

1. Implementation Fidelity: How clearly can concepts be implemented?
   - 0 = pure philosophy/theory with no clear steps
   - 1 = directly executable with clear procedures

2. Actionability: Can knowledge be directly applied?
   - 0 = abstract concepts only
   - 1 = ready-to-use methods and tools

3. Bridge Potential: Does this connect theory to practice?
   - 0 = theory OR practice only
   - 1 = strong theory-practice connection

Also identify:
- Theory Components: theoretical concepts presented
- Practice Components: practical/implementable elements
- Missing Links: what's needed to make this more actionable
- Implementation Suggestions: how to implement this knowledge

Respond in JSON format:
{{
    "implementation_fidelity": 0.0-1.0,
    "actionability": 0.0-1.0,
    "bridge_potential": 0.0-1.0,
    "theory_components": ["list of concepts"],
    "practice_components": ["list of practical elements"],
    "missing_links": ["what's missing"],
    "implementation_suggestions": ["how to implement"],
    "reasoning": "explanation",
    "confidence": 0.0-1.0
}}"""

    def analyze(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConveyanceAnalysis:
        """
        Analyze content for conveyance using local model.
        
        Args:
            content: Text content to analyze
            metadata: Optional metadata about the document
            
        Returns:
            ConveyanceAnalysis with scores and training data
        """
        # Truncate content for analysis (can handle much more with 16K context)
        # Reserve ~2K tokens for prompt and response, use ~14K for content
        # Rough estimate: 1 token ≈ 4 chars
        max_content_chars = (self.max_length - 2048) * 4
        analysis_content = content[:max_content_chars] if len(content) > max_content_chars else content
        
        if self.model is None or self.tokenizer is None:
            # Fallback to pattern-based analysis
            return self._pattern_based_analysis(analysis_content, metadata)
            
        try:
            # Format prompt
            prompt = self.analysis_prompt_template.format(content=analysis_content)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            
            if self.device == "cuda":
                inputs = inputs.to("cuda")
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    analysis_dict = json.loads(json_str)
                    return ConveyanceAnalysis(**analysis_dict)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse model JSON response, using pattern analysis")
                    
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            
        # Fallback to pattern-based analysis
        return self._pattern_based_analysis(analysis_content, metadata)
    
    def _pattern_based_analysis(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConveyanceAnalysis:
        """
        Fallback pattern-based analysis when model is unavailable.
        
        This provides consistent baseline scores for development.
        """
        content_lower = content.lower()
        
        # Count indicators
        algorithm_indicators = [
            "algorithm", "procedure", "step", "method", "function",
            "implement", "code", "pseudocode", "formula", "equation"
        ]
        
        theory_indicators = [
            "theory", "hypothesis", "conjecture", "principle", "axiom",
            "theorem", "lemma", "proposition", "abstract", "conceptual"
        ]
        
        practice_indicators = [
            "example", "application", "use case", "implementation",
            "tool", "library", "framework", "practical", "real-world"
        ]
        
        # Count occurrences
        algorithm_count = sum(1 for word in algorithm_indicators if word in content_lower)
        theory_count = sum(1 for word in theory_indicators if word in content_lower)
        practice_count = sum(1 for word in practice_indicators if word in content_lower)
        
        # Calculate scores
        total_indicators = algorithm_count + theory_count + practice_count
        
        if total_indicators == 0:
            implementation_fidelity = 0.3
            actionability = 0.2
            bridge_potential = 0.1
        else:
            implementation_fidelity = min(0.9, algorithm_count / 10.0)
            actionability = min(0.9, practice_count / 8.0)
            
            # Bridge potential is high when we have both theory and practice
            if theory_count > 0 and practice_count > 0:
                bridge_potential = min(0.9, (theory_count + practice_count) / 15.0)
            else:
                bridge_potential = 0.2
        
        # Extract components
        theory_components = []
        practice_components = []
        
        # Simple extraction based on keywords
        if "information theory" in content_lower:
            theory_components.append("information theory")
        if "entropy" in content_lower:
            theory_components.append("entropy")
        if "algorithm" in content_lower:
            practice_components.append("algorithms")
        if "implementation" in content_lower:
            practice_components.append("implementation details")
            
        return ConveyanceAnalysis(
            implementation_fidelity=implementation_fidelity,
            actionability=actionability,
            bridge_potential=bridge_potential,
            theory_components=theory_components or ["general concepts"],
            practice_components=practice_components or ["general methods"],
            missing_links=["detailed examples", "code implementations"] if implementation_fidelity < 0.7 else [],
            implementation_suggestions=[
                "Create reference implementation",
                "Add practical examples"
            ] if actionability < 0.7 else ["Document is already actionable"],
            reasoning=f"Pattern-based analysis found {algorithm_count} algorithm indicators, "
                     f"{theory_count} theory indicators, and {practice_count} practice indicators.",
            confidence=0.7  # Pattern-based analysis has moderate confidence
        )
    
    def generate_training_example(self, content: str, analysis: ConveyanceAnalysis) -> Dict[str, Any]:
        """
        Generate a training example for fine-tuning from the analysis.
        """
        return {
            "instruction": self.analysis_prompt_template.format(content=content[:1000]),
            "response": json.dumps(asdict(analysis), indent=2),
            "metadata": {
                "model_used": self.model_name if self.model else "pattern_based",
                "content_length": len(content),
                "confidence": analysis.confidence
            }
        }
    
    def save_for_fine_tuning(self, examples: List[Dict[str, Any]], output_path: str = "conveyance_training_data.jsonl"):
        """
        Save training examples in format suitable for fine-tuning.
        """
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'a') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
                
        logger.info(f"Saved {len(examples)} training examples to {output_path}")