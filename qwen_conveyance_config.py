#!/usr/bin/env python3
"""
Configuration for Qwen3-30B conveyance analysis with reasoning.
Optimized for 20K context with DeepSeek R1 reasoning parser.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class QwenConveyanceConfig:
    """Configuration for Qwen-based conveyance analysis."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-30B-A3B-FP8"
    device: str = "cuda:0"  # Primary GPU
    max_model_len: int = 20480  # 20K context
    
    # Reasoning settings
    enable_reasoning: bool = True
    reasoning_parser: str = "deepseek_r1"
    
    # Context window allocation
    max_document_tokens: int = 16384  # ~16K for document
    max_response_tokens: int = 4096   # ~4K for analysis
    
    # Conveyance analysis prompts
    prompts: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize prompts after dataclass creation."""
        self.prompts = {
            "full_analysis": """<|im_start|>system
You are an expert at analyzing academic papers for their conveyance potential - how readily knowledge can be translated into action or implementation.

Analyze the provided document using multi-step reasoning:

1. First, identify the core contributions and claims
2. Assess how clearly these can be implemented or acted upon
3. Look for specific indicators of high/low conveyance
4. Consider the theory-to-practice translation potential
5. Provide detailed reasoning for your scores

Output your analysis in JSON format.
<|im_end|>

<|im_start|>user
Analyze this document for conveyance potential:

{document_text}

Please provide:
1. Implementation Fidelity (0-1): How clearly can concepts be implemented?
2. Actionability (0-1): Can knowledge be directly applied?
3. Bridge Potential (0-1): Does this connect theory to practice?

Include your reasoning chain and specific evidence from the text.
<|im_end|>

<|im_start|>assistant
I'll analyze this document's conveyance potential through systematic reasoning.

<reasoning>
First, let me identify the core contributions...
</reasoning>

Based on my analysis:
""",

            "citation_context": """<|im_start|>system
You are analyzing how one paper builds upon another through citation context analysis.
Focus on the semantic relationship between the citation context and the cited work.
<|im_end|>

<|im_start|>user
Analyze this citation:

Citing paper: {citing_title}
Cited paper: {cited_title}

Citation context:
{context_before}
[CITATION HERE: {citation_text}]
{context_after}

Cited paper abstract:
{cited_abstract}

Determine:
1. Citation type (builds_on, implements, replaces, negates, etc.)
2. Semantic influence score (0-1)
3. How the citing work transforms or uses the cited concepts
<|im_end|>

<|im_start|>assistant
<reasoning>
Let me analyze the relationship between these papers...
</reasoning>
""",

            "benchmark_extraction": """<|im_start|>system
Extract all benchmark results and quantitative improvements from this ML paper.
Be precise about metrics, datasets, and improvement percentages.
<|im_end|>

<|im_start|>user
Extract benchmark results from:

{document_text}

List all:
- Benchmark datasets used
- Metrics reported (accuracy, F1, BLEU, etc.)
- Numerical results
- Improvements over baselines
- Comparison with prior work
<|im_end|>

<|im_start|>assistant""",

            "implementation_assessment": """<|im_start|>system
Assess how implementable the methods in this paper are.
Consider code availability, algorithm clarity, and reproducibility.
<|im_end|>

<|im_start|>user
Assess implementation potential:

{document_text}

Evaluate:
1. Is the algorithm clearly described?
2. Are there implementation details?
3. Is code mentioned or available?
4. How difficult would it be to reproduce?
5. What's missing for full implementation?
<|im_end|>

<|im_start|>assistant"""
        }
    
    # Batch processing settings
    batch_size: int = 1  # Process one document at a time with full context
    
    # GPU memory management
    gpu_memory_fraction: float = 0.75  # Using 75% of GPU0
    offload_to_cpu: bool = False  # Keep on GPU for speed
    
    # Output settings
    save_reasoning_chains: bool = True
    reasoning_output_dir: str = "./conveyance_reasoning"
    
    # Validation settings
    validate_json_output: bool = True
    retry_on_parse_error: bool = True
    max_retries: int = 3
    
    # Performance optimization
    use_flash_attention: bool = True
    dtype: str = "float16"  # Model is FP8 but interface uses FP16
    
    def get_prompt(self, analysis_type: str, **kwargs) -> str:
        """Get formatted prompt for specific analysis type."""
        if analysis_type not in self.prompts:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
        return self.prompts[analysis_type].format(**kwargs)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (1 token ≈ 4 chars)."""
        return len(text) // 4
    
    def can_fit_document(self, document_text: str) -> bool:
        """Check if document fits in context window."""
        estimated_tokens = self.estimate_tokens(document_text)
        return estimated_tokens <= self.max_document_tokens


# Global config instance
qwen_config = QwenConveyanceConfig()


# Inference settings for vLLM
VLLM_ARGS = {
    "model": qwen_config.model_name,
    "max_model_len": qwen_config.max_model_len,
    "gpu_memory_utilization": qwen_config.gpu_memory_fraction,
    "dtype": qwen_config.dtype,
    "enable_reasoning": qwen_config.enable_reasoning,
    "reasoning_parser": qwen_config.reasoning_parser,
    "trust_remote_code": True,
}


def get_sampling_params(temperature: float = 0.7, 
                       max_tokens: int = None) -> Dict[str, Any]:
    """Get sampling parameters for inference."""
    return {
        "temperature": temperature,
        "max_tokens": max_tokens or qwen_config.max_response_tokens,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": ["<|im_end|>", "</reasoning>"]
    }


if __name__ == "__main__":
    print(f"Qwen Conveyance Configuration:")
    print(f"- Model: {qwen_config.model_name}")
    print(f"- Max context: {qwen_config.max_model_len:,} tokens")
    print(f"- Document capacity: ~{qwen_config.max_document_tokens * 4:,} characters")
    print(f"- GPU usage: {qwen_config.gpu_memory_fraction * 100:.0f}%")
    print(f"- Reasoning: {qwen_config.reasoning_parser}")
    print(f"\nReady for conveyance analysis!")