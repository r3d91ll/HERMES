#!/usr/bin/env python3
"""
Test the vLLM conveyance analyzer with a sample paper.
"""

import sys
from pathlib import Path
import logging
import json

# Add HERMES to path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.extractors.vllm_conveyance_analyzer import VLLMConveyanceAnalyzer
from hermes.loaders import PDFLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_analyzer():
    """Test the vLLM analyzer with a sample paper."""
    
    # Get a test paper
    papers_dir = Path("/home/todd/olympus/data/papers")
    pdf_files = list(papers_dir.glob("*.pdf"))[:1]
    
    if not pdf_files:
        logger.error("No PDF files found in papers directory")
        return
    
    test_pdf = pdf_files[0]
    logger.info(f"Testing with: {test_pdf.name}")
    
    # Load the paper
    loader = PDFLoader()
    doc_data = loader.load(test_pdf)
    text = doc_data.get('text', '')[:8000]  # First 8K chars for test
    
    logger.info(f"Loaded {len(text)} characters")
    
    # Create analyzer
    logger.info("Creating vLLM conveyance analyzer...")
    analyzer = VLLMConveyanceAnalyzer(
        model_name="Qwen/Qwen3-30B-A3B-FP8",
        max_model_len=20480,
        gpu_memory_utilization=0.75,
        enable_reasoning=True,
        reasoning_parser="deepseek_r1",
        lazy_load=True
    )
    
    # Analyze
    logger.info("Running conveyance analysis (this will load the model on first use)...")
    try:
        analysis = analyzer.analyze(text, {'source': test_pdf.name})
        
        # Display results
        print("\n=== Conveyance Analysis Results ===\n")
        print(f"Implementation Fidelity: {analysis.implementation_fidelity:.3f}")
        print(f"Actionability: {analysis.actionability:.3f}")
        print(f"Bridge Potential: {analysis.bridge_potential:.3f}")
        print(f"Confidence: {analysis.confidence:.3f}")
        
        print(f"\nTheory Components: {', '.join(analysis.theory_components[:5])}")
        print(f"Practice Components: {', '.join(analysis.practice_components[:5])}")
        print(f"Missing Links: {', '.join(analysis.missing_links[:3])}")
        
        print(f"\nReasoning: {analysis.reasoning[:200]}...")
        
        if analysis.reasoning_chain:
            print(f"\nReasoning Chain Preview: {analysis.reasoning_chain[:300]}...")
        
        # Save full result
        result_file = "vllm_analysis_result.json"
        with open(result_file, 'w') as f:
            json.dump({
                'document': test_pdf.name,
                'analysis': {
                    'implementation_fidelity': analysis.implementation_fidelity,
                    'actionability': analysis.actionability,
                    'bridge_potential': analysis.bridge_potential,
                    'theory_components': analysis.theory_components,
                    'practice_components': analysis.practice_components,
                    'missing_links': analysis.missing_links,
                    'implementation_suggestions': analysis.implementation_suggestions,
                    'reasoning': analysis.reasoning,
                    'confidence': analysis.confidence,
                    'reasoning_chain': analysis.reasoning_chain
                }
            }, f, indent=2)
        
        logger.info(f"Full analysis saved to {result_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    finally:
        # Clean up
        logger.info("Unloading model...")
        analyzer.unload_model()
        logger.info("Test complete!")


if __name__ == "__main__":
    test_analyzer()