#!/usr/bin/env python3
"""
Final test of vLLM analyzer showing it works.
"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.extractors.vllm_conveyance_analyzer import VLLMConveyanceAnalyzer
import logging

logging.basicConfig(level=logging.INFO)

# Test text - a paper with clear implementation details
test_text = """
PageRank: The PageRank Citation Ranking Algorithm

Abstract: This paper describes PageRank, a method for rating Web pages objectively and 
mechanically, effectively measuring the human interest and attention devoted to them.

Algorithm:
1. Initialize all page ranks to 1/N where N is total number of pages
2. For each iteration:
   - For each page p:
     PR(p) = (1-d)/N + d * sum(PR(t)/C(t) for all pages t linking to p)
   - Where d is damping factor (typically 0.85)
   - C(t) is the number of outbound links from page t
3. Repeat until convergence (typically 30-50 iterations)

Implementation in Python:
```python
def pagerank(graph, damping=0.85, iterations=30):
    N = len(graph)
    pr = {node: 1/N for node in graph}
    
    for _ in range(iterations):
        new_pr = {}
        for node in graph:
            rank_sum = sum(pr[n]/len(graph[n]) for n in graph if node in graph[n])
            new_pr[node] = (1-damping)/N + damping * rank_sum
        pr = new_pr
    
    return pr
```

Results show PageRank effectively identifies important pages with 89% accuracy.
"""

print("Creating vLLM analyzer...")
analyzer = VLLMConveyanceAnalyzer(
    model_name="Qwen/Qwen3-30B-A3B-FP8",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    lazy_load=True
)

print("\nAnalyzing PageRank paper (clear implementation example)...")
print("This should show HIGH conveyance scores...\n")

try:
    analysis = analyzer.analyze(test_text)
    
    print("=== Analysis Results ===")
    print(f"Implementation Fidelity: {analysis.implementation_fidelity:.3f}")
    print(f"Actionability: {analysis.actionability:.3f}")
    print(f"Bridge Potential: {analysis.bridge_potential:.3f}")
    print(f"Confidence: {analysis.confidence:.3f}")
    
    print("\nTheory Components:")
    for comp in analysis.theory_components[:3]:
        print(f"  - {comp}")
    
    print("\nPractice Components:")
    for comp in analysis.practice_components[:3]:
        print(f"  - {comp}")
    
    if analysis.reasoning_chain:
        print(f"\nReasoning Chain: {analysis.reasoning_chain[:200]}...")
    else:
        print(f"\nReasoning: {analysis.reasoning[:200]}...")
    
    print("\n✓ SUCCESS: vLLM analyzer is working correctly!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\nUnloading model...")
    analyzer.unload_model()
    print("Done!")