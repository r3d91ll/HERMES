#!/usr/bin/env python3
"""
Test the conveyance analyzer on sample papers.
"""

import sys
from pathlib import Path
from pprint import pprint

# Add HERMES to path
sys.path.insert(0, str(Path(__file__).parent))

from hermes.loaders import PDFLoader
from hermes.extractors import ConveyanceAnalyzer


def test_conveyance(file_path: Path):
    """Test conveyance analysis on a file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {file_path.name}")
    print(f"{'='*60}")
    
    # Load the file
    loader = PDFLoader()
    
    try:
        doc_data = loader.load(file_path)
        content = doc_data['text']  # PDF loader returns 'text' not 'content'
        
        # Analyze conveyance
        analyzer = ConveyanceAnalyzer()
        metrics = analyzer.analyze(content, doc_data.get('metadata'))
        
        # Display results
        print(f"\nConveyance Metrics:")
        print(f"  Implementation Fidelity: {metrics.implementation_fidelity:.3f}")
        print(f"  Actionability:          {metrics.actionability:.3f}")
        print(f"  Bridge Potential:       {metrics.bridge_potential:.3f}")
        
        print(f"\nIndicators:")
        print(f"  Has Algorithms: {metrics.has_algorithms}")
        print(f"  Has Equations:  {metrics.has_equations}")
        print(f"  Has Examples:   {metrics.has_examples}")
        print(f"  Has Code:       {metrics.has_code}")
        print(f"  Has Procedures: {metrics.has_procedures}")
        
        print(f"\nComplexity Score: {metrics.complexity_score:.3f}")
        
        print(f"\nRaw Scores:")
        for key, value in metrics.raw_scores.items():
            print(f"  {key}: {value:.3f}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Test conveyance on sample papers."""
    papers_dir = Path("/home/todd/olympus/data/papers")
    
    # Test on contrasting papers
    test_files = [
        "1948_Shannon_Mathematical_Theory_Communication.pdf",  # High conveyance
        "1970_Derrida_Structure_Sign_Play.pdf",  # Low conveyance
        "2017_Vaswani_Attention_Is_All_You_Need.pdf",  # High conveyance
        "1967_Foucault_Of_Other_Spaces_Heterotopias.pdf"  # Low conveyance
    ]
    
    for filename in test_files:
        file_path = papers_dir / filename
        if file_path.exists():
            test_conveyance(file_path)
        else:
            print(f"File not found: {filename}")


if __name__ == "__main__":
    main()