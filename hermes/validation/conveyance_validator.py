"""
Conveyance validation framework for academic research quality.
Ensures consistency, interpretability, and methodological rigor.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a conveyance analysis."""
    doc_id: str
    is_valid: bool
    consistency_score: float
    issues: List[str]
    suggestions: List[str]
    metadata: Dict


class ConveyanceValidator:
    """
    Validates conveyance analyses for academic research quality.
    
    Ensures:
    1. Internal consistency of scores
    2. Theoretical alignment with multi-dimensional model
    3. Interpretability of results
    4. Methodological rigor
    """
    
    def __init__(self, validation_log_path: str = "validation_log.jsonl"):
        self.validation_log_path = validation_log_path
        self.validation_history = []
        
    def validate_analysis(
        self,
        analysis: Dict[str, any],
        document_content: str,
        doc_id: str
    ) -> ValidationResult:
        """
        Validate a single conveyance analysis.
        
        Args:
            analysis: ConveyanceAnalysis as dict
            document_content: Original document text
            doc_id: Document identifier
            
        Returns:
            ValidationResult with issues and suggestions
        """
        issues = []
        suggestions = []
        
        # 1. Check score bounds
        scores = [
            analysis.get('implementation_fidelity', 0),
            analysis.get('actionability', 0),
            analysis.get('bridge_potential', 0)
        ]
        
        for score_name, score in zip(
            ['implementation_fidelity', 'actionability', 'bridge_potential'],
            scores
        ):
            if not 0.0 <= score <= 1.0:
                issues.append(f"{score_name} out of bounds: {score}")
                
        # 2. Check logical consistency
        impl_fidelity = analysis.get('implementation_fidelity', 0)
        actionability = analysis.get('actionability', 0)
        bridge_potential = analysis.get('bridge_potential', 0)
        
        # High implementation should correlate with actionability
        if impl_fidelity > 0.7 and actionability < 0.3:
            issues.append(
                f"Inconsistent: high implementation_fidelity ({impl_fidelity:.2f}) "
                f"but low actionability ({actionability:.2f})"
            )
            suggestions.append(
                "Review actionability score - implementable content should be actionable"
            )
            
        # Bridge potential requires both theory and practice
        theory_count = len(analysis.get('theory_components', []))
        practice_count = len(analysis.get('practice_components', []))
        
        if bridge_potential > 0.6:
            if theory_count == 0 or practice_count == 0:
                issues.append(
                    f"High bridge_potential ({bridge_potential:.2f}) but missing "
                    f"theory ({theory_count}) or practice ({practice_count}) components"
                )
                suggestions.append(
                    "Bridge potential requires both theoretical and practical components"
                )
                
        # 3. Check component extraction quality
        if not analysis.get('theory_components') and 'theory' in document_content.lower():
            suggestions.append("Document mentions 'theory' but no theory components extracted")
            
        if not analysis.get('practice_components') and any(
            keyword in document_content.lower() 
            for keyword in ['implement', 'algorithm', 'code', 'example']
        ):
            suggestions.append("Document has practice indicators but no practice components extracted")
            
        # 4. Check reasoning quality
        reasoning = analysis.get('reasoning', '')
        if len(reasoning) < 50:
            issues.append("Reasoning too brief for academic standards")
            suggestions.append("Provide detailed reasoning explaining the scores")
            
        # 5. Calculate consistency score
        consistency_score = self._calculate_consistency_score(
            analysis, document_content, issues
        )
        
        # 6. Determine validity
        is_valid = len(issues) == 0 and consistency_score > 0.7
        
        result = ValidationResult(
            doc_id=doc_id,
            is_valid=is_valid,
            consistency_score=consistency_score,
            issues=issues,
            suggestions=suggestions,
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'scores': {
                    'implementation_fidelity': impl_fidelity,
                    'actionability': actionability,
                    'bridge_potential': bridge_potential
                },
                'component_counts': {
                    'theory': theory_count,
                    'practice': practice_count
                }
            }
        )
        
        # Log validation result
        self._log_validation(result)
        
        return result
    
    def _calculate_consistency_score(
        self,
        analysis: Dict,
        document_content: str,
        issues: List[str]
    ) -> float:
        """Calculate overall consistency score."""
        base_score = 1.0
        
        # Deduct for each issue
        base_score -= len(issues) * 0.1
        
        # Check if scores align with content
        impl_fidelity = analysis.get('implementation_fidelity', 0)
        content_lower = document_content.lower()
        
        # Implementation indicators
        impl_keywords = ['algorithm', 'step', 'procedure', 'method', 'implement']
        impl_indicator_count = sum(1 for kw in impl_keywords if kw in content_lower)
        
        # Expected implementation score based on content
        expected_impl = min(1.0, impl_indicator_count / 5.0)
        impl_deviation = abs(impl_fidelity - expected_impl)
        base_score -= impl_deviation * 0.2
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, base_score))
    
    def _log_validation(self, result: ValidationResult):
        """Log validation result for audit trail."""
        with open(self.validation_log_path, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')
            
    def validate_batch(
        self,
        analyses: List[Tuple[Dict, str, str]]
    ) -> Dict[str, any]:
        """
        Validate a batch of analyses.
        
        Args:
            analyses: List of (analysis, document_content, doc_id) tuples
            
        Returns:
            Summary statistics and detailed results
        """
        results = []
        for analysis, content, doc_id in analyses:
            result = self.validate_analysis(analysis, content, doc_id)
            results.append(result)
            
        # Calculate summary statistics
        valid_count = sum(1 for r in results if r.is_valid)
        avg_consistency = np.mean([r.consistency_score for r in results])
        
        # Identify common issues
        all_issues = []
        for r in results:
            all_issues.extend(r.issues)
            
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(':')[0]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
        return {
            'total_validated': len(results),
            'valid_count': valid_count,
            'validity_rate': valid_count / len(results) if results else 0,
            'avg_consistency_score': avg_consistency,
            'common_issues': issue_counts,
            'detailed_results': results
        }
    
    def generate_validation_report(self, output_path: str = "validation_report.md"):
        """Generate a markdown report of validation results."""
        # Read validation log
        validations = []
        with open(self.validation_log_path, 'r') as f:
            for line in f:
                validations.append(json.loads(line))
                
        if not validations:
            logger.warning("No validations found to report")
            return
            
        # Generate report
        report = f"""# Conveyance Validation Report

Generated: {datetime.utcnow().isoformat()}

## Summary

- Total Documents Validated: {len(validations)}
- Valid Analyses: {sum(1 for v in validations if v['is_valid'])}
- Average Consistency Score: {np.mean([v['consistency_score'] for v in validations]):.3f}

## Score Distributions

"""
        
        # Add score distributions
        for score_type in ['implementation_fidelity', 'actionability', 'bridge_potential']:
            scores = [
                v['metadata']['scores'][score_type] 
                for v in validations 
                if score_type in v['metadata']['scores']
            ]
            if scores:
                report += f"### {score_type.replace('_', ' ').title()}\n"
                report += f"- Mean: {np.mean(scores):.3f}\n"
                report += f"- Std Dev: {np.std(scores):.3f}\n"
                report += f"- Range: [{min(scores):.3f}, {max(scores):.3f}]\n\n"
                
        # Add common issues
        all_issues = []
        for v in validations:
            all_issues.extend(v['issues'])
            
        if all_issues:
            report += "## Common Issues\n\n"
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
                
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                report += f"- {issue} ({count} occurrences)\n"
                
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
            
        logger.info(f"Validation report saved to {output_path}")


class InterRaterReliability:
    """
    Check consistency between different analysis methods.
    Critical for academic validity.
    """
    
    def __init__(self):
        self.comparisons = []
        
    def compare_analyses(
        self,
        analysis1: Dict,
        analysis2: Dict,
        method1: str,
        method2: str
    ) -> Dict[str, float]:
        """
        Compare two analyses of the same document.
        
        Returns correlation metrics.
        """
        scores1 = [
            analysis1.get('implementation_fidelity', 0),
            analysis1.get('actionability', 0),
            analysis1.get('bridge_potential', 0)
        ]
        
        scores2 = [
            analysis2.get('implementation_fidelity', 0),
            analysis2.get('actionability', 0),
            analysis2.get('bridge_potential', 0)
        ]
        
        # Calculate correlation
        correlation = np.corrcoef(scores1, scores2)[0, 1]
        
        # Calculate mean absolute difference
        mad = np.mean(np.abs(np.array(scores1) - np.array(scores2)))
        
        comparison = {
            'method1': method1,
            'method2': method2,
            'correlation': correlation,
            'mean_absolute_difference': mad,
            'score_differences': {
                'implementation_fidelity': abs(scores1[0] - scores2[0]),
                'actionability': abs(scores1[1] - scores2[1]),
                'bridge_potential': abs(scores1[2] - scores2[2])
            }
        }
        
        self.comparisons.append(comparison)
        return comparison
    
    def calculate_cohens_kappa(self, threshold: float = 0.1) -> float:
        """
        Calculate Cohen's Kappa for inter-rater reliability.
        
        Args:
            threshold: Maximum difference to consider agreement
            
        Returns:
            Cohen's Kappa score
        """
        if not self.comparisons:
            return 0.0
            
        agreements = 0
        total = 0
        
        for comp in self.comparisons:
            for score_diff in comp['score_differences'].values():
                total += 1
                if score_diff <= threshold:
                    agreements += 1
                    
        observed_agreement = agreements / total if total > 0 else 0
        
        # For continuous scores, expected agreement by chance is low
        expected_agreement = 0.1  # Conservative estimate
        
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return kappa