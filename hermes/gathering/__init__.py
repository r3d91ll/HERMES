"""
Paper gathering modules for HERMES.

This package contains specialized gatherers for different paper sources
and topics, all configurable through YAML.
"""

from .base_gatherer import BaseGatherer
from .arxiv_gatherer import ArxivGatherer
from .chronological_gatherer import ChronologicalGatherer
from .priority_gatherer import PriorityGatherer
from .quantum_gatherer import QuantumObserverGatherer

__all__ = [
    'BaseGatherer',
    'ArxivGatherer', 
    'ChronologicalGatherer',
    'PriorityGatherer',
    'QuantumObserverGatherer'
]