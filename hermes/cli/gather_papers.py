#!/usr/bin/env python3
"""
Unified paper gathering CLI for HERMES.

This provides a single interface to all paper gathering functionality
with configuration file support.
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hermes.gathering import (
    ArxivGatherer,
    ChronologicalGatherer,
    PriorityGatherer,
    QuantumObserverGatherer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_file: Path) -> Dict:
    """Load configuration from YAML file."""
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def gather_chronological(args, config):
    """Gather papers chronologically."""
    chrono_config = config.get('chronological', {})
    rate_config = config.get('rate_limiting', {})
    
    output_dir = args.output_dir or Path(config['output_dirs']['chronological'])
    
    gatherer = ChronologicalGatherer(
        output_dir=output_dir,
        config={**chrono_config, **rate_config}
    )
    
    gatherer.gather_papers(
        start_year=args.start_year or chrono_config.get('start_year', 1998),
        end_year=args.end_year or chrono_config.get('end_year'),
        papers_per_year=args.papers_per_year or chrono_config.get('papers_per_year', 50)
    )
    
    if args.analyze:
        gatherer.analyze_collection()


def gather_topics(args, config):
    """Gather papers by topics."""
    rate_config = config.get('rate_limiting', {})
    output_dir = args.output_dir or Path(config['output_dirs']['priority'])
    
    gatherer = ArxivGatherer(
        output_dir=output_dir,
        config=rate_config
    )
    
    topics = args.topics or config['priority']['topics']
    papers_per_topic = args.papers_per_topic or config['priority']['papers_per_topic']
    
    gatherer.gather_papers(
        topics=topics,
        papers_per_topic=papers_per_topic,
        start_year=args.start_year,
        end_year=args.end_year
    )
    
    if args.analyze:
        gatherer.analyze_collection()


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="HERMES Paper Gathering Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Gather papers chronologically
  %(prog)s chronological --start-year 1998 --papers-per-year 50
  
  # Gather papers by topics
  %(prog)s topics --topics quantum_observer node_embeddings --papers-per-topic 100
  
  # Use custom config
  %(prog)s --config my_config.yaml chronological
  
  # Analyze existing collection
  %(prog)s topics --analyze-only --output-dir ./data/my_papers
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=Path,
                       default=Path(__file__).parent.parent / 'gathering/config/gathering_config.yaml',
                       help='Configuration file')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory (overrides config)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing collection')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Gathering method')
    
    # Chronological gathering
    chrono_parser = subparsers.add_parser('chronological', 
                                         help='Gather papers chronologically by year')
    chrono_parser.add_argument('--start-year', type=int,
                              help='Starting year')
    chrono_parser.add_argument('--end-year', type=int,
                              help='Ending year')
    chrono_parser.add_argument('--papers-per-year', type=int,
                              help='Papers per year')
    chrono_parser.add_argument('--analyze', action='store_true',
                              help='Analyze collection after gathering')
    
    # Topic-based gathering
    topic_parser = subparsers.add_parser('topics',
                                        help='Gather papers by topics')
    topic_parser.add_argument('--topics', nargs='+',
                             help='Topic names from config')
    topic_parser.add_argument('--papers-per-topic', type=int,
                             help='Papers per topic')
    topic_parser.add_argument('--start-year', type=int,
                             help='Filter by start year')
    topic_parser.add_argument('--end-year', type=int,
                             help='Filter by end year')
    topic_parser.add_argument('--analyze', action='store_true',
                             help='Analyze collection after gathering')
    
    # List available topics
    list_parser = subparsers.add_parser('list-topics',
                                       help='List available search topics')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.command == 'list-topics':
        # Load and display topics
        topics_file = Path(__file__).parent.parent / 'gathering/config/search_topics.yaml'
        if topics_file.exists():
            with open(topics_file, 'r') as f:
                topics = yaml.safe_load(f)
            print("\nAvailable search topics:")
            for topic, info in topics.items():
                print(f"\n{topic}:")
                print(f"  Description: {info['description']}")
                print(f"  Categories: {', '.join(info['categories'])}")
                print(f"  Queries: {len(info.get('queries', []))} defined")
                print(f"  Keywords: {len(info.get('keywords', []))} defined")
        return
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'chronological':
        gather_chronological(args, config)
    elif args.command == 'topics':
        gather_topics(args, config)


if __name__ == "__main__":
    main()