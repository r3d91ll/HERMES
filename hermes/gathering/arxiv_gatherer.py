"""
ArXiv paper gatherer with configurable search topics.
"""

import arxiv
import logging
from pathlib import Path
from typing import List, Dict, Optional
import yaml
from tqdm import tqdm

from .base_gatherer import BaseGatherer

logger = logging.getLogger(__name__)


class ArxivGatherer(BaseGatherer):
    """Gather papers from arXiv based on configurable topics."""
    
    def __init__(self, output_dir: Path, config: Dict = None):
        super().__init__(output_dir, config)
        
        # Load search topics
        self.topics_config = self._load_topics_config()
    
    def _load_topics_config(self) -> Dict:
        """Load search topics configuration."""
        config_file = Path(__file__).parent / "config" / "search_topics.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def search_by_topic(self, 
                       topic: str,
                       max_results: int = 100,
                       start_year: Optional[int] = None,
                       end_year: Optional[int] = None) -> List[arxiv.Result]:
        """Search for papers on a specific topic."""
        if topic not in self.topics_config:
            logger.error(f"Unknown topic: {topic}")
            return []
        
        topic_config = self.topics_config[topic]
        all_papers = []
        seen_ids = set()
        
        logger.info(f"Searching {topic}: {topic_config['description']}")
        
        # Search with each query
        queries = topic_config.get('queries', [])
        for query in queries:
            try:
                # Add year filter if specified
                if start_year or end_year:
                    year_filter = self._build_year_filter(start_year, end_year)
                    query = f"{query} AND {year_filter}"
                
                search = arxiv.Search(
                    query=query,
                    max_results=max_results // len(queries) if queries else max_results,
                    sort_by=arxiv.SortCriterion.Relevance,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                for result in search.results():
                    paper_id = result.entry_id.split('/')[-1]
                    if paper_id not in seen_ids:
                        seen_ids.add(paper_id)
                        all_papers.append(result)
                        
            except Exception as e:
                logger.warning(f"Error with query {query}: {e}")
        
        logger.info(f"Found {len(all_papers)} papers for {topic}")
        return all_papers
    
    def _build_year_filter(self, start_year: Optional[int], end_year: Optional[int]) -> str:
        """Build year filter for arXiv query."""
        if start_year and end_year:
            return f"submittedDate:[{start_year}0101 TO {end_year}1231]"
        elif start_year:
            return f"submittedDate:[{start_year}0101 TO 20991231]"
        elif end_year:
            return f"submittedDate:[19000101 TO {end_year}1231]"
        return ""
    
    def gather_papers(self, 
                     topics: List[str] = None,
                     papers_per_topic: int = 50,
                     **kwargs) -> Dict[str, Dict]:
        """
        Gather papers for specified topics.
        
        Args:
            topics: List of topic names from config
            papers_per_topic: Max papers per topic
            **kwargs: Additional arguments (start_year, end_year, etc.)
        """
        if topics is None:
            topics = list(self.topics_config.keys())
        
        all_papers = []
        
        # Gather papers for each topic
        for topic in topics:
            papers = self.search_by_topic(
                topic,
                max_results=papers_per_topic,
                start_year=kwargs.get('start_year'),
                end_year=kwargs.get('end_year')
            )
            all_papers.extend(papers[:papers_per_topic])
        
        # Download papers
        logger.info(f"Downloading {len(all_papers)} papers...")
        downloaded = self.download_batch(all_papers)
        
        logger.info(f"Successfully downloaded {downloaded} papers")
        return self.metadata