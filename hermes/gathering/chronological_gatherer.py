"""
Chronological paper gatherer for temporal analysis.
"""

import arxiv
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time
from tqdm import tqdm

from .base_gatherer import BaseGatherer

logger = logging.getLogger(__name__)


class ChronologicalGatherer(BaseGatherer):
    """Gather papers chronologically by year."""
    
    def __init__(self, output_dir: Path, config: Dict = None):
        super().__init__(output_dir, config)
        
        # Default ML categories
        self.ml_categories = self.config.get('categories', [
            "cs.LG",  # Machine Learning
            "cs.CL",  # Computation and Language
            "cs.CV",  # Computer Vision
            "cs.AI",  # Artificial Intelligence
            "cs.IT",  # Information Theory
            "math.IT", # Information Theory (Math)
            "stat.ML"  # Machine Learning (Statistics)
        ])
    
    def gather_papers_by_year(self,
                            start_year: int = 1998,
                            end_year: Optional[int] = None,
                            papers_per_year: int = 50) -> List[arxiv.Result]:
        """Gather papers chronologically by year."""
        if end_year is None:
            end_year = datetime.now().year
        
        logger.info(f"Gathering papers from {start_year} to {end_year}, {papers_per_year} per year")
        
        all_papers = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"\n=== Gathering papers from {year} ===")
            
            year_papers = self._gather_year_papers(year, papers_per_year)
            all_papers.extend(year_papers)
            
            # Small delay between years
            time.sleep(2)
        
        return all_papers
    
    def _gather_year_papers(self, year: int, max_papers: int) -> List[arxiv.Result]:
        """Gather papers from a specific year."""
        year_papers = []
        papers_per_category = max(10, max_papers // len(self.ml_categories))
        
        for category in self.ml_categories:
            try:
                # Build query for this year and category
                query = f'cat:{category} AND submittedDate:[{year}0101 TO {year}1231]'
                
                search = arxiv.Search(
                    query=query,
                    max_results=papers_per_category,
                    sort_by=arxiv.SortCriterion.Relevance,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                for result in search.results():
                    if result.published.year == year:
                        year_papers.append(result)
                        
            except Exception as e:
                logger.warning(f"Error searching {category} for year {year}: {e}")
        
        # Sort by relevance proxy (comment length + abstract length)
        year_papers = sorted(
            year_papers,
            key=lambda p: len(p.comment or "") + len(p.summary),
            reverse=True
        )[:max_papers]
        
        logger.info(f"Found {len(year_papers)} papers from {year}")
        return year_papers
    
    def gather_papers(self,
                     start_year: int = 1998,
                     end_year: Optional[int] = None,
                     papers_per_year: int = 50,
                     **kwargs) -> Dict[str, Dict]:
        """
        Main method to gather papers chronologically.
        
        Args:
            start_year: Starting year
            end_year: Ending year (None for current year)
            papers_per_year: Papers to gather per year
        """
        # Gather papers
        papers = self.gather_papers_by_year(start_year, end_year, papers_per_year)
        
        # Download papers
        logger.info(f"\nTotal papers to download: {len(papers)}")
        downloaded = 0
        
        for i, paper in enumerate(tqdm(papers, desc="Downloading papers")):
            if self.download_paper(paper):
                downloaded += 1
                self.save_metadata()
                
                # Extra delay every batch_size papers
                if downloaded % self.batch_size == 0 and downloaded > 0:
                    logger.info(f"Downloaded {downloaded} papers, taking a {self.batch_delay}s break...")
                    time.sleep(self.batch_delay)
        
        logger.info(f"\nSuccessfully downloaded {downloaded} papers")
        return self.metadata