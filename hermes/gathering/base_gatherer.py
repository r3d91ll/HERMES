"""
Base class for all paper gatherers.
"""

import arxiv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseGatherer(ABC):
    """Base class for paper gathering implementations."""
    
    def __init__(self, output_dir: Path, config: Dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Load metadata
        self.metadata_file = self.output_dir / "metadata.json"
        self.metadata = self.load_metadata()
        
        # Rate limiting config
        self.download_delay = self.config.get('download_delay', 5)
        self.batch_delay = self.config.get('batch_delay', 30)
        self.batch_size = self.config.get('batch_size', 10)
    
    def load_metadata(self) -> Dict:
        """Load existing metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def download_paper(self, paper: arxiv.Result) -> bool:
        """Download a paper and save its metadata."""
        paper_id = paper.entry_id.split('/')[-1]
        pdf_path = self.output_dir / f"{paper_id}.pdf"
        
        # Skip if already downloaded
        if pdf_path.exists() and paper_id in self.metadata:
            logger.debug(f"Already have {paper_id}")
            return True
        
        try:
            # Download PDF
            paper.download_pdf(dirpath=str(self.output_dir), filename=f"{paper_id}.pdf")
            
            # Save metadata
            self.metadata[paper_id] = self.extract_metadata(paper)
            
            logger.info(f"Downloaded: {paper.title[:60]}...")
            time.sleep(self.download_delay)  # Be polite
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {paper_id}: {e}")
            return False
    
    def extract_metadata(self, paper: arxiv.Result) -> Dict:
        """Extract metadata from paper."""
        return {
            'arxiv_id': paper.entry_id.split('/')[-1],
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'published': paper.published.isoformat(),
            'categories': paper.categories,
            'primary_category': paper.primary_category,
            'abstract': paper.summary,
            'pdf_url': paper.pdf_url,
            'doi': paper.doi,
            'journal_ref': paper.journal_ref,
            'comment': paper.comment,
            'downloaded_at': datetime.now().isoformat()
        }
    
    def download_batch(self, papers: List[arxiv.Result]) -> int:
        """Download a batch of papers with rate limiting."""
        downloaded = 0
        
        for i, paper in enumerate(papers):
            if self.download_paper(paper):
                downloaded += 1
                self.save_metadata()
                
                # Extra delay every batch_size papers
                if downloaded % self.batch_size == 0 and downloaded > 0:
                    logger.info(f"Downloaded {downloaded} papers, taking a {self.batch_delay}s break...")
                    time.sleep(self.batch_delay)
        
        return downloaded
    
    @abstractmethod
    def gather_papers(self, **kwargs) -> Dict[str, Dict]:
        """Gather papers. Must be implemented by subclasses."""
        pass
    
    def analyze_collection(self):
        """Analyze the downloaded collection."""
        if not self.metadata:
            logger.warning("No metadata to analyze")
            return
        
        # Year distribution
        years = {}
        categories = {}
        
        for paper_data in self.metadata.values():
            # Year
            year = datetime.fromisoformat(paper_data['published']).year
            years[year] = years.get(year, 0) + 1
            
            # Categories
            for cat in paper_data['categories']:
                categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n=== Collection Analysis ===")
        print(f"Total papers: {len(self.metadata)}")
        
        print("\nYear distribution:")
        for year in sorted(years.keys()):
            print(f"  {year}: {years[year]} papers")
        
        print("\nTop categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cat}: {count} papers")