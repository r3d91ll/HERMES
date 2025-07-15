#!/usr/bin/env python3
"""
ArXiv Paper Downloader for HADES Bibliography
Downloads papers from arXiv mentioned in the theory bibliography
"""

import arxiv
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
from urllib.parse import urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivPaperDownloader:
    """Download papers from arXiv based on bibliography entries"""
    
    def __init__(self, base_dir: str = "/home/todd/ML-Lab/Olympus/HADES/docs/theory/papers"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.request_delay = 3  # seconds between requests
        
        # ArXiv client with rate limiting
        self.client = arxiv.Client(
            page_size=10,
            delay_seconds=3,
            num_retries=3
        )
        
    def extract_arxiv_papers(self, bibliography_text: str) -> List[Dict[str, str]]:
        """Extract arXiv papers from bibliography text"""
        
        arxiv_papers = []
        
        # Pattern for arXiv papers in bibliography
        arxiv_pattern = r'arXiv:(\d{4}\.\d{4,5})'
        
        lines = bibliography_text.split('\n')
        current_paper = None
        
        for line in lines:
            # Check if line contains an arXiv reference
            arxiv_match = re.search(arxiv_pattern, line)
            
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                
                # Try to extract paper info from the line
                # Pattern for author and title
                author_title_pattern = r'\*\*(.*?)\*\*.*?"(.*?)"'
                match = re.search(author_title_pattern, line)
                
                if match:
                    authors = match.group(1).strip()
                    title = match.group(2).strip()
                else:
                    # Fallback - just use the whole line for context
                    authors = "Unknown"
                    title = line.strip()
                
                paper_info = {
                    'arxiv_id': arxiv_id,
                    'authors': authors,
                    'title': title,
                    'line': line.strip()
                }
                
                arxiv_papers.append(paper_info)
                logger.info(f"Found arXiv paper: {arxiv_id} - {title[:50]}...")
        
        return arxiv_papers
    
    def categorize_paper(self, paper_info: Dict[str, str], bibliography_text: str) -> str:
        """Determine which theoretical category a paper belongs to"""
        
        line = paper_info['line'].lower()
        title = paper_info['title'].lower()
        
        # Find the section this paper appears in
        sections = [
            ('context_amplification', 'context as exponential amplifier'),
            ('information_transformation', 'information as transformation'),
            ('observer_relativity', 'observer-relative information'),
            ('boundary_actors', 'boundary actors, gatekeepers'),
            ('entropy_communication', 'entropy generation in communication'),
            ('multidimensional_models', 'multi-dimensional information models'),
            ('asymmetric_flows', 'asymmetric information flows'),
            ('semantic_similarity', 'semantic similarity through metadata'),
            ('transformer_architectures', 'transformer architectures'),
            ('critical_perspectives', 'critical and interdisciplinary')
        ]
        
        # Find which section the paper appears in
        paper_index = bibliography_text.lower().find(paper_info['arxiv_id'])
        
        best_category = 'general'
        best_distance = float('inf')
        
        for category, section_title in sections:
            section_index = bibliography_text.lower().find(section_title)
            if section_index != -1 and paper_index > section_index:
                distance = paper_index - section_index
                if distance < best_distance:
                    best_distance = distance
                    best_category = category
        
        return best_category
    
    def download_paper(self, arxiv_id: str, category: str, paper_info: Dict[str, str]) -> bool:
        """Download a single paper from arXiv"""
        
        try:
            # Create category directory
            category_dir = self.base_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Search for the paper
            search = arxiv.Search(id_list=[arxiv_id])
            
            # Get the paper
            paper = next(self.client.results(search))
            
            # Create filename
            safe_title = re.sub(r'[^\w\s-]', '', paper.title)
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            filename = f"{arxiv_id}_{safe_title[:50]}.pdf"
            filepath = category_dir / filename
            
            # Download the paper
            logger.info(f"Downloading {arxiv_id} to {category}/{filename}")
            paper.download_pdf(dirpath=str(category_dir), filename=filename)
            
            # Create metadata file
            metadata_file = category_dir / f"{arxiv_id}_metadata.txt"
            with open(metadata_file, 'w') as f:
                f.write(f"ArXiv ID: {paper.entry_id}\n")
                f.write(f"Title: {paper.title}\n")
                f.write(f"Authors: {', '.join([str(author) for author in paper.authors])}\n")
                f.write(f"Published: {paper.published}\n")
                f.write(f"Updated: {paper.updated}\n")
                f.write(f"Summary: {paper.summary}\n")
                f.write(f"Categories: {', '.join(paper.categories)}\n")
                f.write(f"Bibliography Context: {paper_info['line']}\n")
            
            time.sleep(self.request_delay)  # Rate limiting
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {arxiv_id}: {str(e)}")
            return False
    
    def download_all_papers(self, bibliography_path: str) -> Dict[str, List[str]]:
        """Download all arXiv papers from bibliography"""
        
        # Read bibliography
        with open(bibliography_path, 'r') as f:
            bibliography_text = f.read()
        
        # Extract arXiv papers
        arxiv_papers = self.extract_arxiv_papers(bibliography_text)
        logger.info(f"Found {len(arxiv_papers)} arXiv papers in bibliography")
        
        # Download results
        results = {
            'downloaded': [],
            'failed': [],
            'categories': {}
        }
        
        for paper_info in arxiv_papers:
            arxiv_id = paper_info['arxiv_id']
            category = self.categorize_paper(paper_info, bibliography_text)
            
            if category not in results['categories']:
                results['categories'][category] = []
            
            success = self.download_paper(arxiv_id, category, paper_info)
            
            if success:
                results['downloaded'].append(arxiv_id)
                results['categories'][category].append(arxiv_id)
                logger.info(f"✓ Downloaded: {arxiv_id} ({category})")
            else:
                results['failed'].append(arxiv_id)
                logger.error(f"✗ Failed: {arxiv_id}")
        
        # Create summary report
        self.create_download_report(results)
        
        return results
    
    def create_download_report(self, results: Dict) -> None:
        """Create a summary report of downloaded papers"""
        
        report_path = self.base_dir / "download_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# ArXiv Papers Download Report\n\n")
            f.write(f"**Total papers found**: {len(results['downloaded']) + len(results['failed'])}\n")
            f.write(f"**Successfully downloaded**: {len(results['downloaded'])}\n")
            f.write(f"**Failed downloads**: {len(results['failed'])}\n\n")
            
            f.write("## Papers by Category\n\n")
            for category, papers in results['categories'].items():
                f.write(f"### {category.replace('_', ' ').title()}\n")
                f.write(f"- **Count**: {len(papers)}\n")
                f.write(f"- **Papers**: {', '.join(papers)}\n\n")
            
            if results['failed']:
                f.write("## Failed Downloads\n\n")
                for paper_id in results['failed']:
                    f.write(f"- {paper_id}\n")
                f.write("\n")
            
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write("docs/theory/papers/\n")
            for category in results['categories'].keys():
                f.write(f"├── {category}/\n")
                f.write(f"│   ├── [arxiv_id]_[title].pdf\n")
                f.write(f"│   └── [arxiv_id]_metadata.txt\n")
            f.write("└── download_report.md\n")
            f.write("```\n")

def main():
    """Main function to run the downloader"""
    
    bibliography_path = "/home/todd/ML-Lab/Olympus/HADES/docs/theory/bibliobraphy.md"
    
    if not os.path.exists(bibliography_path):
        logger.error(f"Bibliography file not found: {bibliography_path}")
        return
    
    downloader = ArxivPaperDownloader()
    
    logger.info("Starting arXiv paper download process...")
    results = downloader.download_all_papers(bibliography_path)
    
    logger.info("Download process completed!")
    logger.info(f"Downloaded: {len(results['downloaded'])} papers")
    logger.info(f"Failed: {len(results['failed'])} papers")
    
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total papers downloaded: {len(results['downloaded'])}")
    print(f"Failed downloads: {len(results['failed'])}")
    print(f"Papers organized into {len(results['categories'])} categories")
    print(f"Download location: {downloader.base_dir}")
    print("="*50)

if __name__ == "__main__":
    main()