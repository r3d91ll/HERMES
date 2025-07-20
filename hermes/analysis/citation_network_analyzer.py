#!/usr/bin/env python3
"""
Citation Network Analyzer for measuring conveyance through citation patterns.
Profiles authors, institutions, and citation flows in ML research.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict, Counter
import time
import requests
from dataclasses import dataclass, asdict, field
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Author:
    """Author profile with publication and citation metrics."""
    name: str
    semantic_scholar_id: Optional[str] = None
    affiliations: List[str] = field(default_factory=list)
    h_index: Optional[int] = None
    citation_count: int = 0
    paper_count: int = 0
    papers: List[str] = field(default_factory=list)  # Paper IDs
    coauthors: Set[str] = field(default_factory=set)
    research_areas: List[str] = field(default_factory=list)
    yearly_citations: Dict[int, int] = field(default_factory=dict)
    conveyance_score: float = 0.0


@dataclass
class Institution:
    """Institution profile."""
    name: str
    country: Optional[str] = None
    authors: Set[str] = field(default_factory=set)
    papers: List[str] = field(default_factory=list)
    total_citations: int = 0
    avg_conveyance: float = 0.0


@dataclass
class Citation:
    """Citation relationship between papers."""
    citing_paper_id: str
    cited_paper_id: str
    citing_paper_title: str
    cited_paper_title: str
    citation_context: Optional[str] = None
    citation_year: int
    self_citation: bool = False
    citation_velocity: float = 0.0  # How quickly after publication


@dataclass 
class PaperWithCitations:
    """Paper with full citation network data."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: Optional[str] = None
    
    # Citations
    citations_to: List[str] = field(default_factory=list)  # Papers this cites
    citations_from: List[str] = field(default_factory=list)  # Papers citing this
    citation_count: int = 0
    citation_velocity: float = 0.0  # Citations per year
    
    # Network metrics
    pagerank: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Conveyance indicators
    benchmark_improvement: Optional[float] = None
    has_code: bool = False
    implementation_count: int = 0
    industry_citations: int = 0


class CitationNetworkAnalyzer:
    """
    Analyze citation networks to measure conveyance flow.
    """
    
    def __init__(self):
        self.authors: Dict[str, Author] = {}
        self.institutions: Dict[str, Institution] = {}
        self.papers: Dict[str, PaperWithCitations] = {}
        self.citations: List[Citation] = []
        self.citation_graph = nx.DiGraph()
        
    def get_paper_citations(self, arxiv_id: str) -> Dict[str, any]:
        """Get citation data for a paper using Semantic Scholar API."""
        try:
            # Get paper details with citations
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
            params = {
                "fields": "paperId,title,authors,year,venue,citationCount,references,citations"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get citations for {arxiv_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching citations for {arxiv_id}: {e}")
            return None
    
    def get_author_profile(self, author_name: str) -> Optional[Author]:
        """Get detailed author profile from Semantic Scholar."""
        try:
            # Search for author
            search_url = "https://api.semanticscholar.org/graph/v1/author/search"
            params = {"query": author_name, "fields": "name,papers"}
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    author_data = data["data"][0]
                    
                    # Get detailed author info
                    author_id = author_data.get("authorId")
                    detail_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
                    detail_params = {
                        "fields": "name,affiliations,paperCount,citationCount,hIndex,papers.title"
                    }
                    
                    detail_response = requests.get(detail_url, params=detail_params, timeout=10)
                    
                    if detail_response.status_code == 200:
                        details = detail_response.json()
                        
                        author = Author(
                            name=details.get("name", author_name),
                            semantic_scholar_id=author_id,
                            h_index=details.get("hIndex"),
                            citation_count=details.get("citationCount", 0),
                            paper_count=details.get("paperCount", 0)
                        )
                        
                        # Get affiliations
                        affiliations = details.get("affiliations", [])
                        if affiliations:
                            author.affiliations = [aff.get("name", "") for aff in affiliations]
                            
                        return author
                        
        except Exception as e:
            logger.error(f"Error getting author profile for {author_name}: {e}")
            
        return None
    
    def analyze_paper_network(self, paper_data: Dict) -> PaperWithCitations:
        """Analyze a paper's citation network."""
        paper_id = paper_data.get("paperId", "")
        
        paper = PaperWithCitations(
            paper_id=paper_id,
            title=paper_data.get("title", ""),
            authors=[a.get("name", "") for a in paper_data.get("authors", [])],
            year=paper_data.get("year", 0),
            venue=paper_data.get("venue"),
            citation_count=paper_data.get("citationCount", 0)
        )
        
        # Calculate citation velocity
        if paper.year > 0:
            years_since = max(1, 2024 - paper.year)
            paper.citation_velocity = paper.citation_count / years_since
            
        # Get references (papers this cites)
        references = paper_data.get("references", [])
        for ref in references[:50]:  # Limit to avoid huge networks
            if ref and ref.get("paperId"):
                paper.citations_to.append(ref["paperId"])
                
        # Get citations (papers citing this)
        citations = paper_data.get("citations", [])
        for cite in citations[:100]:  # Limit for API rate limits
            if cite and cite.get("paperId"):
                paper.citations_from.append(cite["paperId"])
                
        return paper
    
    def build_citation_network(self, papers: List[Dict]) -> nx.DiGraph:
        """Build citation network graph from papers."""
        G = nx.DiGraph()
        
        # Add all papers as nodes
        for paper_data in papers:
            paper = self.analyze_paper_network(paper_data)
            self.papers[paper.paper_id] = paper
            
            G.add_node(
                paper.paper_id,
                title=paper.title,
                year=paper.year,
                citations=paper.citation_count,
                velocity=paper.citation_velocity
            )
            
        # Add citation edges
        for paper_id, paper in self.papers.items():
            # Add edges for papers this cites
            for cited_id in paper.citations_to:
                if cited_id in self.papers:
                    G.add_edge(paper_id, cited_id, type="cites")
                    
            # Add edges for papers citing this
            for citing_id in paper.citations_from:
                if citing_id in self.papers:
                    G.add_edge(citing_id, paper_id, type="cites")
                    
        self.citation_graph = G
        return G
    
    def calculate_network_metrics(self):
        """Calculate network centrality metrics for all papers."""
        if not self.citation_graph:
            return
            
        # PageRank - importance in citation network
        pagerank = nx.pagerank(self.citation_graph)
        for paper_id, score in pagerank.items():
            if paper_id in self.papers:
                self.papers[paper_id].pagerank = score
                
        # Betweenness centrality - bridge papers
        betweenness = nx.betweenness_centrality(self.citation_graph)
        for paper_id, score in betweenness.items():
            if paper_id in self.papers:
                self.papers[paper_id].betweenness_centrality = score
                
        # Clustering coefficient - local density
        clustering = nx.clustering(self.citation_graph)
        for paper_id, score in clustering.items():
            if paper_id in self.papers:
                self.papers[paper_id].clustering_coefficient = score
    
    def profile_authors_from_papers(self, papers: List[Dict]):
        """Build author profiles from paper data."""
        author_papers = defaultdict(list)
        author_coauthors = defaultdict(set)
        
        # Group papers by author
        for paper_data in papers:
            paper_id = paper_data.get("paperId", "")
            authors = paper_data.get("authors", [])
            
            author_names = [a.get("name", "") for a in authors]
            
            for i, author in enumerate(authors):
                name = author.get("name", "")
                if name:
                    author_papers[name].append(paper_id)
                    
                    # Track coauthors
                    for j, coauthor in enumerate(author_names):
                        if i != j:
                            author_coauthors[name].add(coauthor)
                            
        # Create author profiles
        for author_name, paper_ids in author_papers.items():
            if author_name not in self.authors:
                # Try to get detailed profile
                author = self.get_author_profile(author_name)
                if not author:
                    author = Author(name=author_name)
                    
                author.papers = paper_ids
                author.coauthors = author_coauthors[author_name]
                
                # Calculate author conveyance score
                author_conveyance = []
                for paper_id in paper_ids:
                    if paper_id in self.papers:
                        paper = self.papers[paper_id]
                        # Simple conveyance based on citation velocity
                        author_conveyance.append(paper.citation_velocity)
                        
                if author_conveyance:
                    author.conveyance_score = sum(author_conveyance) / len(author_conveyance)
                    
                self.authors[author_name] = author
                
    def find_high_conveyance_authors(self, top_n: int = 50) -> List[Author]:
        """Find authors with highest conveyance scores."""
        authors_with_scores = [
            (author, author.conveyance_score) 
            for author in self.authors.values()
            if author.paper_count > 2  # Min papers for reliability
        ]
        
        authors_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [author for author, _ in authors_with_scores[:top_n]]
    
    def analyze_institution_patterns(self):
        """Analyze conveyance patterns by institution."""
        institution_authors = defaultdict(set)
        institution_papers = defaultdict(list)
        
        for author_name, author in self.authors.items():
            for affiliation in author.affiliations:
                if affiliation:
                    institution_authors[affiliation].add(author_name)
                    institution_papers[affiliation].extend(author.papers)
                    
        # Create institution profiles
        for inst_name, authors in institution_authors.items():
            inst = Institution(
                name=inst_name,
                authors=authors,
                papers=list(set(institution_papers[inst_name]))
            )
            
            # Calculate institution metrics
            total_citations = 0
            conveyance_scores = []
            
            for author_name in authors:
                if author_name in self.authors:
                    author = self.authors[author_name]
                    total_citations += author.citation_count
                    conveyance_scores.append(author.conveyance_score)
                    
            inst.total_citations = total_citations
            if conveyance_scores:
                inst.avg_conveyance = sum(conveyance_scores) / len(conveyance_scores)
                
            self.institutions[inst_name] = inst
    
    def find_conveyance_communities(self) -> List[Set[str]]:
        """Find communities of high-conveyance collaboration."""
        # Create coauthor network
        coauthor_graph = nx.Graph()
        
        for author_name, author in self.authors.items():
            coauthor_graph.add_node(author_name, conveyance=author.conveyance_score)
            
            for coauthor in author.coauthors:
                if coauthor in self.authors:
                    coauthor_graph.add_edge(author_name, coauthor)
                    
        # Find communities
        communities = list(nx.community.greedy_modularity_communities(coauthor_graph))
        
        # Sort by average conveyance
        community_scores = []
        for community in communities:
            scores = [
                self.authors[author].conveyance_score 
                for author in community 
                if author in self.authors
            ]
            avg_score = sum(scores) / len(scores) if scores else 0
            community_scores.append((community, avg_score))
            
        community_scores.sort(key=lambda x: x[1], reverse=True)
        return [community for community, _ in community_scores]
    
    def generate_network_report(self, output_file: str = "citation_network_analysis.md"):
        """Generate comprehensive citation network report."""
        
        report = f"""# Citation Network Analysis: Conveyance Through Citations

Generated: {datetime.now().isoformat()}

## Network Overview

- Total Papers: {len(self.papers)}
- Total Authors: {len(self.authors)}
- Total Institutions: {len(self.institutions)}
- Citation Edges: {self.citation_graph.number_of_edges()}

## High Conveyance Authors (Top 20)

Authors whose papers spread rapidly through the network:

"""
        
        top_authors = self.find_high_conveyance_authors(20)
        
        for i, author in enumerate(top_authors, 1):
            report += f"""{i}. **{author.name}**
   - Conveyance Score: {author.conveyance_score:.3f}
   - Papers: {author.paper_count}
   - Citations: {author.citation_count:,}
   - H-Index: {author.h_index or 'N/A'}
   - Affiliations: {', '.join(author.affiliations[:2]) if author.affiliations else 'Unknown'}

"""
        
        # Top institutions
        report += "\n## High Conveyance Institutions\n\n"
        
        top_institutions = sorted(
            self.institutions.values(),
            key=lambda x: x.avg_conveyance,
            reverse=True
        )[:10]
        
        for inst in top_institutions:
            report += f"""- **{inst.name}**
  - Avg Conveyance: {inst.avg_conveyance:.3f}
  - Authors: {len(inst.authors)}
  - Total Citations: {inst.total_citations:,}

"""
        
        # Most cited papers (high conveyance)
        report += "\n## Highest Velocity Papers\n\n"
        
        high_velocity = sorted(
            self.papers.values(),
            key=lambda x: x.citation_velocity,
            reverse=True
        )[:10]
        
        for paper in high_velocity:
            report += f"""- **{paper.title}**
  - Citation Velocity: {paper.citation_velocity:.1f}/year
  - Total Citations: {paper.citation_count}
  - Year: {paper.year}
  - PageRank: {paper.pagerank:.4f}

"""
        
        # Bridge papers (high betweenness)
        report += "\n## Bridge Papers (Connecting Communities)\n\n"
        
        bridge_papers = sorted(
            self.papers.values(),
            key=lambda x: x.betweenness_centrality,
            reverse=True
        )[:10]
        
        for paper in bridge_papers:
            report += f"""- **{paper.title}**
  - Betweenness: {paper.betweenness_centrality:.4f}
  - Connects different research areas

"""
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Network report saved to {output_file}")
        
        # Save network data
        self.save_network_data()
    
    def save_network_data(self):
        """Save network data for further analysis."""
        # Save graph
        nx.write_gexf(self.citation_graph, "citation_network.gexf")
        
        # Save detailed data
        network_data = {
            "authors": {name: asdict(author) for name, author in self.authors.items()},
            "institutions": {name: asdict(inst) for name, inst in self.institutions.items()},
            "papers": {pid: asdict(paper) for pid, paper in self.papers.items()}
        }
        
        with open("citation_network_data.json", 'w', encoding='utf-8') as f:
            json.dump(network_data, f, indent=2, default=str)
            
        logger.info("Network data saved to citation_network.gexf and citation_network_data.json")


def main():
    """Run citation network analysis."""
    analyzer = CitationNetworkAnalyzer()
    
    logger.info("Starting citation network analysis...")
    
    # This would be integrated with the ML paper finder
    # For demo, showing the structure
    
    print("""
Citation Network Analyzer Ready!

This will:
1. Get citation data for each paper
2. Build author profiles with h-index, affiliations
3. Create institution rankings by conveyance
4. Find high-conveyance research communities
5. Identify bridge papers connecting fields

Integration points:
- Use with ML paper finder to get citation networks
- Track how benchmark improvements spread through citations
- Measure time from paper → citation → implementation
- Find which authors/institutions drive high conveyance
""")


if __name__ == "__main__":
    main()