#!/usr/bin/env python3
"""
Organize papers using Library of Congress Classification (LCC) system.
Packs maximum spatial information into the WHERE dimension through systematic naming.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional
import re
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LCCOrganizer:
    """
    Organize papers using Library of Congress Classification.
    
    ETHNOGRAPHIC NOTE:
    This is a demonstration of how anthropological theory translates to computational practice
    without appeals to meaninglessness or nihilism. Following Foucault's "The Order of Things",
    we create an epistemic architecture where:
    
    1. The directory hierarchy embodies discursive formations (not arbitrary categories)
    2. ISNE learns the topology of knowledge through spatial proximity (not labels)
    3. The 68-dimensional WHERE embedding captures relational meaning through locality
    
    The trick: ISNE doesn't need to "understand" that QA76.9.I52 means "Information Theory".
    Instead, it learns that papers in this spatial region share citation patterns, vocabulary,
    and semantic relationships. Meaning emerges from structure, not from external definition.
    
    This grounds abstract anthropological concepts (episteme, discourse, archive) in actual
    computational practice. We're not theorizing about knowledge organization - we're building
    a system where machine learning literally navigates Foucault's "archaeology of knowledge"
    through the physical arrangement of files.
    
    The filesystem becomes a material instantiation of conceptual space, proving that
    anthropological insights about meaning, power, and knowledge can directly inform
    information retrieval systems. No postmodern hand-waving required - just good engineering
    informed by deep theory.
    """
    
    def __init__(self, base_dir: Path = Path("/home/todd/olympus/data/papers")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # LCC main classes relevant to our research
        self.lcc_classes = {
            # Science (Q)
            "QA": "Mathematics",
            "QA75-76.95": "Computer Science",
            "QA76.9.A43": "Algorithms",
            "QA76.9.D3": "Data Processing",
            "QA76.9.I52": "Information Theory",
            "QA76.87": "Neural Networks",
            "QA267-268.5": "Machine Theory, Abstract Machines",
            "QA402.5-402.6": "Control Theory, Optimization",
            
            # Technology (T)
            "T57-57.97": "Applied Mathematics, Operations Research",
            "T58.5-58.64": "Information Technology",
            
            # Philosophy, Psychology (B)
            "B51-53": "Logic",
            "BC": "Logic (General)",
            "BD": "Speculative Philosophy",
            "BF": "Psychology",
            "BF311-499": "Cognition, Memory, Intelligence",
            
            # Language and Literature (P)
            "P87-96": "Communication, Mass Media",
            "P98-98.5": "Computational Linguistics",
            "P118-118.7": "Language Acquisition",
            "P325-325.5": "Semantics",
            
            # Social Sciences (H)
            "HM": "Sociology (General)",
            "HM1001-1281": "Social Psychology",
            
            # Anthropology (GN)
            "GN": "Anthropology (General)",
            "GN301-674": "Ethnology, Social and Cultural Anthropology",
            "GN476-477": "Cognitive Anthropology",
            "GN495-496": "Linguistic Anthropology",
            
            # Library Science (Z)
            "Z665-718.8": "Library Science, Information Science",
            "Z699.5-699.5.3": "Information Storage and Retrieval"
        }
        
        # Domain to LCC mapping
        self.domain_lcc_map = {
            # Computer Science / AI / ML
            "machine_learning": "QA76.9.A43",
            "neural_networks": "QA76.87",
            "algorithms": "QA76.9.A43",
            "information_retrieval": "Z699.5",
            "natural_language_processing": "P98",
            "computer_vision": "QA76.9.D3",
            
            # Information Theory
            "information_theory": "QA76.9.I52",
            "entropy": "QA76.9.I52",
            "communication_theory": "P87",
            
            # Anthropology
            "anthropology": "GN",
            "cultural_anthropology": "GN301",
            "cognitive_anthropology": "GN476",
            "linguistic_anthropology": "GN495",
            
            # Cognitive Science
            "cognitive_science": "BF311",
            "psychology": "BF",
            "linguistics": "P118",
            "semantics": "P325"
        }
    
    def classify_paper(self, metadata: Dict) -> Tuple[str, str]:
        """
        Classify a paper and return LCC code and subclass.
        
        Returns:
            (lcc_code, subclass_name)
        """
        title = metadata.get('title', '').lower()
        abstract = metadata.get('abstract', '').lower()
        categories = metadata.get('categories', [])
        
        # Check arXiv categories first
        if any(cat.startswith('cs.LG') for cat in categories):
            return "QA76.9.A43", "Machine Learning"
        elif any(cat.startswith('cs.CL') for cat in categories):
            return "P98", "Computational Linguistics"
        elif any(cat.startswith('cs.CV') for cat in categories):
            return "QA76.9.D3", "Computer Vision"
        elif any(cat.startswith('cs.AI') for cat in categories):
            return "QA76.87", "Artificial Intelligence"
        elif any(cat.startswith('stat.ML') for cat in categories):
            return "QA76.9.A43", "Statistical Machine Learning"
        
        # Check for specific domains in title/abstract
        text = f"{title} {abstract}"
        
        # Information theory keywords
        if any(term in text for term in ["shannon", "entropy", "information theory", "mutual information"]):
            return "QA76.9.I52", "Information Theory"
        
        # Anthropology keywords
        if any(term in text for term in ["anthropolog", "ethnograph", "cultural", "society", "social"]):
            if "cogniti" in text:
                return "GN476", "Cognitive Anthropology"
            elif "linguistic" in text or "language" in text:
                return "GN495", "Linguistic Anthropology"
            else:
                return "GN301", "Cultural Anthropology"
        
        # Default to general CS
        return "QA76.9", "Computer Science General"
    
    def generate_lcc_filename(self, metadata: Dict, original_path: Path) -> str:
        """
        Generate LCC-based filename with maximum information packing.
        
        Format: LCC_YYYY_AuthorInitials_KeywordAcronym_OriginalID.pdf
        Example: QA76.9.A43_2023_LeCun_Y_CNN_2305.14239.pdf
        """
        lcc_code, _ = self.classify_paper(metadata)
        
        # Extract year
        published = metadata.get('published', '')
        year = published[:4] if published else "0000"
        
        # Extract author initials (first author)
        authors = metadata.get('authors', [])
        if authors:
            first_author = authors[0]
            # Handle "LastName, FirstName" or "FirstName LastName"
            if ',' in first_author:
                last, first = first_author.split(',', 1)
            else:
                parts = first_author.strip().split()
                if parts:
                    first = parts[0]
                    last = parts[-1] if len(parts) > 1 else ""
                else:
                    first = last = ""
            
            # Create initials
            first_initial = first.strip()[0].upper() if first.strip() else ""
            last_name = ''.join(c for c in last.strip() if c.isalnum())[:15]  # Max 15 chars
            author_code = f"{last_name}_{first_initial}" if last_name else "Unknown"
        else:
            author_code = "Unknown"
        
        # Extract keyword acronym from title
        title = metadata.get('title', '')
        # Common stop words to exclude
        stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'and', 'or'}
        title_words = [w for w in title.split() if w.lower() not in stop_words]
        
        # Create acronym from significant words (max 4 words)
        if title_words:
            acronym_words = []
            for word in title_words[:4]:
                # Extract capital letters or first letter
                caps = ''.join(c for c in word if c.isupper())
                if caps:
                    acronym_words.append(caps)
                elif word:
                    acronym_words.append(word[0].upper())
            keyword_acronym = ''.join(acronym_words)[:8]  # Max 8 chars
        else:
            keyword_acronym = "PAPER"
        
        # Original ID (arxiv ID or similar)
        original_id = original_path.stem
        if original_id.startswith('arxiv_'):
            original_id = original_id[6:]
        original_id = original_id.replace('.', '_')[:20]  # Max 20 chars
        
        # Combine all parts
        filename = f"{lcc_code}_{year}_{author_code}_{keyword_acronym}_{original_id}.pdf"
        
        # Clean filename
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        
        return filename
    
    def organize_papers(self, source_dir: Path, metadata_file: Optional[Path] = None):
        """
        Organize papers from source directory into LCC structure.
        """
        if metadata_file and metadata_file.exists():
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}
        
        # Create LCC directory structure
        lcc_dirs = {}
        for code in set(code.split('-')[0] for code in self.lcc_classes.keys()):
            main_class = code[0]  # Q, T, B, P, H, G, Z
            class_dir = self.base_dir / main_class / code
            class_dir.mkdir(parents=True, exist_ok=True)
            lcc_dirs[code] = class_dir
        
        # Process papers
        organized_count = 0
        organization_log = []
        
        for pdf_file in source_dir.glob("*.pdf"):
            # Get metadata
            paper_id = pdf_file.stem
            metadata = all_metadata.get(paper_id, {})
            
            if not metadata:
                # Try to infer from filename
                metadata = {'title': paper_id, 'authors': [], 'published': datetime.now().isoformat()}
            
            # Classify and generate new name
            lcc_code, subclass = self.classify_paper(metadata)
            new_filename = self.generate_lcc_filename(metadata, pdf_file)
            
            # Determine destination
            main_code = lcc_code.split('.')[0]
            if main_code in lcc_dirs:
                dest_dir = lcc_dirs[main_code]
            else:
                # Find closest match
                for code in lcc_dirs:
                    if lcc_code.startswith(code):
                        dest_dir = lcc_dirs[code]
                        break
                else:
                    dest_dir = self.base_dir / "QA" / "QA76.9"  # Default to CS
            
            dest_path = dest_dir / new_filename
            
            # Copy file
            shutil.copy2(pdf_file, dest_path)
            organized_count += 1
            
            # Log the organization
            organization_log.append({
                'original': str(pdf_file),
                'lcc_path': str(dest_path),
                'lcc_code': lcc_code,
                'subclass': subclass,
                'metadata': metadata
            })
            
            logger.info(f"Organized: {pdf_file.name} -> {dest_path.relative_to(self.base_dir)}")
        
        # Save organization log
        log_file = self.base_dir / "lcc_organization_log.json"
        with open(log_file, 'w') as f:
            json.dump(organization_log, f, indent=2)
        
        logger.info(f"Organized {organized_count} papers into LCC structure")
        logger.info(f"Organization log saved to {log_file}")
        
        # Create index
        self.create_lcc_index()
    
    def create_lcc_index(self):
        """Create an index of all papers with their LCC classifications."""
        index = {
            'by_class': {},
            'by_year': {},
            'by_author': {},
            'total_papers': 0
        }
        
        for class_dir in self.base_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
                
            for subclass_dir in class_dir.iterdir():
                if not subclass_dir.is_dir():
                    continue
                    
                lcc_code = subclass_dir.name
                papers = list(subclass_dir.glob("*.pdf"))
                
                if papers:
                    index['by_class'][lcc_code] = {
                        'description': self.lcc_classes.get(lcc_code, "Unknown"),
                        'count': len(papers),
                        'papers': [p.name for p in papers]
                    }
                    
                    for paper in papers:
                        index['total_papers'] += 1
                        
                        # Parse filename
                        parts = paper.stem.split('_')
                        if len(parts) >= 4:
                            year = parts[1]
                            author = parts[2]
                            
                            # Index by year
                            if year not in index['by_year']:
                                index['by_year'][year] = []
                            index['by_year'][year].append(paper.name)
                            
                            # Index by author
                            if author not in index['by_author']:
                                index['by_author'][author] = []
                            index['by_author'][author].append(paper.name)
        
        # Save index
        index_file = self.base_dir / "lcc_index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Created LCC index with {index['total_papers']} papers")
        
        # Print summary
        print("\n=== LCC Organization Summary ===")
        print(f"Total papers: {index['total_papers']}")
        print("\nPapers by classification:")
        for lcc_code, info in sorted(index['by_class'].items()):
            print(f"  {lcc_code}: {info['count']} papers - {info['description']}")
        
        print("\nPapers by year:")
        for year in sorted(index['by_year'].keys()):
            print(f"  {year}: {len(index['by_year'][year])} papers")


def main():
    """Organize papers using LCC system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize papers using Library of Congress Classification")
    parser.add_argument("--source", type=Path, default=Path("./data/ml_papers"),
                       help="Source directory containing papers")
    parser.add_argument("--metadata", type=Path, 
                       help="Metadata JSON file (e.g., arxiv_metadata.json)")
    parser.add_argument("--base-dir", type=Path, default=Path("/home/todd/olympus/data/papers"),
                       help="Base directory for organized papers")
    
    args = parser.parse_args()
    
    if not args.source.exists():
        print(f"Source directory {args.source} does not exist!")
        return
    
    # Default metadata location
    if not args.metadata:
        args.metadata = args.source / "arxiv_metadata.json"
    
    organizer = LCCOrganizer(base_dir=args.base_dir)
    
    print(f"""
Library of Congress Classification Organizer
===========================================

This will organize papers from:
  Source: {args.source}
  Metadata: {args.metadata if args.metadata.exists() else "Not found (will use filenames)"}
  
Into LCC structure at:
  {args.base_dir}

The WHERE dimension will encode:
- Library of Congress Classification (domain/subject)
- Publication year
- Author information
- Keyword acronyms
- Original identifiers

This maximizes spatial information in our naming scheme.
""")
    
    if input("Continue? (y/n): ").lower() == 'y':
        organizer.organize_papers(args.source, args.metadata)


if __name__ == "__main__":
    main()