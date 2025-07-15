#!/usr/bin/env python3
"""
Survey the olympus directory for documents, respecting .gitignore files.
This will help us understand the scale of available documents for HADES dataset.
"""

import os
import fnmatch
from pathlib import Path
from collections import defaultdict
import json
from typing import Set, List, Dict, Tuple


class GitignoreParser:
    """Parse and apply .gitignore rules."""
    
    def __init__(self):
        self.patterns: List[Tuple[str, str]] = []  # (pattern, gitignore_path)
        self.gitignore_files: List[str] = []
        
    def load_gitignore(self, gitignore_path: Path):
        """Load patterns from a .gitignore file."""
        if not gitignore_path.exists():
            return
            
        self.gitignore_files.append(str(gitignore_path))
        gitignore_dir = gitignore_path.parent
        
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                    
                # Store pattern with its source directory
                self.patterns.append((line, str(gitignore_dir)))
    
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on gitignore rules."""
        path_str = str(path)
        
        for pattern, gitignore_dir in self.patterns:
            # Handle directory patterns (ending with /)
            if pattern.endswith('/'):
                pattern = pattern[:-1]
                if path.is_dir():
                    if self._match_pattern(path, pattern, gitignore_dir):
                        return True
            else:
                # Check both file and directory
                if self._match_pattern(path, pattern, gitignore_dir):
                    return True
                    
        return False
    
    def _match_pattern(self, path: Path, pattern: str, gitignore_dir: str) -> bool:
        """Match a path against a gitignore pattern."""
        path_str = str(path)
        
        # Absolute patterns (starting with /)
        if pattern.startswith('/'):
            pattern = pattern[1:]
            full_pattern = os.path.join(gitignore_dir, pattern)
            return fnmatch.fnmatch(path_str, full_pattern)
        
        # Relative patterns - check if pattern matches anywhere in path
        path_parts = path_str.split(os.sep)
        for i in range(len(path_parts)):
            partial_path = os.sep.join(path_parts[i:])
            if fnmatch.fnmatch(partial_path, pattern):
                return True
                
        return False


def survey_directory(root_path: Path) -> Dict[str, any]:
    """Survey a directory tree, respecting .gitignore files."""
    
    # Initialize gitignore parser
    gitignore = GitignoreParser()
    
    # Document types to track
    doc_extensions = {
        '.md': 'markdown',
        '.txt': 'text',
        '.pdf': 'pdf',
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'header',
        '.hpp': 'header',
        '.rs': 'rust',
        '.go': 'go',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.tex': 'latex',
        '.ipynb': 'jupyter',
        '.sh': 'shell',
        '.bash': 'shell',
        '.R': 'r',
        '.jl': 'julia',
        '.m': 'matlab',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.rb': 'ruby',
        '.php': 'php',
        '.pl': 'perl',
        '.lua': 'lua',
        '.vim': 'vim',
        '.el': 'elisp',
        '.clj': 'clojure',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.fs': 'fsharp',
        '.dart': 'dart',
        '.vue': 'vue',
        '.svelte': 'svelte',
        '.astro': 'astro',
        '.sol': 'solidity',
        '.proto': 'protobuf',
        '.graphql': 'graphql',
        '.gql': 'graphql',
        '.sql': 'sql',
        '.dockerfile': 'docker',
        '.makefile': 'make',
        '.cmake': 'cmake',
        '.gradle': 'gradle',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'config',
        '.conf': 'config',
        '.log': 'log',
        '.csv': 'csv',
        '.tsv': 'tsv',
        '.jsonl': 'jsonlines',
        '.parquet': 'parquet',
        '.avro': 'avro',
        '.proto': 'protobuf',
        '.thrift': 'thrift',
        '.rst': 'restructuredtext',
        '.adoc': 'asciidoc',
        '.org': 'org',
        '.wiki': 'wiki',
        '.mediawiki': 'mediawiki',
        '.pod': 'pod',
        '.man': 'man',
        '.info': 'info',
        '.texi': 'texinfo'
    }
    
    # Statistics
    stats = defaultdict(int)
    stats['total_files'] = 0
    stats['total_dirs'] = 0
    stats['ignored_files'] = 0
    stats['ignored_dirs'] = 0
    stats['gitignore_files'] = 0
    
    # Document counts by type
    doc_counts = defaultdict(int)
    
    # Project detection
    projects = defaultdict(dict)
    
    # Size statistics
    total_size = 0
    doc_sizes = defaultdict(int)
    
    # Special directories
    special_dirs = {
        'HADES': {'docs': 0, 'code': 0, 'theory': 0},
        'code_with_papers': {},
        'other_projects': {}
    }
    
    # Walk the directory tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        current_path = Path(dirpath)
        
        # Check for .gitignore in current directory
        gitignore_path = current_path / '.gitignore'
        if gitignore_path.exists():
            gitignore.load_gitignore(gitignore_path)
            stats['gitignore_files'] += 1
        
        # Filter out ignored directories
        original_dirnames = dirnames[:]
        dirnames[:] = []
        for dirname in original_dirnames:
            dir_path = current_path / dirname
            if gitignore.should_ignore(dir_path):
                stats['ignored_dirs'] += 1
            else:
                dirnames.append(dirname)
                stats['total_dirs'] += 1
        
        # Process files
        for filename in filenames:
            file_path = current_path / filename
            
            # Check if file should be ignored
            if gitignore.should_ignore(file_path):
                stats['ignored_files'] += 1
                continue
                
            stats['total_files'] += 1
            
            # Get file extension
            ext = file_path.suffix.lower()
            if not ext and filename.lower() in ['dockerfile', 'makefile']:
                ext = f'.{filename.lower()}'
            
            # Track document types
            if ext in doc_extensions:
                doc_type = doc_extensions[ext]
                doc_counts[doc_type] += 1
                
                # Get file size
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    doc_sizes[doc_type] += size
                except:
                    pass
                
                # Categorize by project
                rel_path = file_path.relative_to(root_path)
                parts = rel_path.parts
                
                if len(parts) > 0:
                    if parts[0] == 'HADES':
                        if 'theory' in str(file_path):
                            special_dirs['HADES']['theory'] += 1
                        elif 'docs' in parts or 'documentation' in parts:
                            special_dirs['HADES']['docs'] += 1
                        else:
                            special_dirs['HADES']['code'] += 1
                    elif parts[0] == 'code_with_papers' and len(parts) > 1:
                        project_name = parts[1]
                        if project_name not in special_dirs['code_with_papers']:
                            special_dirs['code_with_papers'][project_name] = 0
                        special_dirs['code_with_papers'][project_name] += 1
                    else:
                        project_name = parts[0]
                        if project_name not in special_dirs['other_projects']:
                            special_dirs['other_projects'][project_name] = 0
                        special_dirs['other_projects'][project_name] += 1
    
    # Compile results
    results = {
        'root_path': str(root_path),
        'statistics': dict(stats),
        'document_counts': dict(doc_counts),
        'document_sizes_bytes': dict(doc_sizes),
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'special_directories': special_dirs,
        'gitignore_files_found': gitignore.gitignore_files,
        'total_documents': sum(doc_counts.values()),
        'unique_document_types': len(doc_counts)
    }
    
    return results


def print_summary(results: Dict[str, any]):
    """Print a formatted summary of the survey results."""
    print("\n" + "="*60)
    print("OLYMPUS DIRECTORY SURVEY RESULTS")
    print("="*60)
    
    print(f"\nRoot Path: {results['root_path']}")
    print(f"\nGeneral Statistics:")
    print(f"  Total Files: {results['statistics']['total_files']:,}")
    print(f"  Total Directories: {results['statistics']['total_dirs']:,}")
    print(f"  Ignored Files: {results['statistics']['ignored_files']:,}")
    print(f"  Ignored Directories: {results['statistics']['ignored_dirs']:,}")
    print(f"  .gitignore Files: {results['statistics']['gitignore_files']:,}")
    
    print(f"\nDocument Statistics:")
    print(f"  Total Documents: {results['total_documents']:,}")
    print(f"  Total Size: {results['total_size_mb']:.2f} MB")
    print(f"  Document Types: {results['unique_document_types']}")
    
    print(f"\nTop Document Types:")
    sorted_docs = sorted(results['document_counts'].items(), key=lambda x: x[1], reverse=True)
    for doc_type, count in sorted_docs[:10]:
        size_mb = results['document_sizes_bytes'].get(doc_type, 0) / (1024 * 1024)
        print(f"  {doc_type:20} {count:8,} files  ({size_mb:8.2f} MB)")
    
    print(f"\nHADES Project Documents:")
    hades = results['special_directories']['HADES']
    print(f"  Theory Documents: {hades['theory']:,}")
    print(f"  Documentation: {hades['docs']:,}")
    print(f"  Code Files: {hades['code']:,}")
    print(f"  Total HADES: {sum(hades.values()):,}")
    
    print(f"\nCode with Papers Projects: {len(results['special_directories']['code_with_papers'])}")
    for project, count in sorted(results['special_directories']['code_with_papers'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {project:30} {count:6,} documents")
    
    print(f"\nOther Projects: {len(results['special_directories']['other_projects'])}")
    for project, count in sorted(results['special_directories']['other_projects'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {project:30} {count:6,} documents")


def main():
    """Main function to run the survey."""
    # Get olympus directory path
    olympus_path = Path("/home/todd/olympus")
    
    if not olympus_path.exists():
        print(f"Error: Directory {olympus_path} does not exist!")
        return
    
    print(f"Surveying {olympus_path}...")
    print("This may take a few moments...")
    
    # Run the survey
    results = survey_directory(olympus_path)
    
    # Print summary
    print_summary(results)
    
    # Save detailed results
    output_file = Path("/home/todd/olympus/HADES/data/olympus_survey_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Quick stats for HADES dataset planning
    print("\n" + "="*60)
    print("HADES DATASET PLANNING INSIGHTS")
    print("="*60)
    
    total_docs = results['total_documents']
    hades_docs = sum(results['special_directories']['HADES'].values())
    code_papers_docs = sum(results['special_directories']['code_with_papers'].values())
    
    print(f"\nPotential HADES Internal Bridges: {hades_docs:,} documents")
    print(f"Potential External Bridges (code_with_papers): {code_papers_docs:,} documents")
    print(f"Potential Validation Set: {total_docs - hades_docs - code_papers_docs:,} documents")
    
    print(f"\nRecommended Sampling Strategy:")
    print(f"  - Use all {hades_docs} HADES documents as known internal bridges")
    print(f"  - Sample ~80 from code_with_papers as external bridges")
    print(f"  - Random sample ~900 from remaining {total_docs - hades_docs:,} for validation")


if __name__ == "__main__":
    main()