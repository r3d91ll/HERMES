"""
Markdown file loader for HERMES pipeline.
Handles .md files with structure extraction and metadata parsing.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import re
import markdown
from markdown.extensions.toc import TocExtension
from markdown.extensions.meta import MetaExtension

from hermes.core.base import BaseLoader

logger = logging.getLogger(__name__)


class MarkdownLoader(BaseLoader):
    """Load and process Markdown files with structure preservation."""
    
    def __init__(self):
        """Initialize markdown loader with parser."""
        self.supported_extensions = ['.md', '.markdown', '.mdown']
        self.md = markdown.Markdown(extensions=[
            'meta',
            'toc',
            'tables',
            'fenced_code',
            'codehilite',
            'footnotes'
        ])
        
    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() in self.supported_extensions
        
    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        Load markdown file with structure extraction.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            Document data with content, structure, and metadata
        """
        if not self.can_load(file_path):
            raise ValueError(f"MarkdownLoader cannot load {file_path.suffix} files")
            
        logger.info(f"Loading markdown file: {file_path}")
        
        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse markdown
        html = self.md.convert(content)
        
        # Extract structure and metadata
        structure = self._extract_structure(content)
        metadata = self._extract_metadata(file_path, content, self.md.Meta)
        
        # Extract code blocks separately for conveyance analysis
        code_blocks = self._extract_code_blocks(content)
        
        return {
            'content': content,
            'html': html,
            'structure': structure,
            'code_blocks': code_blocks,
            'metadata': metadata,
            'file_path': str(file_path),
            'loader': 'MarkdownLoader'
        }
    
    def _extract_structure(self, content: str) -> Dict[str, Any]:
        """Extract document structure from markdown."""
        lines = content.splitlines()
        
        structure = {
            'headers': [],
            'sections': [],
            'lists': 0,
            'links': [],
            'images': [],
            'tables': 0,
            'blockquotes': 0
        }
        
        current_section = None
        section_content = []
        
        for i, line in enumerate(lines):
            # Headers
            if match := re.match(r'^(#{1,6})\s+(.+)$', line):
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Save previous section
                if current_section:
                    current_section['content'] = '\n'.join(section_content)
                    structure['sections'].append(current_section)
                
                # Start new section
                current_section = {
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'content': ''
                }
                section_content = []
                
                structure['headers'].append({
                    'level': level,
                    'text': title,
                    'line': i
                })
            
            # Lists
            elif re.match(r'^[\s]*[-*+]\s+', line) or re.match(r'^[\s]*\d+\.\s+', line):
                structure['lists'] += 1
            
            # Links
            for link_match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line):
                structure['links'].append({
                    'text': link_match.group(1),
                    'url': link_match.group(2),
                    'line': i
                })
            
            # Images
            for img_match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', line):
                structure['images'].append({
                    'alt': img_match.group(1),
                    'src': img_match.group(2),
                    'line': i
                })
            
            # Tables
            if '|' in line and i > 0 and '|' in lines[i-1]:
                if re.match(r'^[\s]*\|[\s]*[-:]+[\s]*\|', line):
                    structure['tables'] += 1
            
            # Blockquotes
            if line.strip().startswith('>'):
                structure['blockquotes'] += 1
            
            # Add to current section
            if current_section:
                section_content.append(line)
        
        # Save last section
        if current_section:
            current_section['content'] = '\n'.join(section_content)
            structure['sections'].append(current_section)
        
        return structure
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks for conveyance analysis."""
        code_blocks = []
        
        # Fenced code blocks
        fenced_pattern = r'```(\w*)\n(.*?)\n```'
        for match in re.finditer(fenced_pattern, content, re.DOTALL):
            language = match.group(1) or 'unknown'
            code = match.group(2)
            
            code_blocks.append({
                'language': language,
                'code': code,
                'type': 'fenced',
                'lines': len(code.splitlines()),
                'executable': language in ['python', 'javascript', 'bash', 'sh']
            })
        
        # Indented code blocks
        indented_pattern = r'\n((?:    .*\n)+)'
        for match in re.finditer(indented_pattern, content):
            code = match.group(1)
            # Remove indentation
            code = '\n'.join(line[4:] for line in code.splitlines())
            
            code_blocks.append({
                'language': 'unknown',
                'code': code,
                'type': 'indented',
                'lines': len(code.splitlines()),
                'executable': False
            })
        
        return code_blocks
    
    def _extract_metadata(self, file_path: Path, content: str, 
                         meta_dict: Dict) -> Dict[str, Any]:
        """Extract metadata from markdown file."""
        # Basic text statistics
        lines = content.splitlines()
        words = content.split()
        
        metadata = {
            'line_count': len(lines),
            'word_count': len(words),
            'character_count': len(content),
            'has_frontmatter': bool(meta_dict),
            'frontmatter': dict(meta_dict) if meta_dict else {},
            'has_toc': '[TOC]' in content or '[[TOC]]' in content,
            'reading_time_minutes': len(words) / 200  # Assuming 200 wpm
        }
        
        # Document type indicators
        metadata['document_type'] = self._infer_document_type(content, metadata)
        
        # Add filesystem metadata
        stats = file_path.stat()
        metadata.update({
            'file_size': stats.st_size,
            'created_time': stats.st_ctime,
            'modified_time': stats.st_mtime,
            'permissions': oct(stats.st_mode)[-3:]
        })
        
        return metadata
    
    def _infer_document_type(self, content: str, metadata: Dict) -> str:
        """Infer the type of markdown document."""
        lower_content = content.lower()
        
        # README
        if 'readme' in metadata.get('file_path', '').lower():
            return 'readme'
        
        # Documentation
        if any(word in lower_content[:500] for word in 
               ['documentation', 'api reference', 'user guide', 'tutorial']):
            return 'documentation'
        
        # Blog/Article
        if metadata.get('frontmatter', {}).get('date') or \
           any(word in lower_content[:200] for word in ['author:', 'date:', 'published:']):
            return 'article'
        
        # Academic
        if any(word in lower_content for word in 
               ['abstract', 'references', 'bibliography', 'citation']):
            return 'academic'
        
        # Technical
        if len([b for b in metadata.get('code_blocks', []) if b['executable']]) > 2:
            return 'technical'
        
        return 'general'