"""
JSON file loader for HERMES pipeline.
Handles .json files with schema detection and structured data extraction.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import json
from collections import Counter

from hermes.core.base import BaseLoader

logger = logging.getLogger(__name__)


class JSONLoader(BaseLoader):
    """Load and process JSON files with structure analysis."""
    
    def __init__(self):
        """Initialize JSON loader."""
        self.supported_extensions = ['.json', '.jsonl', '.ndjson']
        
    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() in self.supported_extensions
        
    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        Load JSON file with structure analysis.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Document data with content, structure, and metadata
        """
        if not self.can_load(file_path):
            raise ValueError(f"JSONLoader cannot load {file_path.suffix} files")
            
        logger.info(f"Loading JSON file: {file_path}")
        
        # Handle different JSON formats
        if file_path.suffix.lower() in ['.jsonl', '.ndjson']:
            data = self._load_jsonl(file_path)
            content = json.dumps(data, indent=2)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
        
        # Analyze structure
        structure = self._analyze_structure(data)
        metadata = self._extract_metadata(file_path, data, structure)
        
        # Extract text content for embedding
        text_content = self._extract_text_content(data)
        
        return {
            'content': content,
            'data': data,
            'text_content': text_content,
            'structure': structure,
            'metadata': metadata,
            'file_path': str(file_path),
            'loader': 'JSONLoader'
        }
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load JSON Lines file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _analyze_structure(self, data: Union[Dict, List]) -> Dict[str, Any]:
        """Analyze JSON structure recursively."""
        structure = {
            'type': type(data).__name__,
            'depth': self._calculate_depth(data),
            'keys': [],
            'array_lengths': [],
            'value_types': Counter(),
            'schema': self._infer_schema(data),
            'has_nested_objects': False,
            'has_arrays': False
        }
        
        if isinstance(data, dict):
            structure['keys'] = list(data.keys())
            structure['key_count'] = len(data)
            
            for key, value in data.items():
                structure['value_types'][type(value).__name__] += 1
                
                if isinstance(value, dict):
                    structure['has_nested_objects'] = True
                elif isinstance(value, list):
                    structure['has_arrays'] = True
                    structure['array_lengths'].append(len(value))
                    
        elif isinstance(data, list):
            structure['length'] = len(data)
            structure['has_arrays'] = True
            
            if data:
                # Sample first few items for type analysis
                for item in data[:10]:
                    structure['value_types'][type(item).__name__] += 1
                    
                    if isinstance(item, dict):
                        structure['has_nested_objects'] = True
                    elif isinstance(item, list):
                        structure['array_lengths'].append(len(item))
        
        return structure
    
    def _calculate_depth(self, obj: Union[Dict, List], current_depth: int = 0) -> int:
        """Calculate maximum depth of nested structure."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in obj[:10])
        else:
            return current_depth
    
    def _infer_schema(self, data: Union[Dict, List]) -> Dict[str, Any]:
        """Infer a simple schema from the data."""
        if isinstance(data, dict):
            schema = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    schema[key] = self._infer_schema(value)
                else:
                    schema[key] = type(value).__name__
            return schema
        elif isinstance(data, list) and data:
            # For lists, analyze first item
            return {
                'type': 'array',
                'items': self._infer_schema(data[0]) if data else None,
                'length': len(data)
            }
        else:
            return type(data).__name__
    
    def _extract_text_content(self, data: Union[Dict, List], 
                            max_depth: int = 10) -> str:
        """Extract all text content from JSON for embedding."""
        text_parts = []
        
        def extract_recursive(obj, depth=0):
            if depth > max_depth:
                return
                
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Add key as context
                    text_parts.append(f"{key}:")
                    extract_recursive(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item, depth + 1)
            elif isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, (int, float, bool)):
                text_parts.append(str(obj))
        
        extract_recursive(data)
        return ' '.join(text_parts)
    
    def _extract_metadata(self, file_path: Path, data: Union[Dict, List], 
                         structure: Dict) -> Dict[str, Any]:
        """Extract metadata from JSON file."""
        metadata = {
            'structure_type': structure['type'],
            'depth': structure['depth'],
            'complexity': self._calculate_complexity(structure),
            'data_type': self._infer_data_type(data, structure),
            'has_schema': self._detect_schema_fields(data),
            'size_metrics': {
                'key_count': structure.get('key_count', 0),
                'total_values': sum(structure['value_types'].values()),
                'unique_types': len(structure['value_types'])
            }
        }
        
        # Check for common patterns
        metadata['patterns'] = self._detect_patterns(data)
        
        # Add filesystem metadata
        stats = file_path.stat()
        metadata.update({
            'file_size': stats.st_size,
            'created_time': stats.st_ctime,
            'modified_time': stats.st_mtime,
            'permissions': oct(stats.st_mode)[-3:]
        })
        
        return metadata
    
    def _calculate_complexity(self, structure: Dict) -> float:
        """Calculate a complexity score for the JSON structure."""
        complexity = 0.0
        
        # Depth adds complexity
        complexity += structure['depth'] * 0.2
        
        # Mixed types add complexity
        complexity += len(structure['value_types']) * 0.1
        
        # Nested structures add complexity
        if structure['has_nested_objects']:
            complexity += 0.3
        if structure['has_arrays']:
            complexity += 0.2
        
        # Large structures are complex
        if structure.get('key_count', 0) > 50:
            complexity += 0.3
        elif structure.get('key_count', 0) > 20:
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _infer_data_type(self, data: Union[Dict, List], 
                        structure: Dict) -> str:
        """Infer the type of data in the JSON file."""
        # API Response
        if isinstance(data, dict):
            common_api_keys = {'status', 'data', 'error', 'message', 'results', 'response'}
            if len(common_api_keys & set(structure['keys'])) >= 2:
                return 'api_response'
        
        # Configuration
        if isinstance(data, dict):
            if all(not isinstance(v, (dict, list)) or 
                  (isinstance(v, dict) and self._calculate_depth(v) <= 2) 
                  for v in data.values()):
                return 'configuration'
        
        # Dataset
        if isinstance(data, list) and len(data) > 10:
            if all(isinstance(item, dict) for item in data[:10]):
                return 'dataset'
        
        # Schema/Model
        if isinstance(data, dict):
            schema_keys = {'properties', 'type', 'required', 'definitions', '$schema'}
            if len(schema_keys & set(structure['keys'])) >= 2:
                return 'schema'
        
        # Nested data
        if structure['depth'] > 3:
            return 'nested_data'
        
        return 'general'
    
    def _detect_schema_fields(self, data: Union[Dict, List]) -> bool:
        """Detect if JSON contains schema-like fields."""
        if isinstance(data, dict):
            schema_indicators = {
                '$schema', 'type', 'properties', 'required',
                'definitions', 'allOf', 'anyOf', 'oneOf'
            }
            return bool(schema_indicators & set(data.keys()))
        return False
    
    def _detect_patterns(self, data: Union[Dict, List]) -> List[str]:
        """Detect common patterns in JSON data."""
        patterns = []
        
        if isinstance(data, dict):
            # Timestamp patterns
            for key in data.keys():
                if any(ts in key.lower() for ts in ['time', 'date', 'created', 'updated']):
                    patterns.append('timestamps')
                    break
            
            # ID patterns
            if any(id_key in data.keys() for id_key in ['id', '_id', 'uuid', 'guid']):
                patterns.append('identifiers')
            
            # Metadata patterns
            if any(meta in data.keys() for meta in ['meta', 'metadata', '_meta']):
                patterns.append('metadata')
                
        elif isinstance(data, list) and data:
            # Check if it's a homogeneous list
            if all(isinstance(item, dict) for item in data[:10]):
                first_keys = set(data[0].keys()) if isinstance(data[0], dict) else set()
                if all(isinstance(item, dict) and set(item.keys()) == first_keys 
                      for item in data[:10]):
                    patterns.append('homogeneous_list')
        
        return patterns