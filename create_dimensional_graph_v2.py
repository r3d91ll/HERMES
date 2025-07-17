#!/usr/bin/env python3
"""
Create a dimensional graph from a directory structure using WHERE/WHAT/CONVEYANCE model.
This properly implements the multi-dimensional information theory.
"""

import os
import hashlib
import json
import mimetypes
import subprocess
from pathlib import Path
from arango import ArangoClient
from dotenv import load_dotenv
from datetime import datetime
import stat

# Load environment variables
load_dotenv()

class DimensionalGraphBuilderV2:
    def __init__(self):
        # Connect to ArangoDB
        host = os.getenv('ARANGO_HOST', 'localhost')
        port = int(os.getenv('ARANGO_PORT', 8529))
        username = os.getenv('ARANGO_USERNAME', 'root')
        password = os.getenv('ARANGO_PASSWORD', '')
        db_name = os.getenv('ARANGO_DATABASE', 'hermes')
        
        client = ArangoClient(hosts=f"http://{host}:{port}")
        self.db = client.db(db_name, username=username, password=password)
        
        # Get collections
        self.nodes = self.db.collection('nodes')
        self.edges_where = self.db.collection('edges_where')
        self.edges_what = self.db.collection('edges_what')
        self.edges_conveyance = self.db.collection('edges_conveyance')  # Renamed from edges_who
        self.edges_composite = self.db.collection('edges_composite')
        
        # Track created nodes
        self.created_nodes = {}
        
    def scan_directory(self, root_path):
        """Scan directory and create nodes for files and directories."""
        root_path = Path(root_path).resolve()
        print(f"\nScanning directory: {root_path}")
        
        nodes_created = 0
        
        # Create root directory node
        root_node_id = self._create_directory_node(root_path)
        nodes_created += 1
        
        # Walk through directory
        for dirpath, dirnames, filenames in os.walk(root_path):
            current_dir = Path(dirpath)
            
            # Create nodes for subdirectories
            for dirname in dirnames:
                dir_path = current_dir / dirname
                if not self._should_skip(dir_path):
                    self._create_directory_node(dir_path)
                    nodes_created += 1
            
            # Create nodes for files
            for filename in filenames:
                file_path = current_dir / filename
                if not self._should_skip(file_path):
                    self._create_file_node(file_path)
                    nodes_created += 1
        
        print(f"✓ Created {nodes_created} nodes")
        return nodes_created
    
    def _should_skip(self, path):
        """Check if path should be skipped."""
        skip_patterns = ['.git', '__pycache__', '.pyc', '.env', 'node_modules', '.DS_Store']
        return any(pattern in str(path) for pattern in skip_patterns)
    
    def _create_directory_node(self, dir_path):
        """Create a node for a directory."""
        node_key = self._path_to_key(dir_path)
        
        # Get directory stats
        stats = dir_path.stat()
        
        node_data = {
            '_key': node_key,
            'type': 'directory',
            'path': str(dir_path),
            'name': dir_path.name,
            'size': 0,  # Directories don't have size
            'created_at': datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'metadata': {
                'is_directory': True,
                'child_count': len(list(dir_path.iterdir())) if dir_path.exists() else 0,
                'permissions': oct(stats.st_mode)[-3:],
                'owner_uid': stats.st_uid,
                'group_gid': stats.st_gid
            }
        }
        
        try:
            result = self.nodes.insert(node_data)
            self.created_nodes[str(dir_path)] = result['_id']
            return result['_id']
        except Exception as e:
            if 'unique constraint violated' in str(e):
                existing = self.nodes.get(node_key)
                self.created_nodes[str(dir_path)] = existing['_id']
                return existing['_id']
            raise
    
    def _create_file_node(self, file_path):
        """Create a node for a file with conveyance metadata."""
        node_key = self._path_to_key(file_path)
        
        # Get file stats
        stats = file_path.stat()
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Check if file is executable
        is_executable = os.access(file_path, os.X_OK)
        
        # Check if file is readable
        is_readable = os.access(file_path, os.R_OK)
        
        # Check if file is writable
        is_writable = os.access(file_path, os.W_OK)
        
        # Read content sample for text files
        content_sample = ""
        encoding = None
        if file_path.suffix in ['.py', '.txt', '.md', '.json', '.yml', '.yaml', '.sh', '.js']:
            try:
                content_bytes = file_path.read_bytes()[:1000]  # First 1KB
                # Try to detect encoding
                try:
                    content_sample = content_bytes.decode('utf-8')
                    encoding = 'utf-8'
                except:
                    try:
                        content_sample = content_bytes.decode('latin-1')
                        encoding = 'latin-1'
                    except:
                        content_sample = ""
                        encoding = 'binary'
            except:
                content_sample = ""
        
        # Check for special file types that affect conveyance
        is_compressed = file_path.suffix in ['.gz', '.zip', '.tar', '.bz2', '.7z']
        is_encrypted = file_path.suffix in ['.gpg', '.enc', '.aes']
        requires_special_tool = file_path.suffix in ['.psd', '.ai', '.sketch', '.fig']
        
        node_data = {
            '_key': node_key,
            'type': 'file',
            'path': str(file_path),
            'name': file_path.name,
            'extension': file_path.suffix,
            'size': stats.st_size,
            'created_at': datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'content_hash': self._hash_file(file_path),
            'mime_type': mime_type or 'unknown',
            'metadata': {
                'is_file': True,
                'content_sample': content_sample[:500],  # Limit to 500 chars
                'encoding': encoding,
                'lines_of_code': self._count_lines(file_path) if file_path.suffix == '.py' else 0,
                # Conveyance-related metadata
                'permissions': oct(stats.st_mode)[-3:],
                'owner_uid': stats.st_uid,
                'group_gid': stats.st_gid,
                'is_readable': is_readable,
                'is_writable': is_writable,
                'is_executable': is_executable,
                'is_compressed': is_compressed,
                'is_encrypted': is_encrypted,
                'requires_special_tool': requires_special_tool
            }
        }
        
        try:
            result = self.nodes.insert(node_data)
            self.created_nodes[str(file_path)] = result['_id']
            return result['_id']
        except Exception as e:
            if 'unique constraint violated' in str(e):
                existing = self.nodes.get(node_key)
                self.created_nodes[str(file_path)] = existing['_id']
                return existing['_id']
            raise
    
    def _path_to_key(self, path):
        """Convert path to a valid ArangoDB key."""
        return hashlib.md5(str(path).encode()).hexdigest()
    
    def _hash_file(self, file_path):
        """Calculate file content hash."""
        try:
            return hashlib.md5(file_path.read_bytes()).hexdigest()
        except:
            return "unreadable"
    
    def _count_lines(self, file_path):
        """Count lines in a file."""
        try:
            return len(file_path.read_text().splitlines())
        except:
            return 0
    
    def create_where_edges(self):
        """Create WHERE edges based on directory structure."""
        print("\nCreating WHERE edges (spatial relationships)...")
        edges_created = 0
        
        # Create parent-child edges
        for path_str, node_id in self.created_nodes.items():
            path = Path(path_str)
            parent = path.parent
            
            if str(parent) in self.created_nodes:
                parent_id = self.created_nodes[str(parent)]
                
                # Direct parent-child = strongest spatial connection
                weight = 1.0
                
                edge_data = {
                    '_from': parent_id,
                    '_to': node_id,
                    'weight': weight,
                    'relationship': 'contains',
                    'dimension': 'WHERE',
                    'metadata': {
                        'parent_path': str(parent),
                        'child_path': str(path),
                        'depth_difference': 1,
                        'path_distance': 1
                    }
                }
                
                try:
                    self.edges_where.insert(edge_data)
                    edges_created += 1
                except Exception as e:
                    if 'unique constraint violated' not in str(e):
                        print(f"Error creating WHERE edge: {e}")
        
        # Create sibling edges (same directory)
        for path_str, node_id in self.created_nodes.items():
            path = Path(path_str)
            parent = path.parent
            
            # Find siblings
            if parent.exists():
                for sibling in parent.iterdir():
                    sibling_str = str(sibling)
                    if sibling_str != path_str and sibling_str in self.created_nodes:
                        sibling_id = self.created_nodes[sibling_str]
                        
                        # Siblings have strong spatial relationship
                        edge_data = {
                            '_from': node_id,
                            '_to': sibling_id,
                            'weight': 0.8,
                            'relationship': 'sibling',
                            'dimension': 'WHERE',
                            'metadata': {
                                'parent_directory': str(parent),
                                'both_files': path.is_file() and sibling.is_file(),
                                'path_distance': 0  # Same directory
                            }
                        }
                        
                        try:
                            self.edges_where.insert(edge_data)
                            edges_created += 1
                        except:
                            pass  # Ignore duplicates
        
        print(f"✓ Created {edges_created} WHERE edges")
        return edges_created
    
    def create_what_edges(self):
        """Create WHAT edges based on content similarity."""
        print("\nCreating WHAT edges (semantic relationships)...")
        edges_created = 0
        
        # Group files by extension for basic semantic grouping
        files_by_type = {}
        for path_str, node_id in self.created_nodes.items():
            path = Path(path_str)
            if path.is_file():
                ext = path.suffix
                if ext not in files_by_type:
                    files_by_type[ext] = []
                files_by_type[ext].append((path_str, node_id))
        
        # Create edges between files of same type
        for ext, files in files_by_type.items():
            for i, (path1, id1) in enumerate(files):
                for path2, id2 in files[i+1:]:
                    # Base semantic similarity for same file type
                    weight = 0.5
                    
                    # Python files might import each other
                    if ext == '.py':
                        try:
                            content1 = Path(path1).read_text()
                            content2 = Path(path2).read_text()
                            
                            name1 = Path(path1).stem
                            name2 = Path(path2).stem
                            
                            # Check for imports (increases semantic connection)
                            if f"import {name2}" in content1 or f"from {name2}" in content1:
                                weight = 0.9
                            elif f"import {name1}" in content2 or f"from {name1}" in content2:
                                weight = 0.9
                            elif name1 in content2 or name2 in content1:
                                weight = 0.7  # References each other
                        except:
                            pass
                    
                    # Configuration files of same type are semantically related
                    elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini']:
                        weight = 0.7
                    
                    # Documentation files
                    elif ext in ['.md', '.rst', '.txt']:
                        weight = 0.6
                    
                    edge_data = {
                        '_from': id1,
                        '_to': id2,
                        'weight': weight,
                        'relationship': 'similar_content',
                        'dimension': 'WHAT',
                        'metadata': {
                            'file_type': ext,
                            'similarity_method': 'file_type_and_imports',
                            'both_executable': ext in ['.py', '.sh', '.js']
                        }
                    }
                    
                    try:
                        self.edges_what.insert(edge_data)
                        edges_created += 1
                    except:
                        pass
        
        print(f"✓ Created {edges_created} WHAT edges")
        return edges_created
    
    def create_conveyance_edges(self):
        """Create CONVEYANCE edges based on ability to access/use information."""
        print("\nCreating CONVEYANCE edges (transmission capabilities)...")
        edges_created = 0
        
        # Get all file nodes
        file_nodes = []
        for path_str, node_id in self.created_nodes.items():
            if Path(path_str).is_file():
                node = self.nodes.get(node_id.split('/')[-1])
                file_nodes.append((node_id, node, path_str))
        
        # Create edges based on conveyance characteristics
        for i, (id1, node1, path1) in enumerate(file_nodes):
            for id2, node2, path2 in file_nodes[i+1:]:
                weight = 0.0
                relationship = ""
                metadata = {}
                
                # Same permissions = easy to convey between
                perms1 = node1['metadata'].get('permissions', '000')
                perms2 = node2['metadata'].get('permissions', '000')
                
                if perms1 == perms2:
                    weight = 0.9
                    relationship = "same_permissions"
                    metadata['permissions'] = perms1
                
                # Both readable = can be conveyed
                if node1['metadata'].get('is_readable') and node2['metadata'].get('is_readable'):
                    if weight == 0:
                        weight = 0.8
                        relationship = "both_readable"
                
                # Check for conveyance barriers
                barriers = []
                
                # Encryption is a major barrier
                if node1['metadata'].get('is_encrypted') or node2['metadata'].get('is_encrypted'):
                    weight *= 0.1  # Drastically reduce conveyance
                    barriers.append('encryption')
                
                # Compression is a minor barrier
                if node1['metadata'].get('is_compressed') or node2['metadata'].get('is_compressed'):
                    weight *= 0.8  # Slightly reduce conveyance
                    barriers.append('compression')
                
                # Special tools required
                if node1['metadata'].get('requires_special_tool') or node2['metadata'].get('requires_special_tool'):
                    weight *= 0.5  # Moderate reduction
                    barriers.append('special_tools')
                
                # Large files are harder to convey
                size1 = node1.get('size', 0)
                size2 = node2.get('size', 0)
                avg_size = (size1 + size2) / 2
                
                if avg_size > 100_000_000:  # 100MB
                    weight *= 0.7
                    barriers.append('large_size')
                elif avg_size > 1_000_000_000:  # 1GB
                    weight *= 0.3
                    barriers.append('very_large_size')
                
                # Same owner = better conveyance
                if node1['metadata'].get('owner_uid') == node2['metadata'].get('owner_uid'):
                    weight *= 1.1  # Slight boost
                    if weight > 1.0:
                        weight = 1.0
                
                # Only create edge if there's meaningful conveyance relationship
                if weight > 0.1:
                    edge_data = {
                        '_from': id1,
                        '_to': id2,
                        'weight': weight,
                        'relationship': relationship,
                        'dimension': 'CONVEYANCE',
                        'metadata': {
                            'barriers': barriers,
                            'avg_size_bytes': avg_size,
                            'both_executable': node1['metadata'].get('is_executable') and node2['metadata'].get('is_executable'),
                            'same_encoding': node1['metadata'].get('encoding') == node2['metadata'].get('encoding')
                        }
                    }
                    
                    try:
                        self.edges_conveyance.insert(edge_data)
                        edges_created += 1
                    except:
                        pass
        
        print(f"✓ Created {edges_created} CONVEYANCE edges")
        return edges_created
    
    def calculate_composite_edges(self):
        """Calculate composite edges from dimensional strengths."""
        print("\nCalculating composite edges (WHERE × WHAT × CONVEYANCE)...")
        edges_created = 0
        
        # Query to find all node pairs that have edges in multiple dimensions
        query = """
        // Find all unique node pairs
        LET node_pairs = (
            FOR edge IN UNION(
                (FOR e IN edges_where RETURN {from: e._from, to: e._to}),
                (FOR e IN edges_what RETURN {from: e._from, to: e._to}),
                (FOR e IN edges_conveyance RETURN {from: e._from, to: e._to})
            )
            RETURN DISTINCT {from: edge.from, to: edge.to}
        )
        
        // For each pair, get weights from all dimensions
        FOR pair IN node_pairs
            LET where_weight = FIRST(
                FOR e IN edges_where 
                FILTER e._from == pair.from AND e._to == pair.to 
                RETURN e.weight
            ) OR 0
            
            LET what_weight = FIRST(
                FOR e IN edges_what 
                FILTER e._from == pair.from AND e._to == pair.to 
                RETURN e.weight
            ) OR 0
            
            LET conveyance_weight = FIRST(
                FOR e IN edges_conveyance 
                FILTER e._from == pair.from AND e._to == pair.to 
                RETURN e.weight
            ) OR 0
            
            LET composite_weight = where_weight * what_weight * conveyance_weight
            LET dimensions_active = (where_weight > 0 ? 1 : 0) + 
                                  (what_weight > 0 ? 1 : 0) + 
                                  (conveyance_weight > 0 ? 1 : 0)
            
            FILTER composite_weight > 0  // Only create edge if composite > 0
            
            RETURN {
                from: pair.from,
                to: pair.to,
                where_weight: where_weight,
                what_weight: what_weight,
                conveyance_weight: conveyance_weight,
                composite_weight: composite_weight,
                dimensions_active: dimensions_active
            }
        """
        
        cursor = self.db.aql.execute(query)
        
        for result in cursor:
            edge_data = {
                '_from': result['from'],
                '_to': result['to'],
                'total_weight': result['composite_weight'],
                'where_weight': result['where_weight'],
                'what_weight': result['what_weight'],
                'conveyance_weight': result['conveyance_weight'],
                'dimensions_active': result['dimensions_active'],
                'dimension': 'COMPOSITE',
                'metadata': {
                    'calculation': 'WHERE × WHAT × CONVEYANCE',
                    'time_constant': 1.0,  # TIME held constant
                    'zero_dimension_test': result['composite_weight'] == 0
                }
            }
            
            try:
                self.edges_composite.insert(edge_data)
                edges_created += 1
            except:
                pass
        
        print(f"✓ Created {edges_created} composite edges")
        return edges_created
    
    def test_zero_dimension_hypothesis(self):
        """Test that when any dimension is zero, composite is zero."""
        print("\nTesting zero-dimension hypothesis...")
        
        # Query for cases where at least one dimension is zero
        query = """
        FOR edge IN edges_composite
            FILTER edge.where_weight == 0 OR 
                   edge.what_weight == 0 OR 
                   edge.conveyance_weight == 0
            RETURN {
                from: edge._from,
                to: edge._to,
                where: edge.where_weight,
                what: edge.what_weight,
                conveyance: edge.conveyance_weight,
                composite: edge.total_weight
            }
        """
        
        cursor = self.db.aql.execute(query)
        results = list(cursor)
        
        if results:
            print(f"Found {len(results)} edges with at least one zero dimension")
            print("All should have composite weight = 0")
            
            violations = [r for r in results if r['composite'] != 0]
            if violations:
                print(f"❌ HYPOTHESIS VIOLATION: Found {len(violations)} edges with zero dimension but non-zero composite!")
                for v in violations[:5]:
                    print(f"  {v}")
            else:
                print("✓ Hypothesis confirmed: All edges with zero dimension have zero composite weight")
        else:
            print("✓ No edges found with zero dimensions (all dimensions active)")
        
        # Show strongest multi-dimensional connections
        strong_query = """
        FOR edge IN edges_composite
            FILTER edge.total_weight > 0.5
            SORT edge.total_weight DESC
            LIMIT 5
            RETURN {
                from: DOCUMENT(edge._from).name,
                to: DOCUMENT(edge._to).name,
                where: edge.where_weight,
                what: edge.what_weight,
                conveyance: edge.conveyance_weight,
                composite: edge.total_weight
            }
        """
        
        print("\nStrongest multi-dimensional connections:")
        cursor = self.db.aql.execute(strong_query)
        for edge in cursor:
            print(f"  {edge['from']} → {edge['to']}")
            print(f"    WHERE: {edge['where']:.2f}, WHAT: {edge['what']:.2f}, CONVEYANCE: {edge['conveyance']:.2f}")
            print(f"    Composite: {edge['composite']:.3f}")
        
        # Show examples of conveyance failures
        barrier_query = """
        FOR node IN nodes
            FILTER node.metadata.is_encrypted == true OR 
                   node.metadata.requires_special_tool == true OR
                   node.metadata.is_readable == false
            LIMIT 3
            RETURN {
                name: node.name,
                path: node.path,
                encrypted: node.metadata.is_encrypted,
                special_tool: node.metadata.requires_special_tool,
                readable: node.metadata.is_readable
            }
        """
        
        print("\nExamples of conveyance barriers:")
        cursor = self.db.aql.execute(barrier_query)
        barriers_found = False
        for node in cursor:
            barriers_found = True
            print(f"  {node['name']}")
            if node.get('encrypted'):
                print("    - Encrypted (requires decryption key)")
            if node.get('special_tool'):
                print("    - Requires special tool to access")
            if not node.get('readable', True):
                print("    - Not readable (permission denied)")
        
        if not barriers_found:
            print("  No files with conveyance barriers found in this directory")


def main():
    # First, we need to create the conveyance collection if it doesn't exist
    print("Setting up conveyance edge collection...")
    
    # Connect to ArangoDB
    load_dotenv()
    host = os.getenv('ARANGO_HOST', 'localhost')
    port = int(os.getenv('ARANGO_PORT', 8529))
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD', '')
    db_name = os.getenv('ARANGO_DATABASE', 'hermes')
    
    client = ArangoClient(hosts=f"http://{host}:{port}")
    db = client.db(db_name, username=username, password=password)
    
    # Create edges_conveyance collection if needed
    if not db.has_collection('edges_conveyance'):
        print("Creating edges_conveyance collection...")
        db.create_collection('edges_conveyance', edge=True)
        print("✓ Created edges_conveyance collection")
    
    # Clear old data
    print("\nClearing old graph data...")
    for collection in ['nodes', 'edges_where', 'edges_what', 'edges_who', 'edges_conveyance', 'edges_composite']:
        if db.has_collection(collection):
            db.collection(collection).truncate()
    print("✓ Cleared old data")
    
    # Choose a test directory
    test_dir = input("\nEnter directory path to scan (press Enter for ISNE directory): ").strip()
    if not test_dir:
        test_dir = "/home/todd/olympus/code_with_papers/inductive-shallow-node-embedding"
    
    test_dir = Path(test_dir).resolve()
    
    if not test_dir.exists():
        print(f"Error: Directory {test_dir} does not exist!")
        return
    
    print(f"\nCreating dimensional graph from: {test_dir}")
    print("="*60)
    
    builder = DimensionalGraphBuilderV2()
    
    # Step 1: Create nodes
    builder.scan_directory(test_dir)
    
    # Step 2: Create dimensional edges
    builder.create_where_edges()
    builder.create_what_edges()
    builder.create_conveyance_edges()
    
    # Step 3: Calculate composite edges
    builder.calculate_composite_edges()
    
    # Step 4: Test hypothesis
    builder.test_zero_dimension_hypothesis()
    
    print("\n" + "="*60)
    print("✅ Dimensional graph creation complete!")
    print("\nGraph now properly models WHERE × WHAT × CONVEYANCE")
    print("You can explore it in ArangoDB Web UI")

if __name__ == "__main__":
    main()