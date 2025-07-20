#!/usr/bin/env python3
"""
Create a dimensional graph from a directory structure.
This demonstrates the multi-dimensional information theory in practice.
"""

import os
import hashlib
import json
from pathlib import Path
from arango import ArangoClient
from dotenv import load_dotenv
from datetime import datetime
import stat

# Load environment variables
load_dotenv()

class DimensionalGraphBuilder:
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
        self.edges_who = self.db.collection('edges_who')
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
        skip_patterns = ['.git', '__pycache__', '.pyc', '.env', 'node_modules']
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
            'owner': self._get_owner(stats),
            'permissions': oct(stats.st_mode)[-3:],
            'created_at': datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'metadata': {
                'is_directory': True,
                'child_count': len(list(dir_path.iterdir())) if dir_path.exists() else 0
            }
        }
        
        try:
            result = self.nodes.insert(node_data)
            self.created_nodes[str(dir_path)] = result['_id']
            return result['_id']
        except Exception as e:
            if 'unique constraint violated' in str(e):
                # Node already exists
                existing = self.nodes.get(node_key)
                self.created_nodes[str(dir_path)] = existing['_id']
                return existing['_id']
            raise
    
    def _create_file_node(self, file_path):
        """Create a node for a file."""
        node_key = self._path_to_key(file_path)
        
        # Get file stats
        stats = file_path.stat()
        
        # Read file content for simple text files
        content_sample = ""
        if file_path.suffix in ['.py', '.txt', '.md', '.json', '.yml', '.yaml']:
            try:
                content_sample = file_path.read_text()[:500]  # First 500 chars
            except:
                content_sample = ""
        
        node_data = {
            '_key': node_key,
            'type': 'file',
            'path': str(file_path),
            'name': file_path.name,
            'extension': file_path.suffix,
            'size': stats.st_size,
            'owner': self._get_owner(stats),
            'permissions': oct(stats.st_mode)[-3:],
            'created_at': datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'content_hash': self._hash_file(file_path),
            'metadata': {
                'is_file': True,
                'content_sample': content_sample,
                'lines_of_code': self._count_lines(file_path) if file_path.suffix == '.py' else 0
            }
        }
        
        try:
            result = self.nodes.insert(node_data)
            self.created_nodes[str(file_path)] = result['_id']
            return result['_id']
        except Exception as e:
            if 'unique constraint violated' in str(e):
                # Node already exists
                existing = self.nodes.get(node_key)
                self.created_nodes[str(file_path)] = existing['_id']
                return existing['_id']
            raise
    
    def _path_to_key(self, path):
        """Convert path to a valid ArangoDB key."""
        # Create a hash of the path for a valid key
        return hashlib.md5(str(path).encode()).hexdigest()
    
    def _get_owner(self, stats):
        """Get owner name from stats."""
        try:
            import pwd
            return pwd.getpwuid(stats.st_uid).pw_name
        except:
            return str(stats.st_uid)
    
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
                
                # Calculate spatial weight based on directory depth
                # Direct parent-child = 1.0, decreases with distance
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
                        'depth_difference': 1
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
                                'both_files': path.is_file() and sibling.is_file()
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
        
        # For this demo, we'll create simple content-based edges
        # In production, this would use embeddings and similarity calculations
        
        # Group files by extension (simple semantic grouping)
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
                    # Same file type = moderate semantic similarity
                    weight = 0.6
                    
                    # Python files might import each other
                    if ext == '.py':
                        content1 = Path(path1).read_text() if Path(path1).exists() else ""
                        content2 = Path(path2).read_text() if Path(path2).exists() else ""
                        
                        # Check for imports (simple heuristic)
                        name1 = Path(path1).stem
                        name2 = Path(path2).stem
                        
                        if f"import {name2}" in content1 or f"from {name2}" in content1:
                            weight = 0.9  # Strong semantic connection
                        elif f"import {name1}" in content2 or f"from {name1}" in content2:
                            weight = 0.9
                    
                    edge_data = {
                        '_from': id1,
                        '_to': id2,
                        'weight': weight,
                        'relationship': 'similar_type',
                        'dimension': 'WHAT',
                        'metadata': {
                            'file_type': ext,
                            'similarity_method': 'file_extension'
                        }
                    }
                    
                    try:
                        self.edges_what.insert(edge_data)
                        edges_created += 1
                    except:
                        pass
        
        print(f"✓ Created {edges_created} WHAT edges")
        return edges_created
    
    def create_who_edges(self):
        """Create WHO edges based on permissions and ownership."""
        print("\nCreating WHO edges (permission relationships)...")
        edges_created = 0
        
        # Group nodes by owner
        nodes_by_owner = {}
        for path_str, node_id in self.created_nodes.items():
            node = self.nodes.get(node_id.split('/')[-1])
            owner = node.get('owner', 'unknown')
            
            if owner not in nodes_by_owner:
                nodes_by_owner[owner] = []
            nodes_by_owner[owner].append((node_id, node))
        
        # Create edges between nodes with same owner
        for owner, nodes in nodes_by_owner.items():
            for i, (id1, node1) in enumerate(nodes):
                for id2, node2 in nodes[i+1:]:
                    # Same owner = full permission alignment
                    weight = 1.0
                    
                    # Check if permissions are also the same
                    if node1.get('permissions') == node2.get('permissions'):
                        relationship = 'same_owner_same_perms'
                    else:
                        relationship = 'same_owner_diff_perms'
                        weight = 0.8  # Slightly lower if permissions differ
                    
                    edge_data = {
                        '_from': id1,
                        '_to': id2,
                        'weight': weight,
                        'relationship': relationship,
                        'dimension': 'WHO',
                        'metadata': {
                            'owner': owner,
                            'perms1': node1.get('permissions'),
                            'perms2': node2.get('permissions')
                        }
                    }
                    
                    try:
                        self.edges_who.insert(edge_data)
                        edges_created += 1
                    except:
                        pass
        
        print(f"✓ Created {edges_created} WHO edges")
        return edges_created
    
    def calculate_composite_edges(self):
        """Calculate composite edges from dimensional strengths."""
        print("\nCalculating composite edges (WHERE × WHAT × WHO)...")
        edges_created = 0
        
        # Query to find all node pairs that have edges in multiple dimensions
        query = """
        // Find all unique node pairs
        LET node_pairs = (
            FOR edge IN UNION(
                (FOR e IN edges_where RETURN {from: e._from, to: e._to}),
                (FOR e IN edges_what RETURN {from: e._from, to: e._to}),
                (FOR e IN edges_who RETURN {from: e._from, to: e._to})
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
            
            LET who_weight = FIRST(
                FOR e IN edges_who 
                FILTER e._from == pair.from AND e._to == pair.to 
                RETURN e.weight
            ) OR 0
            
            LET composite_weight = where_weight * what_weight * who_weight
            LET dimensions_active = (where_weight > 0 ? 1 : 0) + 
                                  (what_weight > 0 ? 1 : 0) + 
                                  (who_weight > 0 ? 1 : 0)
            
            FILTER composite_weight > 0  // Only create edge if composite > 0
            
            RETURN {
                from: pair.from,
                to: pair.to,
                where_weight: where_weight,
                what_weight: what_weight,
                who_weight: who_weight,
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
                'who_weight': result['who_weight'],
                'dimensions_active': result['dimensions_active'],
                'dimension': 'COMPOSITE',
                'metadata': {
                    'calculation': 'WHERE × WHAT × WHO',
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
                   edge.who_weight == 0
            RETURN {
                from: edge._from,
                to: edge._to,
                where: edge.where_weight,
                what: edge.what_weight,
                who: edge.who_weight,
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
                for v in violations[:5]:  # Show first 5
                    print(f"  {v}")
            else:
                print("✓ Hypothesis confirmed: All edges with zero dimension have zero composite weight")
        else:
            print("✓ No edges found with zero dimensions (all dimensions active)")
        
        # Also show some examples of multi-dimensional connections
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
                who: edge.who_weight,
                composite: edge.total_weight
            }
        """
        
        print("\nStrongest multi-dimensional connections:")
        cursor = self.db.aql.execute(strong_query)
        for edge in cursor:
            print(f"  {edge['from']} → {edge['to']}")
            print(f"    WHERE: {edge['where']:.2f}, WHAT: {edge['what']:.2f}, WHO: {edge['who']:.2f}")
            print(f"    Composite: {edge['composite']:.3f}")


def main():
    # Choose a test directory
    test_dir = input("Enter directory path to scan (default: current directory): ").strip()
    if not test_dir:
        test_dir = "."
    
    test_dir = Path(test_dir).resolve()
    
    if not test_dir.exists():
        print(f"Error: Directory {test_dir} does not exist!")
        return
    
    print(f"\nCreating dimensional graph from: {test_dir}")
    print("="*60)
    
    builder = DimensionalGraphBuilder()
    
    # Step 1: Create nodes
    builder.scan_directory(test_dir)
    
    # Step 2: Create dimensional edges
    builder.create_where_edges()
    builder.create_what_edges()
    builder.create_who_edges()
    
    # Step 3: Calculate composite edges
    builder.calculate_composite_edges()
    
    # Step 4: Test hypothesis
    builder.test_zero_dimension_hypothesis()
    
    print("\n" + "="*60)
    print("✅ Dimensional graph creation complete!")
    print("\nYou can now query the graph to explore multi-dimensional relationships.")

if __name__ == "__main__":
    main()