#!/usr/bin/env python3
"""
Minimal test for graph creation with edge values.
This demonstrates the core functionality without the full pipeline.
"""

from arango import ArangoClient
import json

def test_graph_with_edge_values():
    """Create a minimal graph with custom edge values."""
    
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('hermes', username='root', password='')
    
    # Ensure collections exist
    if not db.has_collection('test_nodes'):
        nodes = db.create_collection('test_nodes')
    else:
        nodes = db.collection('test_nodes')
        
    if not db.has_collection('test_edges'):
        edges = db.create_collection('test_edges', edge=True)
    else:
        edges = db.collection('test_edges')
    
    print("Creating test graph...")
    
    # Create nodes representing files in a directory
    node1 = nodes.insert({
        '_key': 'file1',
        'path': '/test/file1.py',
        'type': 'file',
        'size': 1024,
        'content_hash': 'abc123'
    })
    
    node2 = nodes.insert({
        '_key': 'file2',
        'path': '/test/file2.py', 
        'type': 'file',
        'size': 2048,
        'content_hash': 'def456'
    })
    
    node3 = nodes.insert({
        '_key': 'directory',
        'path': '/test',
        'type': 'directory',
        'file_count': 2
    })
    
    print(f"Created nodes: {node1['_key']}, {node2['_key']}, {node3['_key']}")
    
    # Create edges with different weights
    
    # Structural edge: directory contains file (weight = 1.0, certain)
    edge1 = edges.insert({
        '_from': 'test_nodes/directory',
        '_to': 'test_nodes/file1',
        'relationship': 'contains',
        'weight': 1.0,
        'edge_type': 'structural'
    })
    
    edge2 = edges.insert({
        '_from': 'test_nodes/directory',
        '_to': 'test_nodes/file2',
        'relationship': 'contains',
        'weight': 1.0,
        'edge_type': 'structural'
    })
    
    # Semantic edge: files are similar (weight = 0.85, high similarity)
    edge3 = edges.insert({
        '_from': 'test_nodes/file1',
        '_to': 'test_nodes/file2',
        'relationship': 'similar_to',
        'weight': 0.85,
        'edge_type': 'semantic',
        'similarity_metric': 'cosine'
    })
    
    # Import edge: file1 imports file2 (weight = 0.9, strong dependency)
    edge4 = edges.insert({
        '_from': 'test_nodes/file1',
        '_to': 'test_nodes/file2',
        'relationship': 'imports',
        'weight': 0.9,
        'edge_type': 'dependency',
        'import_name': 'module2'
    })
    
    print("\nCreated edges with weights:")
    print(f"- Directory -> File1: weight = 1.0 (structural)")
    print(f"- Directory -> File2: weight = 1.0 (structural)")
    print(f"- File1 <-> File2: weight = 0.85 (semantic similarity)")
    print(f"- File1 -> File2: weight = 0.9 (import dependency)")
    
    # Query to show we can traverse with edge values
    print("\nQuerying edges with weights:")
    
    query = """
    FOR edge IN test_edges
    RETURN {
        from: edge._from,
        to: edge._to,
        relationship: edge.relationship,
        weight: edge.weight,
        type: edge.edge_type
    }
    """
    
    cursor = db.aql.execute(query)
    for edge in cursor:
        print(f"  {edge['from']} --[{edge['relationship']}:{edge['weight']}]--> {edge['to']}")
    
    # Example: Find strongly connected nodes (weight > 0.8)
    print("\nFinding strongly connected nodes (weight > 0.8):")
    
    strong_query = """
    FOR edge IN test_edges
    FILTER edge.weight > 0.8
    RETURN {
        from: edge._from,
        to: edge._to,
        weight: edge.weight,
        relationship: edge.relationship
    }
    """
    
    cursor = db.aql.execute(strong_query)
    for edge in cursor:
        print(f"  Strong connection: {edge['from']} -> {edge['to']} (weight: {edge['weight']})")
    
    # Demonstrate updating edge weights
    print("\nUpdating edge weight based on usage...")
    
    # Simulate increasing weight due to frequent traversal
    update_query = """
    FOR edge IN test_edges
    FILTER edge._from == 'test_nodes/file1' AND edge._to == 'test_nodes/file2' AND edge.relationship == 'similar_to'
    UPDATE edge WITH { weight: edge.weight * 1.05, traversal_count: 1 } IN test_edges
    RETURN NEW
    """
    
    cursor = db.aql.execute(update_query)
    for updated in cursor:
        print(f"  Updated similarity weight to: {updated['weight']}")
    
    print("\n✓ Successfully created graph with custom edge values!")
    print("✓ Demonstrated querying and updating edge weights!")
    
    # Cleanup
    cleanup = input("\nCleanup test collections? (y/n): ")
    if cleanup.lower() == 'y':
        db.delete_collection('test_nodes')
        db.delete_collection('test_edges')
        print("Test collections removed.")

if __name__ == "__main__":
    try:
        test_graph_with_edge_values()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure ArangoDB is running and accessible at localhost:8529")