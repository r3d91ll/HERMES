#!/usr/bin/env python3
"""
Minimal test script for creating a graph from a directory with edge values.
Tests the core functionality of creating nodes and edges with custom values.
"""

import asyncio
from pathlib import Path
from hermes.storage.arango_storage import ArangoStorage
from hermes.core.directory_traverser import DirectoryTraverser
from hermes.loaders import PythonLoader, TextLoader
from rich.console import Console
from rich.table import Table
import sys

console = Console()

async def test_minimal_graph_creation():
    """Test creating a small graph from a single directory."""
    
    # Initialize storage
    console.print("[bold blue]Initializing ArangoDB storage...[/bold blue]")
    storage = ArangoStorage()
    
    # Choose a small test directory - let's use the hermes/loaders directory
    test_directory = Path("/home/todd/olympus/hermes/hermes/loaders")
    
    if not test_directory.exists():
        console.print(f"[red]Error: Directory {test_directory} does not exist![/red]")
        return
    
    console.print(f"[green]Testing with directory: {test_directory}[/green]")
    
    # Test 1: Create nodes manually with custom properties
    console.print("\n[bold yellow]Test 1: Creating nodes manually[/bold yellow]")
    
    # Create a test node
    test_node_1 = {
        "_key": "test_node_1",
        "type": "test",
        "name": "Test Node 1",
        "metadata": {
            "custom_value": 42,
            "description": "First test node"
        }
    }
    
    test_node_2 = {
        "_key": "test_node_2", 
        "type": "test",
        "name": "Test Node 2",
        "metadata": {
            "custom_value": 84,
            "description": "Second test node"
        }
    }
    
    # Store nodes
    node1_id = storage.store_document("test_node_1", test_node_1)
    node2_id = storage.store_document("test_node_2", test_node_2)
    console.print(f"Created nodes: {node1_id}, {node2_id}")
    
    # Test 2: Create edges with custom values
    console.print("\n[bold yellow]Test 2: Creating edges with custom values[/bold yellow]")
    
    # Create edge with custom weight
    edge_data = {
        "_from": f"nodes/{node1_id}",
        "_to": f"nodes/{node2_id}",
        "relationship": "test_relationship",
        "edge_type": "custom",
        "weight": 0.75,  # Custom edge weight
        "metadata": {
            "strength": 0.9,
            "confidence": 0.85,
            "custom_property": "test_value"
        }
    }
    
    edge_id = storage.create_edge(edge_data)
    console.print(f"Created edge with weight {edge_data['weight']}: {edge_id}")
    
    # Test 3: Use directory traverser to create graph from directory
    console.print("\n[bold yellow]Test 3: Creating graph from directory[/bold yellow]")
    
    # Setup loaders
    loaders = {
        ".py": PythonLoader(),
        ".txt": TextLoader(),
    }
    
    # Create traverser
    traverser = DirectoryTraverser(
        storage=storage,
        loaders=loaders,
        analyze_conveyance=False  # Keep it simple for now
    )
    
    # Traverse directory and build graph
    stats = traverser.traverse_and_build(test_directory)
    
    console.print(f"[green]Directory traversal complete![/green]")
    console.print(f"Files processed: {stats['files_processed']}")
    console.print(f"Edges created: {stats['edges_created']}")
    
    # Test 4: Query and display the graph
    console.print("\n[bold yellow]Test 4: Querying the graph[/bold yellow]")
    
    # Query all edges
    query = """
    FOR edge IN edges
    LIMIT 10
    RETURN {
        from: edge._from,
        to: edge._to,
        weight: edge.weight,
        relationship: edge.relationship,
        edge_type: edge.edge_type
    }
    """
    
    cursor = storage.db.aql.execute(query)
    edges = list(cursor)
    
    # Display edges in a table
    table = Table(title="Sample Edges from Graph")
    table.add_column("From", style="cyan")
    table.add_column("To", style="cyan")
    table.add_column("Weight", style="green")
    table.add_column("Relationship", style="yellow")
    table.add_column("Type", style="magenta")
    
    for edge in edges:
        table.add_row(
            edge.get('from', 'N/A').split('/')[-1][:20],
            edge.get('to', 'N/A').split('/')[-1][:20],
            str(edge.get('weight', 'N/A')),
            edge.get('relationship', 'N/A'),
            edge.get('edge_type', 'N/A')
        )
    
    console.print(table)
    
    # Test 5: Create edges with different weights based on relationships
    console.print("\n[bold yellow]Test 5: Creating edges with varied weights[/bold yellow]")
    
    # Create semantic similarity edge
    semantic_edge = {
        "_from": f"nodes/{node1_id}",
        "_to": f"nodes/{node2_id}",
        "relationship": "semantic_similarity",
        "edge_type": "semantic",
        "weight": 0.92,  # High similarity
        "metadata": {
            "similarity_type": "cosine",
            "embedding_model": "jina-v4"
        }
    }
    
    storage.create_edge(semantic_edge)
    console.print(f"Created semantic edge with weight: {semantic_edge['weight']}")
    
    # Create structural edge
    structural_edge = {
        "_from": f"nodes/{node1_id}",
        "_to": f"nodes/{node2_id}",
        "relationship": "references",
        "edge_type": "structural",
        "weight": 1.0,  # Certain connection
        "metadata": {
            "reference_type": "import",
            "line_number": 42
        }
    }
    
    storage.create_edge(structural_edge)
    console.print(f"Created structural edge with weight: {structural_edge['weight']}")
    
    console.print("\n[bold green]✓ All tests completed successfully![/bold green]")
    console.print("\n[yellow]Summary:[/yellow]")
    console.print("- Created nodes with custom properties ✓")
    console.print("- Created edges with custom weights ✓")
    console.print("- Built graph from directory structure ✓")
    console.print("- Assigned different edge values based on relationship types ✓")
    
    return True

async def cleanup_test_nodes():
    """Clean up test nodes created during the test."""
    storage = ArangoStorage()
    
    # Remove test nodes
    try:
        storage.db.collection('nodes').delete('test_node_1')
        storage.db.collection('nodes').delete('test_node_2')
        console.print("[dim]Cleaned up test nodes[/dim]")
    except:
        pass

if __name__ == "__main__":
    try:
        # Run the test
        success = asyncio.run(test_minimal_graph_creation())
        
        if success:
            console.print("\n[bold green]Graph creation functionality test passed![/bold green]")
            console.print("You can now create objects and assign values to edges.")
            
            # Cleanup
            asyncio.run(cleanup_test_nodes())
    except Exception as e:
        console.print(f"[bold red]Error during test: {e}[/bold red]")
        sys.exit(1)