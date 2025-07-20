#!/usr/bin/env python3
"""Create a named graph in ArangoDB for visualization."""

from arango import ArangoClient
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to ArangoDB
host = os.getenv('ARANGO_HOST', 'localhost')
port = int(os.getenv('ARANGO_PORT', 8529))
username = os.getenv('ARANGO_USERNAME', 'root')
password = os.getenv('ARANGO_PASSWORD', '')
db_name = os.getenv('ARANGO_DATABASE', 'hermes')

client = ArangoClient(hosts=f"http://{host}:{port}")
db = client.db(db_name, username=username, password=password)

print("Creating named graph for visualization...")

# Check if graph exists
if db.has_graph('dimensional_graph'):
    print("Graph 'dimensional_graph' already exists, deleting...")
    db.delete_graph('dimensional_graph')

# Create the graph
graph = db.create_graph('dimensional_graph')

# Add edge definitions
edge_definitions = [
    {
        'edge_collection': 'edges_where',
        'from_vertex_collections': ['nodes'],
        'to_vertex_collections': ['nodes']
    },
    {
        'edge_collection': 'edges_what',
        'from_vertex_collections': ['nodes'],
        'to_vertex_collections': ['nodes']
    },
    {
        'edge_collection': 'edges_who',
        'from_vertex_collections': ['nodes'],
        'to_vertex_collections': ['nodes']
    },
    {
        'edge_collection': 'edges_composite',
        'from_vertex_collections': ['nodes'],
        'to_vertex_collections': ['nodes']
    }
]

for edge_def in edge_definitions:
    graph.create_edge_definition(**edge_def)
    print(f"✓ Added edge definition for {edge_def['edge_collection']}")

print("\n✅ Graph 'dimensional_graph' created successfully!")
print("\nYou can now visualize it in ArangoDB Web UI:")
print(f"1. Go to http://{host}:{port}")
print("2. Login with your credentials")
print("3. Navigate to 'Graphs' in the left menu")
print("4. Click on 'dimensional_graph'")
print("5. Click 'Graph Viewer' to see the visualization")

# Run some quick stats
stats_query = """
LET node_count = LENGTH(nodes)
LET where_edges = LENGTH(edges_where)
LET what_edges = LENGTH(edges_what)
LET who_edges = LENGTH(edges_who)
LET composite_edges = LENGTH(edges_composite)

RETURN {
    nodes: node_count,
    edges: {
        where: where_edges,
        what: what_edges,
        who: who_edges,
        composite: composite_edges,
        total: where_edges + what_edges + who_edges + composite_edges
    }
}
"""

cursor = db.aql.execute(stats_query)
stats = cursor.next()

print("\nCurrent graph statistics:")
print(f"- Nodes: {stats['nodes']}")
print(f"- WHERE edges: {stats['edges']['where']}")
print(f"- WHAT edges: {stats['edges']['what']}")
print(f"- WHO edges: {stats['edges']['who']}")
print(f"- Composite edges: {stats['edges']['composite']}")
print(f"- Total edges: {stats['edges']['total']}")