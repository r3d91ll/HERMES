#!/usr/bin/env python3
"""
Queries to examine the dimensional graph in ArangoDB.
Run these in the ArangoDB Web UI at http://192.168.1.69:8529
"""

# First, let's create a named graph in ArangoDB for visualization
create_graph_query = """
// Create a named graph for visualization
// Run this in ArangoDB Web UI

// First, check if graph exists and remove if needed
var graph_module = require("@arangodb/general-graph");
try {
    graph_module._drop("dimensional_graph", true);
} catch (e) {
    // Graph doesn't exist, that's fine
}

// Create the graph with all edge collections
var graph = graph_module._create("dimensional_graph");

// Add edge definitions
graph._addEdgeDefinition({
    collection: "edges_where",
    from: ["nodes"],
    to: ["nodes"]
});

graph._addEdgeDefinition({
    collection: "edges_what", 
    from: ["nodes"],
    to: ["nodes"]
});

graph._addEdgeDefinition({
    collection: "edges_who",
    from: ["nodes"],
    to: ["nodes"]
});

graph._addEdgeDefinition({
    collection: "edges_composite",
    from: ["nodes"],
    to: ["nodes"]
});

"Graph 'dimensional_graph' created successfully!"
"""

# Query to visualize WHERE dimension (spatial relationships)
where_dimension_query = """
// Visualize WHERE dimension - spatial relationships
FOR node IN nodes
    LET edges = (
        FOR edge IN edges_where
            FILTER edge._from == node._id OR edge._to == node._id
            RETURN edge
    )
    FILTER LENGTH(edges) > 0
    RETURN {
        node: node,
        edges: edges
    }
"""

# Query to visualize WHAT dimension (semantic relationships)
what_dimension_query = """
// Visualize WHAT dimension - semantic relationships
FOR node IN nodes
    LET edges = (
        FOR edge IN edges_what
            FILTER edge._from == node._id OR edge._to == node._id
            RETURN edge
    )
    FILTER LENGTH(edges) > 0
    RETURN {
        node: node,
        edges: edges
    }
"""

# Query to visualize WHO dimension (permission relationships)
who_dimension_query = """
// Visualize WHO dimension - permission relationships
FOR node IN nodes
    LET edges = (
        FOR edge IN edges_who
            FILTER edge._from == node._id OR edge._to == node._id
            RETURN edge
    )
    FILTER LENGTH(edges) > 0
    RETURN {
        node: node,
        edges: edges
    }
"""

# Query to visualize composite connections
composite_query = """
// Visualize strongest composite connections
FOR edge IN edges_composite
    FILTER edge.total_weight > 0.5
    LET from_node = DOCUMENT(edge._from)
    LET to_node = DOCUMENT(edge._to)
    RETURN {
        from: from_node.name,
        to: to_node.name,
        weight: edge.total_weight,
        where: edge.where_weight,
        what: edge.what_weight,
        who: edge.who_weight
    }
"""

# Query for graph visualization in ArangoDB UI
graph_viz_query = """
// For Graph Viewer in ArangoDB Web UI
// Go to Graphs -> dimensional_graph -> Graph Viewer

// Example traversal from a specific node
FOR v, e, p IN 1..3 OUTBOUND 'nodes/YOUR_NODE_KEY_HERE' 
    edges_where, edges_what, edges_who, edges_composite
    RETURN p
"""

# Summary statistics query
stats_query = """
// Graph statistics
LET node_count = LENGTH(nodes)
LET where_edges = LENGTH(edges_where)
LET what_edges = LENGTH(edges_what)
LET who_edges = LENGTH(edges_who)
LET composite_edges = LENGTH(edges_composite)

LET avg_where = (
    FOR e IN edges_where
    RETURN e.weight
) 

LET avg_what = (
    FOR e IN edges_what  
    RETURN e.weight
)

LET avg_who = (
    FOR e IN edges_who
    RETURN e.weight
)

LET avg_composite = (
    FOR e IN edges_composite
    RETURN e.total_weight
)

RETURN {
    nodes: node_count,
    edges: {
        where: where_edges,
        what: what_edges,
        who: who_edges,
        composite: composite_edges,
        total: where_edges + what_edges + who_edges + composite_edges
    },
    average_weights: {
        where: AVERAGE(avg_where),
        what: AVERAGE(avg_what),
        who: AVERAGE(avg_who),
        composite: AVERAGE(avg_composite)
    }
}
"""

print("="*60)
print("ArangoDB Graph Visualization Queries")
print("="*60)
print("\n1. First, create the named graph in ArangoDB Web UI:")
print("-"*60)
print(create_graph_query)
print("\n2. Then go to Graphs -> dimensional_graph -> Graph Viewer")
print("\n3. Use these queries in the Query editor:")
print("-"*60)
print("\nGraph Statistics:")
print(stats_query)
print("\nStrongest Connections:")
print(composite_query)

print("\n" + "="*60)
print("To visualize in ArangoDB Web UI:")
print("1. Go to http://192.168.1.69:8529")
print("2. Login with your credentials")
print("3. Go to Queries section")
print("4. Run the create graph query first")
print("5. Then go to Graphs -> dimensional_graph")
print("6. Use the Graph Viewer to explore visually")
print("="*60)