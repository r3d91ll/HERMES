#!/usr/bin/env python3
"""
Setup ArangoDB with dimensional edge collections for multi-dimensional graph testing.
Creates separate collections for WHERE, WHAT, WHO dimensions.
"""

from arango import ArangoClient
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def setup_dimensional_database():
    """Create database with dimensional edge collections."""
    
    # Get credentials from environment
    host = os.getenv('ARANGO_HOST', 'localhost')
    port = int(os.getenv('ARANGO_PORT', 8529))
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD', '')
    db_name = os.getenv('ARANGO_DATABASE', 'hermes')
    
    print(f"Connecting to ArangoDB at {host}:{port}...")
    
    # Connect to ArangoDB
    client = ArangoClient(hosts=f"http://{host}:{port}")
    
    try:
        # Connect to system database first
        sys_db = client.db("_system", username=username, password=password)
        
        # Create database if it doesn't exist
        if not sys_db.has_database(db_name):
            print(f"Creating database '{db_name}'...")
            sys_db.create_database(db_name)
            print(f"✓ Database '{db_name}' created")
        else:
            print(f"✓ Database '{db_name}' already exists")
        
        # Connect to our database
        db = client.db(db_name, username=username, password=password)
        
        # Create collections
        collections_to_create = [
            # Node collection
            ('nodes', False),
            
            # Dimensional edge collections
            ('edges_where', True),   # Spatial/location relationships
            ('edges_what', True),    # Semantic/content relationships  
            ('edges_who', True),     # Permission/ownership relationships
            
            # Composite edge collection for multiplicative results
            ('edges_composite', True)
        ]
        
        for collection_name, is_edge in collections_to_create:
            if not db.has_collection(collection_name):
                print(f"Creating collection '{collection_name}' (edge={is_edge})...")
                db.create_collection(collection_name, edge=is_edge)
                print(f"✓ Collection '{collection_name}' created")
            else:
                print(f"✓ Collection '{collection_name}' already exists")
        
        # Create indexes for better query performance
        print("\nCreating indexes...")
        
        # Node indexes
        nodes = db.collection('nodes')
        nodes.add_hash_index(fields=['type'], unique=False)
        nodes.add_hash_index(fields=['path'], unique=True)
        print("✓ Created indexes on nodes collection")
        
        # Edge indexes for each dimension
        for dim in ['where', 'what', 'who']:
            edges = db.collection(f'edges_{dim}')
            edges.add_hash_index(fields=['weight'], unique=False)
            edges.add_hash_index(fields=['relationship'], unique=False)
        print("✓ Created indexes on dimensional edge collections")
        
        # Composite edge indexes
        composite = db.collection('edges_composite')
        composite.add_hash_index(fields=['total_weight'], unique=False)
        composite.add_hash_index(fields=['dimensions_active'], unique=False)
        print("✓ Created indexes on composite edge collection")
        
        print("\n✅ Database setup complete!")
        print(f"\nCollection structure:")
        print(f"  - nodes: Store all entities (files, directories, concepts)")
        print(f"  - edges_where: Spatial relationships (weight 0.0-1.0)")
        print(f"  - edges_what: Semantic relationships (weight 0.0-1.0)")
        print(f"  - edges_who: Permission relationships (weight 0.0-1.0)")
        print(f"  - edges_composite: Multiplied dimensional strengths")
        
        return db
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    db = setup_dimensional_database()
    
    # Show example of how dimensional edges work
    print("\n" + "="*60)
    print("Example dimensional edge structure:")
    print("="*60)
    
    print("""
WHERE edge (spatial):
{
    "_from": "nodes/file1",
    "_to": "nodes/file2",
    "weight": 1.0,  // Same directory = strong spatial connection
    "relationship": "same_directory",
    "distance": 0   // Path distance
}

WHAT edge (semantic):
{
    "_from": "nodes/file1", 
    "_to": "nodes/file2",
    "weight": 0.85,  // High semantic similarity
    "relationship": "similar_content",
    "similarity_type": "cosine"
}

WHO edge (permissions):
{
    "_from": "nodes/file1",
    "_to": "nodes/file2", 
    "weight": 1.0,  // Same owner = full permission alignment
    "relationship": "same_owner",
    "owner": "todd"
}

COMPOSITE edge (WHERE × WHAT × WHO):
{
    "_from": "nodes/file1",
    "_to": "nodes/file2",
    "total_weight": 0.85,  // 1.0 × 0.85 × 1.0 = 0.85
    "where_weight": 1.0,
    "what_weight": 0.85,
    "who_weight": 1.0,
    "dimensions_active": 3
}
""")
    
    print("\nReady to create dimensional graph from directory!")