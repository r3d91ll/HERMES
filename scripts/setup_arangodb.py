#!/usr/bin/env python3
"""
Setup ArangoDB database and collections for HADES MVP.
Run this after ArangoDB is installed and accessible.
"""

from arango import ArangoClient
import sys
from getpass import getpass


def setup_database(host="localhost", port=8529, username="root", password=None):
    """Create HADES MVP database and collections."""
    
    if password is None:
        password = getpass("Enter ArangoDB root password: ")
    
    # Connect to ArangoDB
    client = ArangoClient(hosts=f"http://{host}:{port}")
    
    try:
        # Connect as root to system database
        sys_db = client.db("_system", username=username, password=password)
        
        # Create hades_mvp database if it doesn't exist
        db_name = "hades_mvp"
        if not sys_db.has_database(db_name):
            print(f"Creating database '{db_name}'...")
            sys_db.create_database(db_name)
            print(f"✓ Database '{db_name}' created successfully")
        else:
            print(f"Database '{db_name}' already exists")
        
        # Connect to hades_mvp database
        db = client.db(db_name, username=username, password=password)
        
        # Create collections
        collections = {
            "nodes": {
                "type": "document",
                "schema": {
                    "rule": {
                        "properties": {
                            "node_id": {"type": "string"},
                            "node_type": {"type": "string", "enum": ["document", "query", "synthetic"]},
                            "what_vector": {"type": "array"},
                            "conveyance_vector": {"type": "array"},
                            "where_vector": {"type": "array"},
                            "content_preview": {"type": "string"},
                            "metadata": {"type": "object"},
                            "creation_time": {"type": "string"},
                            "source_path": {"type": "string"}
                        },
                        "required": ["node_id", "node_type", "what_vector", "conveyance_vector", "where_vector"]
                    }
                }
            },
            "edges": {
                "type": "edge",
                "schema": {
                    "rule": {
                        "properties": {
                            "_from": {"type": "string"},
                            "_to": {"type": "string"},
                            "edge_type": {"type": "string", "enum": ["semantic", "structural", "derived"]},
                            "base_weight": {"type": "number"},
                            "context_amplification": {"type": "number"},
                            "traversal_count": {"type": "integer"}
                        },
                        "required": ["_from", "_to", "edge_type", "base_weight"]
                    }
                }
            }
        }
        
        for coll_name, coll_config in collections.items():
            if not db.has_collection(coll_name):
                print(f"Creating {coll_config['type']} collection '{coll_name}'...")
                if coll_config['type'] == 'edge':
                    db.create_collection(coll_name, edge=True)
                else:
                    db.create_collection(coll_name)
                print(f"✓ Collection '{coll_name}' created")
            else:
                print(f"Collection '{coll_name}' already exists")
        
        # Create indexes
        print("\nCreating indexes...")
        
        # Index on node_id for fast lookups
        nodes_coll = db.collection('nodes')
        nodes_coll.add_hash_index(fields=['node_id'], unique=True)
        print("✓ Created unique index on nodes.node_id")
        
        # Index on node_type for filtering
        nodes_coll.add_hash_index(fields=['node_type'])
        print("✓ Created index on nodes.node_type")
        
        # Create a view for vector similarity search (placeholder)
        print("\nNote: Vector similarity search will be implemented using AQL queries")
        print("Consider using ArangoDB's built-in vector search capabilities in production")
        
        print("\n✅ ArangoDB setup complete!")
        print(f"\nDatabase: {db_name}")
        print("Collections: nodes (document), edges (edge)")
        print("\nYou can now access the database at:")
        print(f"  http://{host}:{port}")
        print(f"  Database: {db_name}")
        print(f"  Username: {username}")
        
        # Save connection info
        print("\nSaving connection template to .env.template...")
        with open(".env.template", "w") as f:
            f.write(f"""# ArangoDB Configuration
ARANGO_HOST={host}
ARANGO_PORT={port}
ARANGO_USERNAME={username}
ARANGO_PASSWORD=your_password_here
ARANGO_DATABASE={db_name}

# Redis Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Dimensional Configuration
WHAT_DIMS=1024
CONVEYANCE_DIMS=922
WHERE_DIMS=102
""")
        print("✓ Created .env.template")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("HADES MVP - ArangoDB Setup")
    print("=" * 40)
    
    # You can customize these or pass as arguments
    setup_database()