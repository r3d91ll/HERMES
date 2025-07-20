#!/usr/bin/env python3
"""Test ArangoDB connection with new network configuration."""

from arango import ArangoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get connection details from .env
host = os.getenv('ARANGO_HOST', 'localhost')
port = int(os.getenv('ARANGO_PORT', 8529))
username = os.getenv('ARANGO_USERNAME', 'root')
password = os.getenv('ARANGO_PASSWORD', '')
db_name = os.getenv('ARANGO_DATABASE', 'hermes')

print(f"Testing connection to ArangoDB at {host}:{port}...")

try:
    # Connect to ArangoDB
    client = ArangoClient(hosts=f"http://{host}:{port}")
    
    # Test system database connection
    sys_db = client.db("_system", username=username, password=password)
    version = sys_db.version()
    print(f"✓ Connected to ArangoDB version: {version}")
    
    # Connect to our database
    db = client.db(db_name, username=username, password=password)
    
    # List collections
    collections = db.collections()
    print(f"\n✓ Connected to database: {db_name}")
    print(f"Collections found: {len(collections)}")
    
    # Check our dimensional collections
    expected_collections = ['nodes', 'edges_where', 'edges_what', 'edges_who', 'edges_composite']
    for coll_name in expected_collections:
        if db.has_collection(coll_name):
            coll = db.collection(coll_name)
            count = coll.count()
            print(f"  - {coll_name}: {count} documents")
        else:
            print(f"  - {coll_name}: NOT FOUND")
    
    print("\n✅ Connection test successful!")
    print(f"ArangoDB is accessible at http://{host}:{port}")
    
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check if ArangoDB is running: systemctl status arangodb3")
    print("2. Check if it's listening on the right interface: netstat -luntp | grep 8529")
    print("3. Verify credentials in .env file")
    print("4. Test with curl: curl http://192.168.1.69:8529/_api/version")