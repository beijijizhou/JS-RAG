import os
import numpy as np
from pinecone import Pinecone

# Fetch API key from environment variables
api_key = os.getenv("PINECONE_API_KEY")

# Check if the API key is loaded
if not api_key:
    raise ValueError("API key not set. Please set PINECONE_API_KEY.")

print(f"API Key: {api_key}")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Specify your existing index name
index_name = "testing"

# Check available indexes (optional, for verification)
existing_indexes = pc.list_indexes().names()
print(f"Available indexes: {existing_indexes}")

if index_name not in existing_indexes:
    print(f"Index {index_name} not found. Please check the name or create it.")
else:
    # Connect to the existing index
    index = pc.Index(index_name)
    print(f"Connected to index: {index_name}")

    # Verify the connection by fetching index stats
    index_stats = index.describe_index_stats()
    print(f"Index stats: {index_stats}")