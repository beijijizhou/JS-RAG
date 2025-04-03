import os
from pinecone import Pinecone
import numpy as np

# Pinecone setup
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("API key not set. Please set PINECONE_API_KEY.")
pc = Pinecone(api_key=api_key)
index_name = "testing"
index = pc.Index(index_name)
query_payload = {
    "inputs": {
        "text": "what is promise."
    },
    "top_k": 3
}

results = index.search(
    namespace="js-rag",
    query=query_payload
)

print(results)