import chromadb
import pinecone
import numpy as np

# Configuration
CHROMA_PATH = "/Users/hongzhonghu/Desktop/rag/chroma"
INDEX_NAME = "js-vector-db"
PINECONE_API_KEY = ""  # Replace with your actual API key
ENVIRONMENT = "gcp-starter"  # Free-tier environment
BATCH_SIZE = 100  # Adjust batch size as needed

# Initialize ChromaDB client
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("langchain")

# Get all stored embeddings, IDs, and metadata
all_items = collection.get(include=["embeddings", "metadatas", "documents", "ids"])
embeddings = all_items["embeddings"]
ids = all_items["ids"]
metadata = all_items["metadatas"]

print(f"Extracted {len(embeddings)} vectors from ChromaDB")

# Initialize Pinecone
print("Initializing Pinecone...")
pinecone.init(api_key=PINECONE_API_KEY, environment=ENVIRONMENT)

# Create index if not exists
if INDEX_NAME not in pinecone.list_indexes():
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pinecone.create_index(name=INDEX_NAME, dimension=len(embeddings[0]), metric="cosine")

# Connect to the index
index = pinecone.Index(INDEX_NAME)
print("Connected to Pinecone.")

# Upload vectors in batches
print("Uploading vectors to Pinecone...")
for i in range(0, len(embeddings), BATCH_SIZE):
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_vectors = embeddings[i:i+BATCH_SIZE]
    batch_metadata = metadata[i:i+BATCH_SIZE]

    # Format data for Pinecone upsert
    pinecone_data = [
        (str(batch_ids[j]), np.array(batch_vectors[j]).tolist(), batch_metadata[j])
        for j in range(len(batch_ids))
    ]

    # Upsert batch into Pinecone
    index.upsert(vectors=pinecone_data)
    print(f"Uploaded {i+len(batch_ids)}/{len(embeddings)} vectors")

# Verify upload
print(f"Total vectors in Pinecone: {index.describe_index_stats()}")
print("Upload complete!")
