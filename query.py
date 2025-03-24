from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer

# Connect to the persistent database
client = Client(Settings(
    persist_directory="/Users/hongzhonghu/Desktop/rag/chroma_db",
    is_persistent=True
))
collection = client.get_collection("mdn_js_docs")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Query with more context
query = "JavaScript hoisting behavior with var and let"
query_embedding = embedder.encode([query], convert_to_tensor=False).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=3)

# Print results cleanly
for i, doc in enumerate(results["documents"][0]):  # Access the first query's results
    print(f"Result {i + 1}:")
    print(doc.strip())
    print("-" * 50)