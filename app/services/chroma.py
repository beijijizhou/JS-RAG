import chromadb
# from app.config.settings import settings

# Initialize ChromaDB client
CHROMA_PATH = "/Users/hongzhonghu/Desktop/rag/chroma"
client = chromadb.PersistentClient(path=CHROMA_PATH)
# client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("langchain")
# print(f"Connected to ChromaDB at {settings.chroma_host}:{settings.chroma_port}")
count = collection.count()
print(f"Number of items: {count}")

def query_chroma(embedding: list[float], n_results: int = 3):
    """Query ChromaDB with an embedding and return results."""
    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    return [
        {
            "document": results["documents"][0][i].strip(),
            "id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }
        for i in range(len(results["documents"][0]))
    ]