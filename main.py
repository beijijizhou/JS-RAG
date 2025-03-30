from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb

app = FastAPI()

# Define a Pydantic model for input validation
class TextInput(BaseModel):
    texts: list[str]

# Load the sentence transformer model once when server starts
print("Loading model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# Connect to ChromaDB
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("langchain")
print("Connected to ChromaDB collection 'langchain'.")

@app.post("/embed")
async def embed(input: TextInput):
    # Take the first text input (e.g., "promise")
    query = input.texts[0]
    
    # Generate embedding for the query
    query_embedding = embedder.encode(query, convert_to_tensor=False).tolist()
    
    # Query ChromaDB with the embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    # Format the response
    response = {
        "query": query,
        "results": [
            {
                "document": results["documents"][0][i].strip(),
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]
    }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)