from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
import uvicorn

app = FastAPI()

# Define a Pydantic model for input validation
class TextInput(BaseModel):
    texts: list[str]

# Load model once when server starts
print("Loading model...")
start = time.time()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded in {time.time() - start:.2f} seconds")

@app.post("/embed")
async def embed(input: TextInput):
    embeddings = embedder.encode(input.texts, convert_to_tensor=False).tolist()
    return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)