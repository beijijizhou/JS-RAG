import sys
import os

from fastapi import FastAPI
from app.api.endpoints import router as api_router


# Add the parent directory (rag/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API with ChromaDB",
    version="1.0.0"
)

# Include API routes
app.include_router(api_router)

# Optional health check
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

