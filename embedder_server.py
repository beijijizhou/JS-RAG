from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# Load model once when server starts
print("Loading model...")
start = time.time()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded in {time.time() - start:.2f} seconds")

@app.route('/embed', methods=['POST'])
def embed():
    texts = request.json.get('texts', [])
    embeddings = embedder.encode(texts, convert_to_tensor=False).tolist()
    return jsonify({"embeddings": embeddings})

if __name__ == '__main__':
    app.run(port=5000)