import chromadb
import time
import requests

# Start total timing
total_start = time.time()

# Start timing for connection
start_connect = time.time()
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("langchain")
connect_time = time.time() - start_connect
print(f"Connection time: {connect_time:.2f} seconds")

# Start timing for query
query = "what is promise"
start_query = time.time()

# Get embeddings from the server
response = requests.post("http://localhost:5000/embed", json={"texts": [query]})
query_embedding = response.json()["embeddings"]

results = collection.query(query_embeddings=query_embedding, n_results=3)
query_time = time.time() - start_query
print(f"Query time: {query_time:.2f} seconds")

# Rest of your code remains the same...

# Print results
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i].strip()
    doc_id = results["ids"][0][i]
    metadata = results["metadatas"][0][i]

    print(f"Result {i + 1}:")
    print(f"Document: {doc}")
    print(f"ID: {doc_id}")
    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print(f"Distance: {results['distances'][0][i]:.4f}")
    print("-" * 50)