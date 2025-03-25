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
for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i + 1}:")
    print(doc.strip())
    print("-" * 50)