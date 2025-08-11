from chromadb import Client

def test_chroma():
    client = Client()
    collection = client.get_or_create_collection(name="test_collection")

    # Add sample data
    collection.add(
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["test document"],
        metadatas=[{"source": "test"}],
        ids=["test1"]
    )

    # Retrieve data by id
    results = collection.get(ids=["test1"])
    print("Retrieved documents:", results['documents'])

if __name__ == "__main__":
    test_chroma()
