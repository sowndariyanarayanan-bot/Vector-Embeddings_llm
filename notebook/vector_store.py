import os
import chromadb
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
folder = "dataset/documents"

documents = []
ids = []
metadatas = []

for i, filename in enumerate(sorted(os.listdir(folder))):

    if filename.endswith(".txt"):

        with open(os.path.join(folder, filename),
                  "r", encoding="utf-8") as file:

            documents.append(file.read())

        ids.append(f"doc_{i+1}")

        metadatas.append({
            "title": filename,
            "category": "AI and Education",
            "source": filename
        })

# Generate embeddings
embeddings = model.encode(documents)

# Create ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="ai_education_documents",
    metadata={"hnsw:space": "cosine"}
)

# Store documents
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Vector database created successfully!")
print("Total documents:", collection.count())