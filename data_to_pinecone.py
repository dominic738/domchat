import json
import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index_name = "dominic-personal-data"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

all_chunks = []
chunk_files = [
    'identity_chunks.json',
    'projects_chunks.json',
    'experience_chunks.json',
    'hobbies_chunks.json',
    'project_details_chunks.json'
]

for filename in chunk_files:
    with open(f'./RAG/{filename}', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {filename}")

print(f"\n Total chunks to upload: {len(all_chunks)}")


vectors = []
for i, chunk in enumerate(all_chunks):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk['text']
    )
    embedding = response.data[0].embedding
    
    vectors.append({
        "id": chunk['id'],
        "values": embedding,
        "metadata": {"text": chunk['text']}
    })
    
    print(f"Embedded {i+1}/{len(all_chunks)}: {chunk['id']}")


index.upsert(vectors=vectors)
print(f"\n Successfully uploaded {len(vectors)} vectors to Pinecone!")