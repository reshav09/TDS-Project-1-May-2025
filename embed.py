# Reshav Sharma-2025-TDS-PROJECT 1: tds_virtual_ta/embed.py

import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBED_DIM = 1536

client = OpenAI(
    api_key=os.environ.get("AIPROXY_TOKEN"),
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)

# Load vector index
def load_index(index_path="vector_store.faiss"):
    return faiss.read_index(index_path)

# Load metadata
def load_metadata(meta_path="data/metadata.json"):
    with open(meta_path) as f:
        return json.load(f)

# Generate embedding for a given text
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype='float32')
    


