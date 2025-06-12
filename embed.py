# Reshav Sharma-2025-TDS-PROJECT 1: tds_virtual_ta/embed.py

import numpy as np
import json

# Load precomputed embeddings
def load_embeddings():
    return np.load("data/embeddings.npy")

# Load metadata
def load_metadata():
    with open("data/metadata.json") as f:
        return json.load(f)
