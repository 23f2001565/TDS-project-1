# retriever.py
import faiss, json, numpy as np
from sentence_transformers import SentenceTransformer

import os
print(os.getcwd())  # e.g. "/Users/Shared/TDS-project-1"


class SubthreadRetriever:
    def __init__(self, index_path="../faiss.index", meta_path="../data/metadata.json", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)

    def retrieve(self, query, top_k=3):
        q_emb = self.model.encode(query)
        q_emb = q_emb / np.linalg.norm(q_emb)
        D, I = self.index.search(np.array([q_emb.astype("float32")]), top_k)
        return [self.metadata[i] | {"score": float(D[0][idx])} for idx, i in enumerate(I[0])]
