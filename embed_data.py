# embed_index.py
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm

def clean_text(text): return " ".join(text.strip().split())

def normalize(v): return v / np.linalg.norm(v)

def build_threads(posts):
    reply_map = defaultdict(list)
    posts_by_number = {p["post_number"]: p for p in posts}
    for p in posts:
        reply_map[p.get("reply_to_post_number")].append(p)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    collected = []
    def dfs(post_num):
        p = posts_by_number[post_num]
        collected.append(p)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

def create_index(json_path="data/raw_data.json", model_name="all-MiniLM-L6-v2", out_index="faiss.index", out_meta="data/metadata.json"):
    with open(json_path) as f: data = json.load(f)
    model = SentenceTransformer(model_name)

    # Group by topic
    topics = defaultdict(lambda: {"topic_title": "", "posts": []})
    for p in data:
        topics[p["topic_id"]]["posts"].append(p)
        topics[p["topic_id"]]["topic_title"] = p["topic_title"]

    embeddings, metadata = [], []

    print("🔄 Building embeddings...")
    for tid, topic in tqdm(topics.items()):
        topic_title = topic["topic_title"]
        posts = sorted(topic["posts"], key=lambda p: p["post_number"])
        reply_map, posts_by_number = build_threads(posts)

        for root in reply_map[None]:
            subthread = extract_subthread(root["post_number"], reply_map, posts_by_number)
            combined_text = f"Topic title: {topic_title}\n\n" + "\n\n---\n\n".join(clean_text(p["content"]) for p in subthread)
            emb = normalize(model.encode(combined_text))
            embeddings.append(emb)
            metadata.append({
                "topic_id": tid,
                "topic_title": topic_title,
                "post_numbers": [p["post_number"] for p in subthread],
                "root_post_number": root["post_number"],
                "combined_text": combined_text
            })

    index = faiss.IndexFlatIP(len(embeddings[0]))
    index.add(np.vstack(embeddings).astype("float32"))
    faiss.write_index(index, out_index)
    with open(out_meta, "w") as f: json.dump(metadata, f)
    print(f"✅ Saved {len(metadata)} subthreads to {out_index} and {out_meta}")

if __name__ == "__main__":
    create_index()
