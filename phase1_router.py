"""
Phase 1: persona router

Embeds each bot persona at startup and stores them in a lightweight cosine
similarity store. Incoming posts get embedded the same way, then we just
compare and filter by threshold. Bots below the cutoff don't respond.

Using the Groq chat API to generate embeddings by prompting for a 32-float
JSON array — not ideal for production but works well enough here.
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"C:\Users\Ronack\Desktop\grid07\.env")

import os
import json
import numpy as np
import requests


# bot definitions — description is what gets embedded
BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. I am highly "
            "optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying "
            "society. I am highly critical of AI, social media, and billionaires. "
            "I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, "
            "and making money. I speak in finance jargon and view everything "
            "through the lens of ROI."
        ),
    },
}


# ── embeddings ───────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """
    Calls Groq and asks the model to output a 32-float semantic vector for
    the given text. Hacky but it works for this demo — swap for a real
    embedding endpoint if you need higher accuracy.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "llama-3.1-8b-instant",
        "max_tokens": 256,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a semantic embedding engine. "
                    "When given text, respond ONLY with a JSON array of exactly 32 floats "
                    "in the range [-1, 1] that semantically represent the text. "
                    "No explanation, no markdown — raw JSON array only."
                ),
            },
            {"role": "user", "content": f"Embed this text: {text}"},
        ],
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # model sometimes wraps output in backtick fences, strip them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    vector = np.array(json.loads(raw.strip()), dtype=float)
    FIXED_DIM = 64
    if len(vector) >= FIXED_DIM:
        vector = vector[:FIXED_DIM]
    else:
        vector = np.pad(vector, (0, FIXED_DIM - len(vector)))
    return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── vector store ─────────────────────────────────────────────────────────────

class InMemoryVectorStore:
    """
    Dead simple cosine-similarity store. No external deps, just a list of
    dicts. Good enough to mimic pgvector/Chroma/FAISS for this use case.
    """

    def __init__(self):
        self._store: list[dict] = []

    def add(self, doc_id: str, metadata: dict, vector: np.ndarray):
        self._store.append({"id": doc_id, "metadata": metadata, "vector": vector})
        print(f"  [VectorStore] stored '{doc_id}' (dim={len(vector)})")

    def query(self, query_vector: np.ndarray, threshold: float) -> list[dict]:
        """Return all entries with cosine similarity >= threshold, sorted desc."""
        results = []
        for entry in self._store:
            sim = cosine_similarity(query_vector, entry["vector"])
            results.append({
                **entry["metadata"],
                "bot_id": entry["id"],
                "similarity": round(sim, 4),
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return [r for r in results if r["similarity"] >= threshold]


# ── startup ───────────────────────────────────────────────────────────────────

def build_persona_store() -> InMemoryVectorStore:
    """Embed all personas and load into the store. Call this once at startup."""
    store = InMemoryVectorStore()
    print("\n[Phase 1] building persona store...")
    for bot_id, persona in BOT_PERSONAS.items():
        print(f"  embedding {bot_id} ({persona['name']})...")
        vec = get_embedding(persona["description"])
        store.add(bot_id, {"name": persona["name"], "description": persona["description"]}, vec)
    print("[Phase 1] done.\n")
    return store


# ── routing ───────────────────────────────────────────────────────────────────

def route_post_to_bots(
    post_content: str,
    store: InMemoryVectorStore,
    threshold: float = 0.85,
) -> list[dict]:
    """
    Embed the post, run it against the persona store, return matched bots.
    Bots below threshold are filtered out entirely.
    """
    print(f"[Router] post: \"{post_content}\"")
    post_vec = get_embedding(post_content)
    matches = store.query(post_vec, threshold)

    if matches:
        print(f"[Router] {len(matches)} match(es):")
        for m in matches:
            print(f"  → {m['bot_id']} ({m['name']})  sim={m['similarity']}")
    else:
        print(f"[Router] nothing matched above {threshold}")

    return matches


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    store = build_persona_store()

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits new all-time high amid regulatory ETF approvals.",
        "Big Tech companies are buying up farmland and displacing rural communities.",
    ]

    for post in test_posts:
        print("=" * 70)
        route_post_to_bots(post, store, threshold=0.3)
        print()
