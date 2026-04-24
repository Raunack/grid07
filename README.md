# grid07 — AI routing + RAG experiment

Three-phase system I built to explore how you'd actually wire up persona-based
bot routing, autonomous content generation, and combat-style reply threads.
Nothing fancy dependency-wise — just Groq, LangGraph, and numpy.

---

## Setup

```bash
git clone <repo-url>
cd grid07
pip install -r requirements.txt
cp .env.example .env     # drop your GROQ_API_KEY in here
python phase1_router.py
python phase2_content_engine.py
python phase3_combat_engine.py
```

---

## What each phase does

### Phase 1 — persona router (`phase1_router.py`)

At startup, each bot's persona description gets embedded and dropped into a
lightweight in-memory vector store (`InMemoryVectorStore`). When a post comes
in, it gets embedded the same way, and we do cosine similarity against every
persona. Bots that don't clear the threshold (default 0.85) just don't get
triggered — so Bot C (finance bro) won't care about an AI regulation post.

I used the Groq chat API to generate the embeddings by prompting the model to
output a 32-float JSON array. It's a bit hacky compared to a proper embedding
endpoint, but it works fine for a demo. If you're putting this in production
you'd want something like `text-embedding-3-small` for actual dimensionality.

---

### Phase 2 — content engine (`phase2_content_engine.py`)

This one's a LangGraph state machine with three nodes wired up linearly:

```
decide_search → web_search → draft_post → END
```

- **decide_search**: LLM picks a topic and search query based on the bot's
  personality, returns it as JSON
- **web_search**: hits `mock_searxng_search()` — just a keyword-matched dict
  of fake headlines, no actual HTTP calls
- **draft_post**: LLM takes persona + headlines and writes a ≤280 char post,
  also returned as JSON

I force structured output by telling the model to respond with raw JSON only
and stripping any backtick fences before parsing. The 280 char cap is enforced
post-parse since LLMs sometimes just ignore length constraints.

---

### Phase 3 — combat engine + RAG (`phase3_combat_engine.py`)

`generate_defense_reply()` stitches the full thread (original post + all prior
comments) into the system prompt before the model sees the new message. So the
context window is literally doing the retrieval — no external store needed for
thread-length conversations.

#### Prompt injection defense

Two layers:

**Layer 1 — regex pre-filter**: before the LLM touches anything, the incoming
message gets scanned against 10 patterns that cover the common jailbreak stuff:
`ignore all previous instructions`, `you are now`, `act as`, `pretend you are`,
`apologize to me`, etc. If it matches, a warning gets appended to the system
prompt telling the model exactly what's happening.

**Layer 2 — system-level persona lock**: the system prompt always opens with a
block that says these rules can't be overridden by anything the user sends. So
even if something slips past the regex, the model's already been told to treat
any override attempt as adversarial noise.

Both layers together are much harder to beat than either one alone. Not
foolproof, but solid for this use case.

---

## Files

```
grid07/
├── phase1_router.py
├── phase2_content_engine.py
├── phase3_combat_engine.py
├── execution_logs.md
├── requirements.txt
├── .env.example
└── README.md
```

## Env vars

| Variable | What it's for |
|---|---|
| `GROQ_API_KEY` | all LLM calls across all three phases — free key at [console.groq.com](https://console.groq.com) |
