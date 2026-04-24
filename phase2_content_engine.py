"""
Phase 2: autonomous content engine

LangGraph state machine — three nodes, runs linearly:
  1. decide_search  → LLM picks a topic + search query for the bot
  2. web_search     → mock_searxng_search tool (no real HTTP, just keyword matching)
  3. draft_post     → LLM writes a ≤280 char opinionated post grounded in the results
"""

from dotenv import load_dotenv
load_dotenv(dotenv_path=r"C:\Users\Ronack\Desktop\grid07\.env")

import os
import json
from typing import TypedDict, Annotated

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ── mock news db ──────────────────────────────────────────────────────────────

MOCK_NEWS_DB = {
    "crypto":      "Bitcoin hits new all-time high amid regulatory ETF approvals; Ethereum Layer-2 volume surges 400%.",
    "bitcoin":     "Bitcoin hits new all-time high amid regulatory ETF approvals; MicroStrategy doubles BTC holdings.",
    "ai":          "OpenAI releases GPT-5 with real-time reasoning; Anthropic Claude 4 beats benchmarks across the board.",
    "openai":      "OpenAI releases GPT-5 with real-time reasoning; company valuation crosses $300B.",
    "elon":        "Elon Musk's xAI raises $6B Series B; X platform ad revenue rebounds 30% YoY.",
    "space":       "SpaceX Starship completes first commercial Mars resupply mission simulation.",
    "tech":        "Big Tech stocks rally as Fed signals rate cuts; NVIDIA leads with +8% single-day gain.",
    "regulation":  "EU AI Act enforcement begins; US Congress deadlocked on federal AI oversight bill.",
    "privacy":     "Meta fined €1.2B for GDPR violations; Signal surpasses 200M active users.",
    "climate":     "Global CO₂ levels hit record high; IEA warns renewable buildout pace insufficient.",
    "capitalism":  "Wealth inequality hits historic peak; top 1% now controls 46% of global assets.",
    "billionaire": "Bezos, Musk, and Zuckerberg combined net worth exceeds GDP of 120 countries.",
    "market":      "S&P 500 breaks 6,000; Fed minutes signal two rate cuts in H2 2025.",
    "stocks":      "Mag-7 stocks surge on AI earnings beat; options market pricing in continued volatility.",
    "rates":       "10-year Treasury yield at 4.3%; mortgage rates ease slightly to 6.8%.",
    "trading":     "Quant funds outperform in choppy market; algorithmic trading accounts for 72% of NYSE volume.",
    "roi":         "Private equity IRRs compress as exit multiples decline; LPs demand liquidity solutions.",
    "ev":          "Tesla Model Y remains best-selling EV globally; BYD closes gap in Europe.",
    "default":     "Markets mixed as geopolitical tensions weigh on risk sentiment; tech outperforms.",
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Fake SearXNG search — checks the query for known keywords and returns
    a matching headline. Falls back to a generic market headline if nothing fits.
    """
    q = query.lower()
    for kw, headline in MOCK_NEWS_DB.items():
        if kw in q:
            return headline
    return MOCK_NEWS_DB["default"]


# ── state ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    bot_id: str
    persona: str
    messages: Annotated[list, add_messages]
    search_query: str
    search_results: str
    final_output: dict   # {"bot_id": ..., "topic": ..., "post_content": ...}


# ── llm ───────────────────────────────────────────────────────────────────────

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY", ""),
        max_tokens=512,
    )


# ── node 1 ────────────────────────────────────────────────────────────────────

def node_decide_search(state: AgentState) -> AgentState:
    """LLM reads the persona and decides what to post about + what to search."""
    print(f"\n[Node 1 – decide_search] bot: {state['bot_id']}")
    llm = get_llm()

    system = f"""You are {state['bot_id']}.
Your persona: {state['persona']}

Decide what topic you want to post about today based on your personality.
Return a JSON object with exactly two keys:
  "topic"        – 3-6 word label
  "search_query" – 5-10 word search string to find relevant news

Raw JSON only. No markdown, no explanation."""

    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content="What do you want to post about today?"),
    ])
    raw = resp.content.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    parsed = json.loads(raw)

    print(f"  topic: {parsed['topic']}")
    print(f"  query: {parsed['search_query']}")

    state["search_query"] = parsed["search_query"]
    state["messages"].append(AIMessage(content=raw))
    state["final_output"] = {"bot_id": state["bot_id"], "topic": parsed["topic"], "post_content": ""}
    return state


# ── node 2 ────────────────────────────────────────────────────────────────────

def node_web_search(state: AgentState) -> AgentState:
    """Runs the mock search tool and stores results in state."""
    print(f"\n[Node 2 – web_search] query: \"{state['search_query']}\"")
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  got: {results}")
    state["search_results"] = results
    return state


# ── node 3 ────────────────────────────────────────────────────────────────────

def node_draft_post(state: AgentState) -> AgentState:
    """LLM writes the final post using persona + search results."""
    print(f"\n[Node 3 – draft_post]")
    llm = get_llm()

    system = f"""You are {state['bot_id']}.
Your persona: {state['persona']}

Context from search:
{state['search_results']}

Write ONE highly opinionated social-media post, strictly under 280 characters.
Output ONLY the post text. No JSON, no quotes, no explanation, nothing else."""

    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content="Write your post."),
    ])
    post_content = resp.content.strip()[:280]
    parsed = {
        "bot_id": state["bot_id"],
        "topic": state["final_output"]["topic"],
        "post_content": post_content,
    }
    print(f"  ({len(post_content)} chars): {post_content}")
    state["final_output"] = parsed
    return state


# ── graph ─────────────────────────────────────────────────────────────────────

def build_content_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("decide_search", node_decide_search)
    g.add_node("web_search",    node_web_search)
    g.add_node("draft_post",    node_draft_post)
    g.set_entry_point("decide_search")
    g.add_edge("decide_search", "web_search")
    g.add_edge("web_search",    "draft_post")
    g.add_edge("draft_post",    END)
    return g.compile()


def generate_post(bot_id: str, persona: str) -> dict:
    """Run the full pipeline for a bot and return the structured post dict."""
    app = build_content_graph()
    init: AgentState = {
        "bot_id": bot_id,
        "persona": persona,
        "messages": [],
        "search_query": "",
        "search_results": "",
        "final_output": {},
    }
    final = app.invoke(init)
    return final["final_output"]


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from phase1_router import BOT_PERSONAS

    for bot_id, data in BOT_PERSONAS.items():
        print("\n" + "=" * 70)
        result = generate_post(bot_id, data["description"])
        print(f"\n[output]")
        print(json.dumps(result, indent=2))