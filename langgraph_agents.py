# langgraph_agents.py
"""
LangGraph-style agent orchestration for phone/plan procurement.

- Calls mock API at http://localhost:8000
- Populates ChromaDB with devices / rateplans / addons / account
- Runs a Supervisor LLM (LMStudio at http://localhost:1234/v1) to ask clarifications or provide filters
- Specialized agents fetch and store to Chroma, recommender queries Chroma and returns JSON recommendations
- Falls back to function-based orchestration if `langgraph` package is not installed
"""

import os
import time
import json
import hashlib
import struct
import requests
from typing import List, Dict, Any
import chromadb
#from chromadb.config import Settings
from chromadb import EmbeddingFunction, Embeddings
from typing import cast

# LMStudio / LLM endpoint config
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
LM_MODEL = os.getenv("LM_MODEL", "qwen2.5-coder-7b-instruct")  # change to your model name if needed

# Mock API base (FastAPI script)
MOCK_API_BASE = os.getenv("MOCK_API_BASE", "http://localhost:8000")

# Chroma DB directory
CHROMA_DIR = "./chroma_langgraph_db"
#os.environ["CHROMA_DB_IMPL"] = "sqlite"

# Try import langgraph; fallback if missing
USE_LANGGRAPH = False
try:
    import langgraph as lg  # type: ignore
    # We'll still be defensive about exact API shape — not all langgraph versions are identical.
    USE_LANGGRAPH = True
    print("langgraph import: OK — will attempt LangGraph node-style execution.")
except Exception as e:
    print("langgraph not available or import failed. Falling back to function orchestration. To use LangGraph, pip install langgraph.")
    USE_LANGGRAPH = False


#########################
# ChromaDB Embedding Function Class
#########################
class DeterministicEmbeddingFunction(EmbeddingFunction[str]):
    """
    Custom embedding function that creates deterministic embeddings from text.
    Implements the ChromaDB EmbeddingFunction interface properly.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def __call__(self, input: list[str]) -> Embeddings:
        """
        Generate embeddings for a list of texts.

        Args:
            input: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        embeddings = []
        for text in input:
            embedding = self._deterministic_embedding(text, self.dim)
            embeddings.append(embedding)
        return cast(Embeddings, embeddings)

    def _deterministic_embedding(self, text: str, dim: int) -> List[float]:
        """Generate deterministic embedding from text hash."""
        import hashlib
        import struct

        h = hashlib.sha256(text.encode("utf-8")).digest()
        floats = []
        while len(floats) < dim:
            for i in range(0, len(h), 4):
                if len(floats) >= dim:
                    break
                chunk = h[i:i + 4]
                if len(chunk) < 4:
                    chunk = chunk.ljust(4, b'\0')
                intval = struct.unpack(">I", chunk)[0]
                val = (intval / 0xFFFFFFFF) * 2 - 1
                floats.append(val)
            h = hashlib.sha256(h).digest()
        return floats[:dim]


#########################
# ChromaDB helpers
#########################
def create_chroma_client(chroma_dir: str = CHROMA_DIR):
    """
    Create ChromaDB client with updated configuration.
    """
    try:
        # New way (ChromaDB >= 0.4.0)
        client = chromadb.PersistentClient(path=chroma_dir)
        return client
    except AttributeError:
        # Fallback for older versions
        try:
            from chromadb.config import Settings
            client = chromadb.Client(Settings(persist_directory=chroma_dir))
            return client
        except ImportError:
            # Very old versions
            client = chromadb.Client()
            return client


def reset_and_populate_chroma(client):
    # Delete existing collection if present
    try:
        client.delete_collection("purchase_data")
    except Exception:
        pass

    # Create collection with proper embedding function
    collection = client.create_collection(
        name="purchase_data",
        #embedding_function=DeterministicEmbeddingFunction(dim=128)
    )

    # endpoints we will fetch
    endpoints = [
        ("device", f"{MOCK_API_BASE}/devices"),
        ("rateplan", f"{MOCK_API_BASE}/rateplans"),
        ("addon", f"{MOCK_API_BASE}/addons"),
        ("account", f"{MOCK_API_BASE}/account"),
    ]

    print("Fetching mock API endpoints and inserting into ChromaDB...")
    for category, url in endpoints:
        try:
            r = requests.get(url, timeout=6)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = [data]
            for item in data:
                doc_id = f"{category}:{item.get('id', item.get('account_id', 'unknown'))}"
                doc_text = item.get("description", "")
                metadata = {"category": category}
                metadata.update({k: v for k, v in item.items() if k != "description"})
                collection.add(ids=[doc_id], documents=[doc_text], metadatas=[metadata])
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    # Persist is automatic with PersistentClient, but keeping for compatibility
    try:
        client.persist()
    except AttributeError:
        # persist() method doesn't exist in newer versions
        pass

    print("ChromaDB populated.")


def query_chroma(client, category: str, top_k: int = 4, query_text: str = ""):
    """
    Query ChromaDB collection with updated embedding approach.
    """
    coll = client.get_collection("purchase_data")

    # Use the collection's embedding function instead of manual embedding
    res = coll.query(
        query_texts=[query_text or category],
        n_results=top_k,
        where={"category": category}
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for i in range(len(ids)):
        out.append({
            "id": ids[i],
            "document": docs[i],
            "metadata": metas[i],
        })
    return out

#########################
# LMStudio chat wrapper
#########################
def lmstudio_chat(messages: List[Dict[str, str]], model: str = LM_MODEL, temperature: float = 0.0, max_tokens: int = 800):
    url = LMSTUDIO_URL.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "max_tokens": int(max_tokens)}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "choices": []}

def extract_text(resp: Dict[str, Any]) -> str:
    try:
        choice = resp["choices"][0]
        if "message" in choice:
            return choice["message"].get("content","")
        else:
            return choice.get("text","")
    except Exception:
        return ""

#########################
# Supervisor & Recommender prompts
#########################
SUPERVISOR_SYSTEM = """
You are the Supervisor agent for a phone/service procurement assistant.  Your expected outputs:

- If you need more info: ONLY return a JSON object exactly of the form:
  {"need_more_info": true, "clarifying_questions": ["q1", "q2", ...]}

- If you have what you need: ONLY return JSON:
  {"need_more_info": false, "query_categories": ["device","rateplan","addon","account"],
   "filters": {"budget": 1000, "priority":"photography", ...}
  }

DO NOT return any additional text outside the JSON.
"""

JSON_VALIDATOR = """
You are a JSON validator. You will find a valid JSON object in user content, and then return it as a response well-formed.
"""

RECOMMENDER_SYSTEM = """
You are the Recommendation Agent. You're given:
- user_request
- filters
- top results lists for devices, plans, addons, and account (with metadata)

Return ONLY a JSON object of the form:
{
  "recommendations": [
     {
       "device": {"id":"d1","name":"iPhone ...","price":999},
       "plan": {"id":"p1","name":"...","price":80},
       "addons": [{"id":"a2","name":"Device Protection","price":12}],
       "reason": "one-line reason",
       "first_month_total": 999 + 80 + 12,
       "monthly_total": 80 + 12
     }, ...
  ],
  "summary":"short 2-4 sentence summary of top recommendation and important notes"
}
"""

#########################
# Supervisor logic
#########################
def supervisor_call(user_request: str, clarifications: List[Dict[str,str]] = None):
    clar_text = ""
    if clarifications:
        clar_text = "\n\nPrevious clarifications:\n" + "\n".join([f"Q: {c['question']} A: {c['answer']}" for c in clarifications])

    messages = [
        {"role":"system","content": SUPERVISOR_SYSTEM},
        {"role":"user","content": f"User request:\n{user_request}\n\n{clar_text}"}
    ]
    resp = lmstudio_chat(messages, temperature=0.0, max_tokens=400)
    txt = extract_text(resp)
    try:
        return json.loads(txt)
    except Exception:
        # agent validation
        messages = [
            {"role": "system", "content": JSON_VALIDATOR},
            {"role": "user", "content": f"{txt}"}
        ]
        resp = lmstudio_chat(messages, temperature=0.0, max_tokens=400)
        txt = extract_text(resp)
        try:
            return json.loads(txt)
        except Exception:
            # Fallback heuristic
            print("Supervisor returned non-JSON; applying fallback. Raw:")
            print(txt[:1000])
            return {"need_more_info": True, "clarifying_questions": ["Please provide budget (USD) and primary use case (photography/streaming/travel/family)."]}

#########################
# Recommender logic
#########################
def recommender_call(user_request: str, filters: Dict[str,Any], chroma_client):
    # Query chroma for top items
    devices = query_chroma(chroma_client, "device", top_k=2, query_text=filters.get("priority",""))
    plans = query_chroma(chroma_client, "rateplan", top_k=2, query_text=filters.get("priority",""))
    addons = query_chroma(chroma_client, "addon", top_k=2, query_text=filters.get("priority",""))
    account_hits = query_chroma(chroma_client, "account", top_k=1, query_text="account") or []
    account_meta = account_hits[0]["metadata"] if account_hits else {"balance":0}

    context = {
        "user_request": user_request,
        "filters": filters,
        "account": account_meta,
        "devices": [{"id":d["metadata"].get("id"), "name":d["metadata"].get("name"), "price":d["metadata"].get("price"), "description": d["document"]} for d in devices],
        "plans": [{"id":p["metadata"].get("id"), "name":p["metadata"].get("name"), "price":p["metadata"].get("price"), "description": p["document"]} for p in plans],
        "addons": [{"id":a["metadata"].get("id"), "name":a["metadata"].get("name"), "price":a["metadata"].get("price"), "description": a["document"]} for a in addons]
    }

    messages = [
        {"role":"system","content": RECOMMENDER_SYSTEM},
        {"role":"user","content":"Context JSON:\n" + json.dumps(context, indent=2)}
    ]
    resp = lmstudio_chat(messages, temperature=0.0, max_tokens=1000)
    txt = extract_text(resp)
    try:
        parsed = json.loads(txt)
        # sanitize numeric fields
        for rec in parsed.get("recommendations", []):
            dev_price = rec.get("device",{}).get("price",0)
            plan_price = rec.get("plan",{}).get("price",0)
            addons_price = sum([a.get("price",0) for a in rec.get("addons",[])])
            if "first_month_total" not in rec:
                rec["first_month_total"] = dev_price + plan_price + addons_price
            if "monthly_total" not in rec:
                rec["monthly_total"] = plan_price + addons_price
        return parsed
    except Exception:
        print("Recommender returned non-JSON. Raw (first 1000 chars):")
        print(txt[:1000])
        # fallback: build greedy combos
        recs = []
        if context["devices"] and context["plans"]:
            for i, dev in enumerate(context["devices"][:3]):
                plan = context["plans"][0]
                candidate_addons = context["addons"][:1]
                first = dev["price"] + plan["price"] + sum(a["price"] for a in candidate_addons)
                monthly = plan["price"] + sum(a["price"] for a in candidate_addons)
                recs.append({
                    "device":dev, "plan":plan, "addons":candidate_addons,
                    "reason": f"Greedy combination suited to {filters.get('priority','general use')}",
                    "first_month_total": first,
                    "monthly_total": monthly
                })
        return {"recommendations": recs, "summary": "Fallback recommendations generated."}

#########################
# CLI orchestration (Supervisor asks clarifying questions)
#########################
def interactive_loop(chroma_client):
    clarifications: List[Dict[str,str]] = []
    print("=== LangGraph-style Procurement Assistant ===")
    print("Type your request (e.g. 'I want a phone for photography, budget $1000, travel internationally monthly') or 'quit' to exit.")
    while True:
        req = input("\nUSER REQUEST> ").strip()
        if req.lower() in ("quit","exit"):
            print("Bye.")
            break

        # Supervisor decision loop
        while True:
            decision = supervisor_call(req, clarifications)
            if decision.get("need_more_info"):
                questions = decision.get("clarifying_questions", [])
                if not questions:
                    questions = ["Please provide budget and primary use-case (photography/streaming/travel/family)."]
                for q in questions:
                    ans = input(f"Supervisor: {q}\nYOUR ANSWER> ").strip()
                    clarifications.append({"question": q, "answer": ans})
                # loop to re-evaluate
                continue
            else:
                filters = decision.get("filters", {})
                # If budget missing, try to parse from original request / clarifications
                if "budget" not in filters:
                    import re
                    m = re.search(r"\$?(\d{3,6})", req.replace(",",""))
                    if m:
                        filters["budget"] = int(m.group(1))
                # add clarifications to filters where reasonable
                for c in clarifications:
                    q=a_q=c["question"].lower()
                    a=c["answer"]
                    if "budget" in q:
                        try:
                            filters["budget"] = int(''.join(filter(str.isdigit, a)))
                        except: pass
                    if "travel" in q or "international" in q:
                        filters["travels_internationally"] = "yes" in a.lower() or "y" in a.lower()
                    if "priority" in q or "use" in q:
                        filters["priority"] = a
                # call recommender
                parsed = recommender_call(req, filters, chroma_client)
                print_recommendations(parsed)
                # reset clarifications
                clarifications = []
                break

def print_recommendations(parsed: Dict[str,Any]):
    recs = parsed.get("recommendations", [])
    if not recs:
        print("No recommendations could be generated.")
        print(parsed.get("summary",""))
        return
    # Print a table
    print("\nRecommendations:\n")
    print(f"{'Device':30} | {'Plan':20} | {'Add-ons':25} | {'1st mo total':12} | {'Monthly total':12}")
    print("-"*110)
    for r in recs:
        dev = r["device"]
        plan = r["plan"]
        addons = r.get("addons", [])
        dev_label = f"{dev.get('name')} (${dev.get('price')})"
        plan_label = f"{plan.get('name')} (${plan.get('price')})"
        addons_label = ", ".join([f"{a.get('name')}(${a.get('price')})" for a in addons]) or "None"
        first = r.get("first_month_total", 0)
        monthly = r.get("monthly_total", 0)
        print(f"{dev_label[:30]:30} | {plan_label[:20]:20} | {addons_label[:25]:25} | ${first:<11.2f} | ${monthly:<11.2f}")
    print("\nSummary:")
    print(parsed.get("summary",""))
    print("\n")

#########################
# Entrypoint
#########################
def main():
    # Create & populate ChromaDB
    client = create_chroma_client(CHROMA_DIR)
    reset_and_populate_chroma(client)

    # Check LMStudio reachability (informational)
    try:
        r = requests.get(LMSTUDIO_URL.rstrip("/") + "/models", timeout=3)
        print("LMStudio reachable at", LMSTUDIO_URL)
    except Exception as e:
        print("Warning: LMStudio not reachable at", LMSTUDIO_URL)
        print("LLM calls may fail. Error:", e)

    # Start interactive loop
    interactive_loop(client)

if __name__ == "__main__":
    main()
