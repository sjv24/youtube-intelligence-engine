import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b", temperature=0)

# ---------- State ----------
class InterstellarState(TypedDict):
    question: str
    category: str
    script_answer: str
    textbook_answer: str
    comments_answer: str
    final_answer: str

# ---------- Classifier ----------
def classify_question(state: InterstellarState):
    prompt = f"""
    You are a question classifier for an Interstellar movie RAG system.
    Question: {state['question']}
    
    Classify into ONE category only:
    - script       (plot, characters, dialogue, events in the movie)
    - textbook     (science, physics, gravity, black holes, wormholes)
    - comments     (fan opinions, reactions, what people think)
    - multi        (needs both script and textbook)
    
    Return only the category name, nothing else.
    """
    result = llm.invoke(prompt)
    category = result.content.strip().lower()
    print(f"  [Supervisor] Routed to: {category}")
    return {"category": category}

# ---------- Router ----------
def route_question(state: InterstellarState):
    category = state["category"]
    if "script" in category:
        return "script_agent"
    elif "textbook" in category:
        return "textbook_agent"
    elif "comments" in category:
        return "comments_agent"
    else:
        return "multi_agent"