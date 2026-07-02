import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")
import weaviate
import dspy
from langgraph.graph import StateGraph, END
from pipeline.embedding import local_embeddings, openai_embeddings
from pipeline.retrieval import semanticRetrieval, mmrRetrieval, loadFAISS, faissRetrievalWithThreshold, hybridRetrieval
from agents.supervisor import InterstellarState, classify_question, route_question
from agents.script_agent import script_agent
from agents.textbook_agent import textbook_agent
from agents.comments_agent import comments_agent
from agents.multi_agent import multi_agent
from agents.final_agent import final_agent

# ---------- connect to weaviate ----------
client = weaviate.connect_to_local()

# ---------- load FAISS ----------
vectorstore = loadFAISS()

# ---------- LLM ----------
lm = dspy.LM('ollama/qwen3:8b',
             api_base='http://localhost:11434',
             api_key=None,
             temperature=0.9)

lm_2 = dspy.LM('ollama/llama3.2',
             api_base='http://localhost:11434',
             api_key=None,
             temperature=0.9)

# ---------- build graph ----------
def buildGraph():
    builder = StateGraph(InterstellarState)

    builder.add_node("classifier", classify_question)
    builder.add_node("script_agent", script_agent)
    builder.add_node("textbook_agent", textbook_agent)
    builder.add_node("comments_agent", comments_agent)
    builder.add_node("multi_agent", multi_agent)
    builder.add_node("final_agent", final_agent)

    builder.set_entry_point("classifier")
    builder.add_conditional_edges(
        "classifier",
        route_question,
        {
            "script_agent": "script_agent",
            "textbook_agent": "textbook_agent",
            "comments_agent": "comments_agent",
            "multi_agent": "multi_agent"
        }
    )

    for node in ["script_agent", "textbook_agent", "comments_agent", "multi_agent"]:
        builder.add_edge(node, "final_agent")

    builder.add_edge("final_agent", END)
    return builder.compile()

'''
def generateAnswerNoContext(question):
    class getAnswerNoContext(dspy.Signature):
        """provide a short answer about the question"""
        question = dspy.InputField(desc="question")
        answer = dspy.OutputField(desc="short answer")

    gen_no_context = dspy.Predict(getAnswerNoContext)
    with dspy.context(lm=lm):
        result = gen_no_context(question=question)
    return result.answer
'''

import mlflow

'''
if __name__ == "__main__":
    mlflow.set_experiment("interstellar-rag")
    mlflow.langchain.autolog()
    
    graph = buildGraph()

    print("Interstellar RAG System ready!")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        
        if question.lower() == "exit":
            print("Goodbye!")
            break
        
        if not question:
            continue

        with mlflow.start_run():
            mlflow.log_param("question", question)
            
            result = graph.invoke({"question": question})
            #no_context_answer = generateAnswerNoContext(question)
            
            print(f"\n[Routed to]: {result.get('category', 'unknown')}")
            #print(f"\n=== WITH CONTEXT (RAG) ===")
            print(result['final_answer'])
            #print(f"\n=== WITHOUT CONTEXT (LLM alone) ===")
            #print(no_context_answer)
            
            mlflow.log_param("category", result.get("category", "unknown"))
            mlflow.log_text(result["final_answer"], "rag_answer.txt")
            #mlflow.log_text(no_context_answer, "no_context_answer.txt")
        
        print()

    client.close()
'''