import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
import weaviate
from rag.pipeline.embedding import local_embeddings
from rag.pipeline.retrieval import semanticRetrieval, hybridRetrieval, mmrRetrieval, loadFAISS, faissRetrievalWithThreshold

client = weaviate.connect_to_local()
collection_script = client.collections.use("Interstellar_Script")
vectorstore = loadFAISS()

lm = dspy.LM('ollama/qwen3:8b',
             api_base='http://localhost:11434',
             api_key=None,
             temperature=0.9)

class getAnswer(dspy.Signature):
    """provide a detailed answer about the movie plot given the context"""
    question = dspy.InputField(desc="question about the movie")
    context = dspy.InputField(desc="relevant script excerpts")
    answer = dspy.OutputField(desc="detailed answer based on the script")

gen_answer = dspy.Predict(getAnswer)

def script_agent(state):
    question = state["question"]

    # all retrieval methods
    docs = semanticRetrieval(collection_script, question, local_embeddings)
    hybrid_docs = hybridRetrieval(collection_script, question, local_embeddings, k=5)
    mmr_docs = mmrRetrieval(vectorstore, question)
    faiss_docs = faissRetrievalWithThreshold(vectorstore, question, threshold=0.80)

    faiss_context = " ".join([text for text, score in faiss_docs])
    context = "".join(docs[1]) + " " + "".join(hybrid_docs[1]) + " " + " ".join(mmr_docs) + " " + faiss_context

    with dspy.context(lm=lm):
        result = gen_answer(question=question, context=context)

    return {"script_answer": result.answer}