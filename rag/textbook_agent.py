import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
import weaviate
from rag.pipeline.embedding import local_embeddings
from rag.pipeline.retrieval import semanticRetrieval, hybridRetrieval, multiHop

client = weaviate.connect_to_local()
collection_textbook = client.collections.use("Interstellar_Textbook")

lm = dspy.LM('ollama/qwen3:8b',
             api_base='http://localhost:11434',
             api_key=None,
             temperature=0.9)

class getAnswer(dspy.Signature):
    """provide a detailed scientific answer given the context from the textbook"""
    question = dspy.InputField(desc="science question")
    context = dspy.InputField(desc="relevant textbook excerpts")
    answer = dspy.OutputField(desc="detailed scientific answer")

gen_answer = dspy.Predict(getAnswer)

def textbook_agent(state):
    question = state["question"]

    # multiHop for complex science questions
    docs = multiHop(question, collection_textbook, local_embeddings, lm, hops=2)
    hybrid_docs = hybridRetrieval(collection_textbook, question, local_embeddings, k=5)

    context = "".join(docs[1]) + " " + "".join(hybrid_docs[1])

    with dspy.context(lm=lm):
        result = gen_answer(question=question, context=context)

    return {"textbook_answer": result.answer}