import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
import weaviate
from pipeline.embedding import openai_embeddings
from pipeline.retrieval import semanticRetrieval, hybridRetrieval

client = weaviate.connect_to_local()
collection_comments = client.collections.use("Interstellar_Comments")

lm = dspy.LM('ollama/qwen3:8b',
             api_base='http://localhost:11434',
             api_key=None,
             temperature=0.9)

class getAnswer(dspy.Signature):
    """summarize what fans think about the topic based on the comments"""
    question = dspy.InputField(desc="question about fan opinions")
    context = dspy.InputField(desc="relevant fan comments")
    answer = dspy.OutputField(desc="summary of fan opinions")

gen_answer = dspy.Predict(getAnswer)

def comments_agent(state):
    question = state["question"]

    docs = semanticRetrieval(collection_comments, question, openai_embeddings)
    hybrid_docs = hybridRetrieval(collection_comments, question, openai_embeddings, k=5)

    context = "".join(docs[1]) + " " + "".join(hybrid_docs[1])

    with dspy.context(lm=lm):
        result = gen_answer(question=question, context=context)

    return {"comments_answer": result.answer}