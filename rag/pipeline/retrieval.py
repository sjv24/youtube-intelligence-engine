import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weaviate
import dspy
from weaviate.classes import query
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from pipeline.embedding import local_embeddings, openai_embeddings

# ---------- FAISS ----------
def mmrRetrieval(vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 20,
            "lambda_mult": 0.3
        }
    )
    docs = retriever.invoke(question)
    text = []
    for doc in docs:
        text.append(doc.page_content)
    return text

def loadFAISS():
    faiss_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "vector_stores", "faiss_store")
    return FAISS.load_local(faiss_path, local_embeddings, allow_dangerous_deserialization=True)

# ---------- FAISS with Score Threshold ----------
def faissRetrievalWithThreshold(vectorstore, question, k=5, threshold=0.80):
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=k)
    results = []
    for doc, score in docs_and_scores:
        similarity = 1 / (1 + score)
        if similarity >= threshold:
            results.append((doc.page_content, similarity))
    return results

# ---------- Weaviate ----------
def semanticRetrieval(collection, text, embeddings, k=5, threshold=0.5):
    vector_ = embeddings.embed_query(text)
    try:
        response = collection.query.near_vector(
            near_vector=vector_,
            limit=k,
            certainty=threshold,
            return_metadata=query.MetadataQuery(certainty=True),
        )
        return [(obj.properties["text"],
                obj.metadata.certainty) for obj in response.objects], \
                [(obj.properties["text"]) for obj in response.objects]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return [], []
    
# ---------- Hybrid Retrieval ----------
def hybridRetrieval(collection, text, embeddings, k=5, alpha=0.5):
    try:
        vector_ = embeddings.embed_query(text)
        response = collection.query.hybrid(
            query=text,
            vector=vector_,
            alpha=alpha,
            limit=k,
            return_metadata=query.MetadataQuery(score=True)
        )
        return [(obj.properties["text"], obj.metadata.score) for obj in response.objects], \
               [(obj.properties["text"]) for obj in response.objects]
    except Exception as e:
        print(f"Hybrid retrieval error: {e}")
        return [], []
    
# ---------- Re-ranking ----------
def reRankCrossEncoder(question, chunks, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", top_k=5):
    reranker = CrossEncoder(model_name)
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]



# ---------- HyDE ----------
class getHyDE(dspy.Signature):
    """provide a short answer or brief explanation"""
    question = dspy.InputField(desc="question or a concept")
    answer = dspy.OutputField(desc="short answer to question or better explanation")

def HyDE(question, collection, embeddings, lm, n_documents=1):
    gen_answer = dspy.Predict(getHyDE)
    docs = []
    org_question = question
    with dspy.context(lm=lm):
        for i in range(n_documents):
            result = gen_answer(question=question)
            docs.append(result.answer)
            question = result.answer
    final_q = reRankCrossEncoder(org_question, docs, top_k=1)[0][0]
    final_docs = semanticRetrieval(collection, final_q, embeddings, k=3)
    return final_docs

# ---------- Multi-Hop ----------
class extractKeywords(dspy.Signature):
    """extract the most important keywords and concepts from the text relevant to the question"""
    question = dspy.InputField(desc="original question")
    context = dspy.InputField(desc="retrieved text to extract keywords from")
    keywords = dspy.OutputField(desc="comma separated list of key concepts and phrases")

class refineQuery(dspy.Signature):
    """rewrite and expand the question using the extracted keywords to improve document retrieval"""
    original_question = dspy.InputField(desc="original question")
    keywords = dspy.InputField(desc="key concepts extracted from previous retrieval")
    refined_question = dspy.OutputField(desc="improved and expanded question for better retrieval")

def multiHop(question, collection, embeddings, lm, hops=2, k=5, top_k=3):
    """
    Multi-Hop Query Expansion:
    each hop retrieves docs, extracts keywords, refines the query
    all retrieved docs are deduplicated and reranked at the end
    """
    extract_kw = dspy.Predict(extractKeywords)
    refine_q = dspy.Predict(refineQuery)

    all_docs = []       # accumulate all retrieved chunks across hops
    current_question = question

    for hop in range(hops):
        print(f"  [Multi-Hop] Hop {hop + 1}/{hops} — Query: {current_question[:80]}...")

        # step 1 — retrieve docs with current question
        results = semanticRetrieval(collection, current_question, embeddings, k=k)
        chunks_with_scores = results[0]   # [(text, certainty), ...]
        chunks_text = results[1]          # [text, ...]

        if not chunks_text:
            print(f"  [Multi-Hop] No results on hop {hop + 1}, stopping early.")
            break

        # accumulate
        all_docs.extend(chunks_with_scores)

        # step 2 — extract keywords from retrieved chunks
        context_blob = " ".join(chunks_text)
        with dspy.context(lm=lm):
            kw_result = extract_kw(question=question, context=context_blob)
            keywords = kw_result.keywords

        # step 3 — refine the query using keywords (skip on last hop)
        if hop < hops - 1:
            with dspy.context(lm=lm):
                rq_result = refine_q(original_question=question, keywords=keywords)
                current_question = rq_result.refined_question

    # deduplicate by text
    seen = set()
    unique_docs = []
    for text, score in all_docs:
        if text not in seen:
            seen.add(text)
            unique_docs.append(text)

    # rerank
    if len(unique_docs) == 0:
        return [], []

    reranked = reRankCrossEncoder(question, unique_docs, top_k=top_k)
    top_texts = [chunk for chunk, score in reranked]

    # return in same format as semanticRetrieval for compatibility
    return [(t, None) for t in top_texts], top_texts