import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Local embedding model (for script + textbook)
local_embeddings = OllamaEmbeddings(model="embeddinggemma")

# OpenAI embedding model (for comments - faster for large datasets)
try:
    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception:
    openai_embeddings = None

def getLocalEmbedding(text):
    return local_embeddings.embed_query(text)

def getOpenAIEmbedding(text):
    if openai_embeddings is None:
        raise RuntimeError("OpenAI API key not set — can't use OpenAI embeddings")
    return openai_embeddings.embed_query(text)