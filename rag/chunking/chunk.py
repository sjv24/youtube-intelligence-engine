import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.general_functions import saveJson, saveList, readJson
import dspy

# 1. load the data
wiki_data = readJson("wiki.json")
pdf_data = readJson("pdf.json")

# 2. sliding window chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=25,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
sliding_chunks = text_splitter.split_text(wiki_data["summary"])

# 3. propositional chunking — Task 1: use LLM to extract topics instead of hardcoding
lm = dspy.LM('ollama/qwen3:8b',
             api_base='http://localhost:11434',
             api_key=None,
             temperature=0.3)

class extractTopics(dspy.Signature):
    """extract exactly 3 main topics from the text as a short comma separated list"""
    text = dspy.InputField(desc="a chunk of text")
    topics = dspy.OutputField(desc="3 main topics as comma separated keywords, nothing else")

get_topics = dspy.Predict(extractTopics)

def getTopicPrefix(chunk):
    try:
        with dspy.context(lm=lm):
            result = get_topics(text=chunk)
        return result.topics.strip()
    except Exception:
        return "Interstellar film"

print("Generating propositional chunks with LLM topics...")
propositional_Chunk = []
for chunk in sliding_chunks:
    topics = getTopicPrefix(chunk)
    propositional_Chunk.append(f"{topics}, {chunk}")

# 4. structured chunking — Task 2A: global chunk IDs
pdf_text = pdf_data["text"]
pdf_title = pdf_data["title"]
structured_chunks = []
stuctured_sents = []

global_chunk_id = 0
full_text = " ".join([t if isinstance(t, str) else t.get("text", "") for t in pdf_text])
paragraph_chunks = text_splitter.split_text(full_text)

for par_num, chunk in enumerate(paragraph_chunks):
    sent_detail = {
        "chunk_id": global_chunk_id,
        "par_num": par_num,
        "sent_id": 0,
        "title": pdf_title,
        "chunk": chunk,
        "contextual_chunk": pdf_title + ", " + chunk
    }
    stuctured_sents.append(sent_detail)
    structured_chunks.append(sent_detail)
    global_chunk_id += 1

# 5. textbook chunking
textbook_data = readJson("textbook.json")
textbook_text = " ".join([t.get("text", "") if isinstance(t, dict) else t for t in textbook_data["text"]])
textbook_chunks = text_splitter.split_text(textbook_text)

textbook_sents = []
for par_num, chunk in enumerate(textbook_chunks):
    sent_detail = {
        "chunk_id": global_chunk_id,
        "par_num": par_num,
        "sent_id": 0,
        "title": textbook_data["title"],
        "chunk": chunk,
        "contextual_chunk": textbook_data["title"] + ", " + chunk
    }
    textbook_sents.append(sent_detail)
    global_chunk_id += 1

# save
saveJson(structured_chunks, "structured_chunks.json")
saveJson(stuctured_sents, "structured_sents.json")
saveList(sliding_chunks, "basic_chunks.csv")
saveList(propositional_Chunk, "propositional_chunks.csv")
saveJson(textbook_sents, "textbook_sents.json")

print(f"Done! Created {len(textbook_sents)} textbook chunks")
print(f"Done! Created {len(stuctured_sents)} structured chunks")