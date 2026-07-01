# YouTube Intelligence Engine

An end-to-end NLP system that turns large-scale YouTube discussion around a topic into
searchable, analyzable knowledge — and then lets you *talk to it*. The project scrapes and
cleans tens of thousands of YouTube comments, runs a full classical + neural NLP analysis
pipeline over them (sentiment, aspect-based sentiment, named-entity recognition, EDA), and
serves everything through a multi-agent **Retrieval-Augmented Generation (RAG)** system with a
themed Streamlit chat interface.

The reference deployment is built around the film **Interstellar (2014)** and the companion
book **The Science of Interstellar**, combining three knowledge sources:

- **Script / plot** — the screenplay and a Wikipedia summary of the film.
- **Science / textbook** — *The Science of Interstellar* (Kip Thorne).
- **Audience** — ~41k scraped, cleaned, sentiment- and aspect-annotated YouTube comments.

A routing agent reads each question and dispatches it to the source (or combination of sources)
best equipped to answer it.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Rebuilding the Data & Indexes](#rebuilding-the-data--indexes)
- [Performance Notes](#performance-notes)
- [Roadmap](#roadmap)

---

## Architecture

```
                        ┌─────────────────────────────┐
   User question  ──▶   │  Streamlit app (rag/app)     │
                        │  "ENDURANCE // Mission Query" │
                        └──────────────┬──────────────┘
                                       │  graph.invoke()
                        ┌──────────────▼──────────────┐
                        │   LangGraph state machine    │
                        │                              │
                        │   classifier (qwen3:8b)      │
                        │        │ route               │
                        │  ┌─────┼───────────┬───────┐ │
                        │  ▼     ▼           ▼       ▼ │
                        │ script textbook comments multi│
                        │  agent  agent    agent   agent│
                        │  └─────┴─────┬─────┴───────┘ │
                        │              ▼                │
                        │         final_agent           │
                        └──────────────┬──────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Retrieval layer              │
                        │  • Weaviate (script/textbook/ │
                        │    comments collections)      │
                        │  • FAISS (script chunks)      │
                        │  • Cross-encoder re-ranking    │
                        └──────────────────────────────┘
```

**Agents** (LangGraph nodes in `rag/agents/`):

| Node | Responsibility | Retrieval used |
|------|----------------|----------------|
| `classify_question` | Routes the query to a category (`script` / `textbook` / `comments` / `multi`) | — |
| `script_agent` | Plot, characters, dialogue, events | Weaviate semantic + hybrid + FAISS MMR + threshold |
| `textbook_agent` | Physics, black holes, wormholes, gravity | Multi-hop query expansion + hybrid |
| `comments_agent` | Fan opinions & reactions | Weaviate semantic + hybrid (OpenAI embeddings) |
| `multi_agent` | Cross-references all three, then synthesizes | Runs the three agents above |
| `final_agent` | Selects/returns the answer for the UI | — |

---

## Features

- **Multi-agent RAG** orchestrated with LangGraph — an LLM classifier routes each question to
  the right knowledge source, with a dedicated "multi" path that synthesizes across all sources.
- **Multiple retrieval strategies** — dense semantic search, hybrid (BM25 + vector) search, MMR
  for diversity, score-thresholded FAISS, cross-encoder re-ranking, plus **HyDE** and
  **multi-hop query expansion** for harder questions (`rag/pipeline/retrieval.py`).
- **Hybrid embedding strategy** — local Ollama embeddings (`embeddinggemma`) for the small,
  private script/textbook corpora; OpenAI `text-embedding-3-small` for the large comment corpus.
- **Local, private LLMs** — generation runs on Ollama (`qwen3:8b`), no cloud LLM required.
- **Full NLP analysis pipeline** over the comment corpus (in `notebooks/`):
  - Data collection via the YouTube Data API
  - Cleaning, contraction/acronym expansion, language detection & translation
  - Exploratory data analysis (word clouds, distributions)
  - Sentiment analysis (VADER + `cardiffnlp/twitter-roberta-base-sentiment-latest`)
  - Named-entity recognition (spaCy `en_core_web_lg`)
  - Aspect-Based Sentiment Analysis (PyABSA)
- **Themed Streamlit UI** — an "Interstellar mission control" interface with a warp/hyperspace
  transition, ambient audio, a live elapsed-time readout during retrieval, and a persistent
  "Transmission Archive" of past chats (rename / delete, saved to disk).
- **Experiment tracking** — optional MLflow logging of questions, routing decisions, and answers.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Orchestration | LangGraph, DSPy |
| LLM (generation) | Ollama — `qwen3:8b` |
| Embeddings | Ollama `embeddinggemma` (local) · OpenAI `text-embedding-3-small` (comments) |
| Vector store | Weaviate (Docker) + FAISS (local) |
| Re-ranking | `sentence-transformers` CrossEncoder (`ms-marco-MiniLM-L6-v2`) |
| Ingestion | `unstructured` (PDF), `wikipedia-api`, NLTK, LangChain text splitters |
| NLP analysis | spaCy, PyABSA, Transformers, VADER, tweetnlp, scikit-learn |
| Data collection | YouTube Data API (`google-api-python-client`) |
| Tracking | MLflow |

---

## Repository Structure

```
youtube-intelligence-engine/
├── rag/                          # The RAG application
│   ├── app/
│   │   ├── main.py               # Streamlit app (chat UI, warp overlay, archive)
│   │   ├── chat_history.json     # Persisted chat sessions (runtime, git-ignored)
│   │   └── music.mp3
│   ├── agents/                   # LangGraph nodes
│   │   ├── supervisor.py         # classifier + router
│   │   ├── script_agent.py
│   │   ├── textbook_agent.py
│   │   ├── comments_agent.py
│   │   ├── multi_agent.py
│   │   └── final_agent.py
│   ├── pipeline/
│   │   ├── generator.py          # buildGraph(): wires the LangGraph state machine
│   │   ├── embedding.py          # local (Ollama) + OpenAI embedding wrappers
│   │   └── retrieval.py          # semantic/hybrid/MMR/HyDE/multi-hop + re-ranking
│   ├── extraction/extract.py     # Wikipedia + PDF ingestion
│   ├── chunking/chunk.py         # sliding-window / propositional / structured chunking
│   ├── indexing/                 # build the Weaviate + FAISS indexes
│   │   ├── index_script.py
│   │   ├── index_textbook.py
│   │   └── index_comments.py
│   ├── vector_stores/faiss_store/  # persisted FAISS index (script chunks)
│   └── utils/general_functions.py  # path + JSON/CSV helpers
├── notebooks/                    # Data collection + NLP analysis (largely run on Colab)
│   ├── data_generation.ipynb     # YouTube Data API scraping
│   ├── data_cleaning.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── sentiment_analysis.ipynb
│   ├── named_entity_recognition.ipynb
│   └── absa_notebook.ipynb        # PyABSA (needs its own environment — see notes)
├── data/
│   ├── raw/                      # source PDFs (screenplay, textbook)
│   ├── processed/                # extracted + chunked JSON/CSV
│   ├── raw_data.csv              # scraped comments
│   ├── cleaned_comments.csv
│   ├── absa_results_full.csv     # comments + aspect sentiment (indexed into Weaviate)
│   ├── sentiment_results.csv
│   └── ner_results.csv
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Ingestion & chunking (`rag/extraction`, `rag/chunking`)
- `extract.py` pulls a Wikipedia summary (`wikipedia-api`) and parses the PDFs with
  `unstructured`'s `partition_pdf` (hi-res).
- `chunk.py` produces several chunk families with LangChain's `RecursiveCharacterTextSplitter`:
  sliding-window "basic" chunks, **propositional** chunks (topics extracted by an LLM), and
  contextualized structured/textbook chunks.

### 2. Indexing (`rag/indexing`)
- `index_script.py` / `index_textbook.py` embed chunks locally with `embeddinggemma` and load
  them into Weaviate (`Interstellar_Script`, `Interstellar_Textbook`); the script also builds a
  local FAISS store.
- `index_comments.py` embeds ~41k comments with OpenAI `text-embedding-3-small` and loads them
  into the `Interstellar_Comments` collection along with their ABSA sentiment/aspects.

### 3. Retrieval (`rag/pipeline/retrieval.py`)
Semantic (`near_vector`), hybrid (BM25 + vector), MMR, score-thresholded FAISS, cross-encoder
re-ranking, HyDE, and multi-hop query expansion — mixed per agent.

### 4. Generation & orchestration (`rag/pipeline/generator.py`, `rag/agents`)
DSPy signatures wrap `qwen3:8b` for each agent. `buildGraph()` assembles the LangGraph:
`classifier → (script | textbook | comments | multi) → final_agent → END`.

---

## Setup

### Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com/)** running locally
- **Docker** (for Weaviate)
- An **OpenAI API key** (used for comment embeddings)
- *(Optional, for re-scraping)* a **YouTube Data API v3 key**

### 1. Clone & create a virtual environment
```bash
git clone <repo-url>
cd youtube-intelligence-engine
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables
Create a `.env` file in the repo root (it is git-ignored):
```dotenv
OPENAI_API_KEY=sk-...
# Only needed if you re-run notebooks/data_generation.ipynb:
YOUTUBE_API_KEY=...
```

### 3. Pull the Ollama models
```bash
ollama pull qwen3:8b
ollama pull embeddinggemma
ollama pull llama3.2        # referenced by the pipeline
```

### 4. Start Weaviate (Docker)
```bash
docker run -d --name weaviate -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  cr.weaviate.io/semitechnologies/weaviate:1.25.0
```
The code connects via `weaviate.connect_to_local()` (defaults to `localhost:8080`).

### 5. Build the indexes
Run once, from the `rag/` directory (they populate the Weaviate collections the agents expect):
```bash
cd rag
python indexing/index_script.py
python indexing/index_textbook.py
python indexing/index_comments.py     # ~41k OpenAI embeddings — the slow step
```
> `index_comments.py` embeds comments one-by-one via the OpenAI API, so it can take a while.

---

## Running the App

From the repo root, with Ollama and Weaviate running:
```bash
streamlit run rag/app/main.py
```
Then open the local URL Streamlit prints, and ask a question (e.g. *"How accurate is the
tesseract scene?"*). Past chats are saved in `rag/app/chat_history.json` and appear in the
sidebar **Archive**, where you can rename or delete them.

### Optional: MLflow tracking
`rag/pipeline/generator.py` contains a commented CLI loop that logs each run to MLflow. Uncomment
it and run `mlflow ui` to inspect questions, routing decisions, and answers.

---

## Rebuilding the Data & Indexes

The processed data and analysis outputs are already committed under `data/`, so you don't need
to regenerate them to run the app. To rebuild from scratch:

1. **Collect comments** — `notebooks/data_generation.ipynb` (needs a YouTube Data API key).
2. **Clean** — `notebooks/data_cleaning.ipynb`.
3. **Analyze** — `sentiment_analysis.ipynb`, `named_entity_recognition.ipynb`,
   `absa_notebook.ipynb`, `exploratory_data_analysis.ipynb`.
4. **Ingest & chunk sources** — `rag/extraction/extract.py`, then `rag/chunking/chunk.py`.
5. **Re-index** — the three scripts in `rag/indexing/`.

> **PyABSA note:** the ABSA notebook pins `transformers<4.30`, which conflicts with the versions
> other components need. Run it in a **separate virtual environment** (as it was on Colab) rather
> than the main one.

---

## Performance Notes

`qwen3:8b` is a reasoning model, so each generation emits hidden "thinking" tokens. On a laptop
RTX 4060 (8 GB VRAM):

- **Single-domain questions** (script / textbook / comments): ~30–70 s.
- **Multi-source questions**: ~2–4 min (runs three agents + a synthesis step).

The main one-time cost is `index_comments.py` (serial OpenAI embedding of ~41k comments). The
app caches the compiled graph across reruns via `@st.cache_resource`.

---

## Roadmap

- Batch the OpenAI comment embeddings to cut indexing time dramatically.
- Optional `/no_think` fast path for the classifier to reduce latency.
- Expose retrieval-source citations in the UI.
- Generalize beyond the Interstellar reference corpus to arbitrary topics.

---

*Built for CSCI 370 — RAG pipeline demo.*
