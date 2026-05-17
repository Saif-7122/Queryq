# Queryq — Document Intelligence Platform

Queryq is a high-performance, full-stack RAG (Retrieval-Augmented Generation) platform that allows users to chat with their PDF and TXT documents. It combines local semantic search with keyword-based retrieval to provide highly accurate, grounded answers using Llama 3 via Groq.

## 🏗️ Architecture

```text
                                    +-----------------------+
                                    |     React Frontend    |
                                    | (Vite + Tailwind CSS) |
                                    +-----------+-----------+
                                                |
                                      /upload   |   /chat
                                                v
+---------------------------------------------------------------------------------------+
|                                    FastAPI Backend                                    |
|                                                                                       |
|  +-------------------+      +-----------------------+      +-----------------------+  |
|  |   Ingest Pipe     |      |    Document Store     |      |   Retrieval Pipe      |  |
|  | (PyMuPDF / S-BERT)|      | (In-Memory + BM25)    |      | (Dense + BM25 + RRF)  |  |
|  +---------+---------+      +-----------+-----------+      +-----------+-----------+  |
|            |                            |                              ^              |
|            +----------------------------+------------------------------+              |
|                                         |                                             |
|                                         v                                             |
|                               +-----------------------+                               |
|                               |       Groq LLM        |                               |
|                               | (Llama 3.3 70B)       |                               |
|                               +-----------------------+                               |
+---------------------------------------------------------------------------------------+
```

## 🚀 Features

- **Hybrid Search**: Combines Dense (Semantic) and BM25 (Keyword) retrieval.
- **RRF Fusion**: Uses Reciprocal Rank Fusion to merge search results fairly.
- **Smart Chunking**: Sentence-boundary-aware splitting with sliding window overlap.
- **Document Manager**: Add or remove documents mid-session with instant index rebuilding.
- **Source Citation**: Bot answers include expandable source panels with text highlighting.
- **Confidence Scoring**: Visual badges (Green/Yellow/Red) showing retrieval similarity.
- **Out-of-Scope Guard**: Dual-layer protection (Confidence threshold + LLM System Prompt).
- **Premium UI**: Dark-mode, glassmorphic design with smooth micro-animations.

## 🛠️ Tech Stack

| Technology | Role | Reason |
| :--- | :--- | :--- |
| **FastAPI** | Backend Framework | High performance, async-first, and automatic Swagger documentation. |
| **React + Vite** | Frontend | Lightning-fast development and optimized production bundles. |
| **Tailwind CSS** | Styling | Rapid UI building with a cohesive, modern utility-first approach. |
| **Groq (Llama 3.3)** | LLM | Extreme inference speed (up to 500+ tokens/sec) and state-of-the-art reasoning. |
| **Sentence-Transformers** | Local Embeddings | Local embedding generation using `bge-small-en-v1.5` for cost/privacy. |
| **PyMuPDF** | PDF Parsing | Robust and fast text extraction from complex PDF structures. |
| **Rank-BM25** | Keyword Search | Standard industry implementation for classical text retrieval. |

## 📖 Deep Dive: Pipeline Logic

### Ingestion & Chunking
Queryq doesn't just cut text at arbitrary lengths.
- **Sentence-Aware**: It splits text into individual sentences and groups them.
- **Size & Overlap**: Each chunk is targeted at **512 tokens** with a **50-token overlap**.
- **Context Preservation**: The overlap ensures that sentences aren't orphaned from their context across chunk boundaries.

### Hybrid Retrieval (RRF)
We use a two-pronged search approach:
1. **Dense Search**: Measures semantic similarity (meaning) using cosine similarity.
2. **BM25 Search**: Measures keyword frequency (exact matches).
3. **Reciprocal Rank Fusion (RRF)**: Merges the two lists using the formula `1 / (60 + rank)`. This ensures that a document ranking well in *both* lists rises to the top, while protecting against outliers.

### Out-of-Scope Detection
To prevent hallucinations, Queryq employs two layers of defense:
- **Retrieval Threshold**: If the best-retrieved chunk has a cosine similarity below **0.30**, the system signals "out-of-scope" immediately.
- **LLM Guard**: The system prompt forces the LLM to strictly answer only from provided context. If the answer isn't there, it returns a specific "not found" string.

## 💻 Local Setup

### Backend
1. `cd backend`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Configure `.env`: Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
6. Run: `uvicorn main:app --reload`

### Frontend
1. `cd frontend`
2. Install: `npm install`
3. Run: `npm run dev`

---
*Created with by Saif*
