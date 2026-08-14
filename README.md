# Queryq
A document chat application that uses hybrid retrieval to answer questions based on uploaded files.

Live demo: https://queryq.vercel.app/
API docs: https://queryq-backend.onrender.com/docs

## What it does
Users upload PDF or text files, which the backend parses and splits into manageable text chunks. When a user asks a question, the system retrieves the most relevant chunks using both keyword and dense vector search, passing them to a language model to generate an answer. The application cites the specific source document, page number, and text passage used, and explicitly refuses to answer if the information is not found in the uploaded documents.

## Screenshots

<!-- SCREENSHOT: upload zone and document cards -->
![Upload zone and document cards](Screenshot%202026-05-18%20140344.png)

<!-- SCREENSHOT: chat response with source panel open -->
![Chat response with source panel open](Screenshot%202026-05-18%20140326.png)

<!-- SCREENSHOT: out-of-scope refusal -->
![Out-of-scope query refusal](Screenshot%202026-05-18%20141209.png)

## Features
- PDF, plain text, DOCX, and Image (PNG/JPG) file upload
- Cloud-based OCR via OCR.space for scanned documents and images
- Multi-document support
- Drag and drop file interface
- Local embeddings generation
- Hybrid BM25 and dense vector retrieval
- Reciprocal Rank Fusion (RRF) algorithm
- Source citation and highlighting
- Retrieval confidence scores
- Out-of-scope query detection
- Conversation history context
 
## Architecture

Browser
  |
  v
FastAPI
  |-- ingest.py (PDF/DOCX/Image parsing, OCR, and text chunking)
  |-- langchain_store.py (Chroma DB persistent vector store and BM25 initialization)
  |-- llm.py (LangChain orchestration, ChatGroq, and RRF via EnsembleRetriever)

## Tech stack
Backend | Python, FastAPI, LangChain
Frontend | React, Vite, Tailwind CSS
Parsing & OCR | PyMuPDF, OCR.space API, python-docx, Pillow
Vector DB & Search | Chroma DB, SentenceTransformers, Rank-BM25
LLM | Groq API (via LangChain)
Deployment | Render, Vercel

## Chunking strategy
Documents are parsed and split using sentence-boundary detection to avoid breaking mid-sentence. The strategy targets chunks of approximately 512 tokens with an overlap of 50 tokens between adjacent chunks to preserve context. Each chunk stores metadata including the source document name, page number, and original text.

## Retrieval strategy
Dense search uses a local embedding model to convert the user query into a vector and calculates cosine similarity against document chunks. This captures the semantic meaning of the question.

BM25 search tokenizes the query and compares it against an inverted index of the document chunks. This ensures exact keyword matches and specific domain terms are not lost.

Reciprocal Rank Fusion (RRF) combines the results from both methods. It takes the ranking positions from the dense and keyword searches, calculates a fused score, and returns the top combined results to provide to the language model.

Out-of-scope queries are handled in two layers. First, if the highest retrieval confidence score is below a predefined threshold, the system immediately flags the query as out of scope. Second, the language model is instructed via its system prompt to state it cannot find the answer if the provided chunks do not contain relevant information.

## Real document test
Test document: [AI-Introduction-Sample.txt](frontend/public/sample.txt)

Test case 1
- Question: Who coined the term Artificial Intelligence and when?
- Answer: The term "Artificial Intelligence" was coined in 1956 by John McCarthy.
- Source: AI-Introduction-Sample.txt, page N/A
- Confidence: 98.5%
- Passage: The term "Artificial Intelligence" was coined in 1956 by John McCarthy at the Dartmouth Conference, which is widely considered the birth of AI as a formal academic discipline.

Test case 2
- Question: In which paper were transformers introduced?
- Answer: Transformers were introduced in the 2017 paper "Attention Is All You Need".
- Source: AI-Introduction-Sample.txt, page N/A
- Confidence: 97.8%
- Passage: Transformers: Introduced in the 2017 paper "Attention Is All You Need," transformers revolutionized natural language processing and led to models like GPT and BERT.

Out of scope test
- Question: What is the capital of France?
- Answer: I could not find this in the uploaded documents.

## Local setup
Backend setup:
1. Navigate to the backend directory
2. Create and activate a Python virtual environment
3. Install dependencies using pip install -r requirements.txt
4. Create a .env file with your GROQ_API_KEY
5. Start the server with uvicorn main:app --reload
6. Access Swagger API documentation at http://127.0.0.1:8000/docs

Frontend setup:
1. Navigate to the frontend directory
2. Install dependencies using npm install
3. Start the development server using npm run dev

## API endpoints
POST /upload | Accepts multipart form data with multiple files, chunks them, and stores them in memory
POST /chat | Accepts a question and chat history, retrieves context, and returns a generated answer
GET /documents | Returns a list of uploaded documents and their respective chunk counts
DELETE /documents/{name} | Removes a specific document and all of its associated chunks from the store

## Project structure
backend/
  main.py
  llm.py
  ingest.py
  store.py
  retrieval.py
  requirements.txt
  .env.example
frontend/
  index.html
  vite.config.js
  package.json
  src/
    main.jsx
    App.jsx
    index.css

---
Built by Saif — Alfaleus Technologies internship screening
