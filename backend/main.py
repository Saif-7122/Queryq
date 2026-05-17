"""
main.py — FastAPI application entry point.

Endpoints:
  POST   /upload                  — Accept files, run full ingestion pipeline, store in memory.
  POST   /chat                    — Accept question + history, run retrieval + LLM, return answer.
  GET    /documents               — Return the list of currently loaded document names.
  DELETE /documents/{doc_name}    — Remove a document and all its chunks from the store.
"""

import os
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env before any module that reads env vars (e.g. llm.py)
load_dotenv()

# Verify GROQ_API_KEY is present at startup so we fail fast and clearly.
if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Copy backend/.env.example to backend/.env "
        "and fill in your Groq API key."
    )

from ingest import chunk_text, embed_chunks, parse_file
from llm import answer_question
from retrieval import retrieve
from store import DocumentStore

# ---------------------------------------------------------------------------
# Singleton DocumentStore
# ---------------------------------------------------------------------------

document_store = DocumentStore()

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing to initialise beyond what's already at module level.
    yield
    # Shutdown: clean up the in-memory store.
    document_store.clear()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Queryq API",
    description="Hybrid RAG backend — upload documents and ask questions.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins during development; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []

class SourceItem(BaseModel):
    text: str
    doc_name: str
    page_num: int
    confidence: float

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    out_of_scope: bool

class UploadResponse(BaseModel):
    doc_name: str
    chunk_count: int

class DocumentMetadata(BaseModel):
    doc_name: str
    chunk_count: int

class DocumentsResponse(BaseModel):
    documents: list[DocumentMetadata]

class DeleteResponse(BaseModel):
    removed: str
    remaining_docs: list[str]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=list[UploadResponse], summary="Upload and ingest documents")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Accept one or more PDF or TXT files.

    For each file:
      1. Parse pages from raw bytes (PyMuPDF for PDF, decode for TXT).
      2. Chunk pages into sentence-boundary-aware token windows.
      3. Embed chunks locally via sentence-transformers.
      4. Add chunks to the in-memory DocumentStore.

    Returns a summary per file with the number of chunks stored.
    """
    results: list[UploadResponse] = []

    for file in files:
        filename = file.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower()

        if ext not in ("pdf", "txt"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}' for '{filename}'. Only PDF and TXT are accepted.",
            )

        file_bytes = await file.read()

        try:
            # Full ingestion pipeline
            pages = parse_file(file_bytes, filename)
            chunks = chunk_text(pages)
            chunks_with_embeddings = embed_chunks(chunks)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process '{filename}': {exc}",
            ) from exc

        document_store.add_document(chunks_with_embeddings)

        results.append(UploadResponse(doc_name=filename, chunk_count=len(chunks_with_embeddings)))

    return results


@app.post("/chat", response_model=ChatResponse, summary="Ask a question about uploaded documents")
async def chat(request: ChatRequest):
    """
    Answer a question grounded in the uploaded documents.

    Steps:
      1. Embed the query locally.
      2. Run hybrid retrieval (dense + BM25 + RRF).
      3. If the top confidence is below the threshold, return out_of_scope=True.
      4. Call the Groq LLM with the retrieved context and conversation history.

    Returns the answer, source citations, and an out-of-scope flag.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if document_store.get_embedding_matrix() is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload at least one document first.",
        )

    # 1. Embed the query using the same model as ingest.py
    from ingest import _get_embedding_model  # lazy import to avoid circular issues at module level

    model = _get_embedding_model()
    query_embedding: np.ndarray = model.encode(
        # BGE models expect the query prefix at retrieval time
        "Represent this sentence for searching relevant passages: " + request.question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # 2. Hybrid retrieval
    retrieved = retrieve(
        query_text=request.question,
        query_embedding=query_embedding,
        store=document_store,
        top_k=5,
    )

    # 3. Out-of-scope guard
    if not retrieved:
        return ChatResponse(
            answer="I could not find this in the uploaded documents.",
            sources=[],
            out_of_scope=True,
        )

    # 4. LLM answer
    history_dicts = [msg.model_dump() for msg in request.history]
    llm_result = answer_question(
        question=request.question,
        chunks=retrieved,
        history=history_dicts,
    )

    # 5. Build source citations
    sources = [
        SourceItem(
            text=item["chunk"]["text"],
            doc_name=item["chunk"]["doc_name"],
            page_num=item["chunk"]["page_num"],
            confidence=round(item["confidence"], 4),
        )
        for item in retrieved
    ]

    return ChatResponse(
        answer=llm_result["answer"],
        sources=sources,
        out_of_scope=False,
    )


@app.get("/documents", response_model=DocumentsResponse, summary="List uploaded documents")
async def list_documents():
    """
    Return the unique set of document names currently held in the store, 
    along with their chunk counts.
    """
    chunks = document_store.get_all_chunks()
    
    # Count chunks per document while preserving insertion order
    counts: dict[str, int] = {}
    for chunk in chunks:
        name = chunk.get("doc_name", "")
        if name:
            counts[name] = counts.get(name, 0) + 1
            
    doc_metadata = [
        DocumentMetadata(doc_name=name, chunk_count=count)
        for name, count in counts.items()
    ]

    return DocumentsResponse(documents=doc_metadata)


@app.delete("/documents/{doc_name}", response_model=DeleteResponse, summary="Delete a document from the store")
async def delete_document(doc_name: str):
    """
    Remove all chunks associated with `doc_name` from the in-memory store
    and rebuild the BM25 and embedding indices.

    Returns the doc_name and how many chunks were removed.
    Raises 404 if the document was not found in the store.
    """
    # Check the document exists first
    existing = {c.get("doc_name") for c in document_store.get_all_chunks()}
    if doc_name not in existing:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_name}' not found in the store.",
        )

    removed_count = document_store.remove_document(doc_name)
    
    # Get remaining docs after deletion
    remaining_docs = list({c.get("doc_name") for c in document_store.get_all_chunks() if c.get("doc_name")})

    return DeleteResponse(removed=doc_name, remaining_docs=remaining_docs)
