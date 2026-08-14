"""
main.py — FastAPI application entry point.

Endpoints:
  POST   /upload                  — Accept files, run full ingestion pipeline, store in Chroma.
  POST   /chat                    — Accept question + history, run retrieval + LLM, return answer.
  GET    /documents               — Return the list of currently loaded document names.
  DELETE /documents/{doc_name}    — Remove a document and all its chunks from the store.
"""

import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.retrievers import EnsembleRetriever

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Copy backend/.env.example to backend/.env "
        "and fill in your Groq API key."
    )

from ingest import chunk_text, parse_file
from llm import answer_question, stream_answer_question
from langchain_store import LangchainStore

document_store = LangchainStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Chroma is persistent, so we don't clear it on shutdown

app = FastAPI(
    title="Queryq API",
    description="Hybrid RAG backend — upload documents and ask questions.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.get("/", summary="Root / Health Check")
async def root():
    return {"status": "ok", "message": "Queryq backend is running!"}


@app.post("/upload", response_model=list[UploadResponse], summary="Upload and ingest documents")
async def upload_files(files: list[UploadFile] = File(...)):
    results: list[UploadResponse] = []

    for file in files:
        filename = file.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower()

        if ext not in ("pdf", "txt", "docx", "png", "jpg", "jpeg"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}' for '{filename}'. Only PDF, TXT, DOCX, and images are accepted.",
            )

        file_bytes = await file.read()

        try:
            pages = parse_file(file_bytes, filename)
            chunks = chunk_text(pages)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process '{filename}': {exc}",
            ) from exc

        document_store.add_documents(chunks)

        results.append(UploadResponse(doc_name=filename, chunk_count=len(chunks)))

    return results

def _get_ensemble_retriever(k: int = 10):
    bm25 = document_store.get_bm25_retriever(k=k)
    vector = document_store.get_vector_retriever(k=k)
    
    if bm25 is None or vector is None:
        return None
        
    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.5, 0.5]
    )


@app.post("/chat", response_model=ChatResponse, summary="Ask a question about uploaded documents")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    ensemble_retriever = _get_ensemble_retriever(k=10)
    if ensemble_retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload at least one document first.",
        )

    retrieved_docs = ensemble_retriever.invoke(request.question)

    if not retrieved_docs:
        return ChatResponse(
            answer="I could not find this in the uploaded documents.",
            sources=[],
            out_of_scope=True,
        )

    history_dicts = [msg.model_dump() for msg in request.history]
    
    # Langchain Chain Call
    llm_result = answer_question(
        question=request.question,
        docs=retrieved_docs,
        history=history_dicts,
    )

    sources = [
        SourceItem(
            text=doc.page_content,
            doc_name=doc.metadata.get("doc_name", "unknown"),
            page_num=doc.metadata.get("page_num", 1),
            confidence=1.0, # Confidence is abstract with RRF in langchain
        )
        for doc in retrieved_docs
    ]

    return ChatResponse(
        answer=llm_result["answer"],
        sources=sources,
        out_of_scope=False,
    )


@app.post("/chat/stream", summary="Stream an answer about uploaded documents (SSE)")
async def chat_stream(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    ensemble_retriever = _get_ensemble_retriever(k=10)
    if ensemble_retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload at least one document first.",
        )

    retrieved_docs = ensemble_retriever.invoke(request.question)

    sources_payload = [
        {
            "text": doc.page_content,
            "doc_name": doc.metadata.get("doc_name", "unknown"),
            "page_num": doc.metadata.get("page_num", 1),
            "confidence": 1.0,
        }
        for doc in retrieved_docs
    ]

    history_dicts = [msg.model_dump() for msg in request.history]

    async def event_stream():
        if not retrieved_docs:
            yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'out_of_scope': True})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': 'I could not find this in the uploaded documents.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload, 'out_of_scope': False})}\n\n"

        for token in stream_answer_question(request.question, retrieved_docs, history_dicts):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/documents", response_model=DocumentsResponse, summary="List uploaded documents")
async def list_documents():
    metadatas = document_store.get_documents_metadata()
    
    counts: dict[str, int] = {}
    for meta in metadatas:
        name = meta.get("doc_name", "")
        if name:
            counts[name] = counts.get(name, 0) + 1
            
    doc_metadata = [
        DocumentMetadata(doc_name=name, chunk_count=count)
        for name, count in counts.items()
    ]

    return DocumentsResponse(documents=doc_metadata)


@app.delete("/documents/{doc_name}", response_model=DeleteResponse, summary="Delete a document from the store")
async def delete_document(doc_name: str):
    existing = {m.get("doc_name") for m in document_store.get_documents_metadata()}
    if doc_name not in existing:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_name}' not found in the store.",
        )

    removed_count = document_store.remove_document(doc_name)
    
    remaining_docs = list({m.get("doc_name") for m in document_store.get_documents_metadata() if m.get("doc_name")})

    return DeleteResponse(removed=doc_name, remaining_docs=remaining_docs)
