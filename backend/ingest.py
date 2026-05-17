"""
ingest.py — Document ingestion pipeline.

Steps:
  1. parse_file   : Extract raw text pages from PDF or TXT bytes.
  2. chunk_text   : Split pages into sentence-boundary-aware, overlapping token chunks.
  3. embed_chunks : Embed chunks locally using sentence-transformers (no API calls).
"""

import uuid
import re
from typing import Any

# ---------------------------------------------------------------------------
# 1. parse_file
# ---------------------------------------------------------------------------

def parse_file(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Parse a PDF or plain-text file from raw bytes.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to detect type and as doc_name).

    Returns:
        List of page dicts:
          {
            "text":     str,   # raw text of the page / full text for .txt
            "page_num": int,   # 1-based page number (always 1 for .txt)
            "doc_name": str,   # original filename
          }
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        return _parse_pdf(file_bytes, filename)
    elif ext == "txt":
        return _parse_txt(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: '.{ext}'. Only PDF and TXT are supported.")


def _parse_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract per-page text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. Install with: pip install PyMuPDF"
        ) from e

    pages = []
    # Open from bytes in-memory (no temp file needed)
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")  # plain text extraction
            text = text.strip()
            if text:  # skip blank pages
                pages.append({
                    "text": text,
                    "page_num": page_index,
                    "doc_name": filename,
                })
    return pages


def _parse_txt(file_bytes: bytes, filename: str) -> list[dict]:
    """Decode a plain-text file as a single 'page'."""
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    return [{
        "text": text,
        "page_num": 1,
        "doc_name": filename,
    }]


# ---------------------------------------------------------------------------
# 2. chunk_text
# ---------------------------------------------------------------------------

# Simple regex to split on sentence boundaries: period/!/? followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

# Rough approximation: 1 token ≈ 4 characters (works well for English prose).
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex boundary."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    pages: list[dict],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """
    Split page texts into token-sized chunks that respect sentence boundaries.

    Strategy:
      - Accumulate sentences into a chunk until adding the next sentence
        would exceed `chunk_size` tokens.
      - When a chunk is full, carry the last `overlap` tokens' worth of
        sentences into the next chunk to preserve context across boundaries.

    Args:
        pages:      Output of parse_file().
        chunk_size: Target max tokens per chunk (default 512).
        overlap:    Token overlap between consecutive chunks (default 50).

    Returns:
        List of chunk dicts:
          {
            "id":          str,   # UUID4 for deduplication / vector DB keying
            "doc_name":    str,
            "page_num":    int,
            "chunk_index": int,   # 0-based index within the document
            "text":        str,
          }
    """
    chunks: list[dict] = []
    global_chunk_index = 0

    for page in pages:
        sentences = _split_sentences(page["text"])
        if not sentences:
            continue

        # Slide a window of sentences to build chunks.
        window: list[str] = []
        window_tokens: int = 0

        def flush_window() -> None:
            nonlocal global_chunk_index
            chunk_text_str = " ".join(window)
            chunks.append({
                "id": str(uuid.uuid4()),
                "doc_name": page["doc_name"],
                "page_num": page["page_num"],
                "chunk_index": global_chunk_index,
                "text": chunk_text_str,
            })
            global_chunk_index += 1

        for sentence in sentences:
            s_tokens = _approx_tokens(sentence)

            # Edge case: a single sentence exceeds chunk_size — emit it alone.
            if not window and s_tokens >= chunk_size:
                window = [sentence]
                window_tokens = s_tokens
                flush_window()
                window, window_tokens = [], 0
                continue

            if window_tokens + s_tokens > chunk_size:
                # Emit current window as a chunk.
                flush_window()

                # Build the overlap tail: walk backward through window sentences
                # until we've collected ~`overlap` tokens.
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for sent in reversed(window):
                    t = _approx_tokens(sent)
                    if overlap_tokens + t > overlap:
                        break
                    overlap_sentences.insert(0, sent)
                    overlap_tokens += t

                window = overlap_sentences + [sentence]
                window_tokens = sum(_approx_tokens(s) for s in window)
            else:
                window.append(sentence)
                window_tokens += s_tokens

        # Flush any remaining sentences in the window.
        if window:
            flush_window()

    return chunks


# ---------------------------------------------------------------------------
# 3. embed_chunks
# ---------------------------------------------------------------------------

# Module-level cache so the model is loaded only once per process.
_embedding_model: Any = None
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _get_embedding_model():
    """Lazy-load and cache the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embedding. "
                "Install with: pip install sentence-transformers"
            ) from e
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add a local embedding vector to each chunk using BAAI/bge-small-en-v1.5.

    The model is loaded once and cached for the process lifetime — no API calls.

    BGE models benefit from a query prefix at retrieval time, but for document
    ingestion we embed the raw text (no prefix). The retrieval module should
    prefix queries with "Represent this sentence for searching relevant passages: ".

    Args:
        chunks: Output of chunk_text() — list of chunk dicts.

    Returns:
        The same list of chunk dicts, each now containing:
          "embedding": list[float]  — 384-dimensional vector for bge-small-en-v1.5
    """
    if not chunks:
        return chunks

    model = _get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    # encode() returns a numpy ndarray of shape (n_chunks, 384).
    # convert_to_numpy=True is the default; tolist() converts to plain Python lists
    # so the dicts stay JSON-serialisable.
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,  # recommended for cosine similarity with BGE
        convert_to_numpy=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks
