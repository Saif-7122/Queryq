"""
ingest.py — Document ingestion pipeline.

Steps:
  1. parse_file   : Extract raw text from PDF, TXT, DOCX, or Image bytes.
  2. chunk_text   : Split pages into sentence-boundary-aware, overlapping token chunks.
"""

import requests
import os
import io
import uuid
import re
from typing import Any
from PIL import Image



def parse_file(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Parse a PDF, TXT, DOCX, or Image file from raw bytes.
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        return _parse_pdf(file_bytes, filename)
    elif ext == "txt":
        return _parse_txt(file_bytes, filename)
    elif ext == "docx":
        return _parse_docx(file_bytes, filename)
    elif ext in ("png", "jpg", "jpeg"):
        return _parse_image(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: '.{ext}'.")


def _ocr_space_bytes(file_bytes: bytes, ext: str = "jpg") -> str:
    """Helper to call OCR.space API."""
    api_key = os.environ.get("OCR_SPACE_API_KEY")
    if not api_key:
        return ""
    payload = {'apikey': api_key, 'language': 'eng'}
    try:
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'file': (f'image.{ext}', file_bytes, f'image/{ext}')},
            data=payload,
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            if not result.get("IsErroredOnProcessing"):
                texts = [p.get("ParsedText", "") for p in result.get("ParsedResults", [])]
                return "\n".join(texts).strip()
    except Exception:
        pass
    return ""

def _parse_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract per-page text from a PDF using PyMuPDF (fitz) and OCR.space.
    
    - For pages with little/no text (scanned): run OCR on the full page render.
    - For pages with text but also embedded images: extract text normally AND
      run OCR on each embedded image to capture figure/diagram content.
    """
    try:
        import fitz
    except ImportError as e:
        raise ImportError("PyMuPDF is required for PDF parsing.") from e

    has_ocr_key = bool(os.environ.get("OCR_SPACE_API_KEY"))
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            ocr_parts = []

            if has_ocr_key:
                if len(text) < 50:
                    # Scanned page — render the whole page and OCR it
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("jpeg")
                        ocr_text = _ocr_space_bytes(img_bytes, "jpg")
                        if ocr_text:
                            ocr_parts.append(ocr_text)
                    except Exception:
                        pass
                else:
                    # Text page — also OCR any embedded images (figures, diagrams)
                    try:
                        image_list = page.get_images(full=True)
                        for img_info in image_list:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            img_bytes = base_image["image"]
                            img_ext = base_image.get("ext", "jpg")
                            ocr_text = _ocr_space_bytes(img_bytes, img_ext)
                            if ocr_text:
                                ocr_parts.append(ocr_text)
                    except Exception:
                        pass

            # Combine native text with any OCR'd image text
            combined = text
            if ocr_parts:
                combined = text + "\n" + "\n".join(ocr_parts) if text else "\n".join(ocr_parts)
            combined = combined.strip()

            if combined:
                pages.append({
                    "text": combined,
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

def _parse_image(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text from an image using OCR.space."""
    ext = filename.rsplit(".", 1)[-1].lower()
    text = _ocr_space_bytes(file_bytes, ext)
        
    if not text:
        return []
    return [{
        "text": text,
        "page_num": 1,
        "doc_name": filename,
    }]

def _parse_docx(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text from a DOCX file."""
    try:
        import docx
    except ImportError as e:
        raise ImportError("python-docx is required for DOCX parsing.") from e
        
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    if not text:
        return []
    return [{
        "text": text,
        "page_num": 1,
        "doc_name": filename,
    }]


_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex boundary."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    pages: list[dict],
    chunk_size: int = 256,
    overlap: int = 25,
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
            "id":          str,
            "doc_name":    str,
            "page_num":    int,
            "chunk_index": int,
            "text":        str,
          }
    """
    chunks: list[dict] = []
    global_chunk_index = 0

    for page in pages:
        sentences = _split_sentences(page["text"])
        if not sentences:
            continue

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

            if not window and s_tokens >= chunk_size:
                window = [sentence]
                window_tokens = s_tokens
                flush_window()
                window, window_tokens = [], 0
                continue

            if window_tokens + s_tokens > chunk_size:
                flush_window()

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

        if window:
            flush_window()

    return chunks



