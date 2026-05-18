"""
llm.py — LLM integration logic using Groq.

This module provides functions to interact with Groq's API for answering 
questions based on retrieved document context.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"

def _build_messages(question: str, chunks: list[dict], history: list[dict] = None) -> list[dict]:
    """Build the messages list for Groq from context chunks and chat history."""
    context_parts = []
    for i, item in enumerate(chunks, 1):
        chunk = item["chunk"]
        doc_name = chunk.get("doc_name", "Unknown")
        page_num = chunk.get("page_num", "N/A")
        text = chunk.get("text", "")
        context_parts.append(f"[Chunk {i} | doc: {doc_name} | page: {page_num}]: {text}")

    context_str = "\n\n".join(context_parts)

    system_prompt = (
        "Answer ONLY using the provided context chunks. "
        "If the answer is not present, respond exactly: 'I could not find this in the uploaded documents.' "
        "Do not use outside knowledge."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"})
    return messages


def answer_question(question: str, chunks: list[dict], history: list[dict] = None) -> dict:
    """
    Generate an answer to a question using retrieved chunks and conversation history.

    Args:
        question: The user's question.
        chunks:   List of result dicts from retrieval.py, containing 'chunk' metadata and 'text'.
        history:  List of message dicts (e.g., [{"role": "user", "content": "..."}]) representing the chat history.

    Returns:
        A dictionary containing the generated answer and the model used.
        {
            "answer": str,
            "model_used": str
        }
    """
    messages = _build_messages(question, chunks, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Error communicating with LLM: {str(e)}"

    return {"answer": answer, "model_used": MODEL_NAME}


def stream_answer_question(question: str, chunks: list[dict], history: list[dict] = None):
    """
    Generator that yields raw text tokens from Groq's streaming API.

    Args:
        question: The user's question.
        chunks:   Retrieved chunks from retrieval.py.
        history:  Prior conversation history.

    Yields:
        str — individual text tokens as they arrive from the LLM.
    """
    messages = _build_messages(question, chunks, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )
        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as e:
        yield f"\n[Error communicating with LLM: {str(e)}]"
