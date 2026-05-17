"""
llm.py — LLM integration logic using Groq.

This module provides functions to interact with Groq's API for answering 
questions based on retrieved document context.
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client
# Ensure GROQ_API_KEY is set in your environment or .env file
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"

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
    # 1. Format the retrieved chunks into a readable context block
    context_parts = []
    for i, item in enumerate(chunks, 1):
        chunk = item["chunk"]
        doc_name = chunk.get("doc_name", "Unknown")
        page_num = chunk.get("page_num", "N/A")
        text = chunk.get("text", "")
        
        context_parts.append(f"[Chunk {i} | doc: {doc_name} | page: {page_num}]: {text}")
    
    context_str = "\n\n".join(context_parts)

    # 2. Define the system prompt with strict instructions
    system_prompt = (
        "Answer ONLY using the provided context chunks. "
        "If the answer is not present, respond exactly: 'I could not find this in the uploaded documents.' "
        "Do not use outside knowledge."
    )

    # 3. Construct the message sequence for the chat completion
    messages = [{"role": "system", "content": system_prompt}]
    
    # Include conversation history if provided
    if history:
        messages.extend(history)

    # Add the current context and question
    user_query = f"Context:\n{context_str}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_query})

    # 4. Call the Groq API
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,  # Minimize creativity for factual extraction
            max_tokens=1024,
            top_p=1,
            stream=False,
        )

        answer = completion.choices[0].message.content
    except Exception as e:
        # Basic error handling for API calls
        answer = f"Error communicating with LLM: {str(e)}"

    return {
        "answer": answer,
        "model_used": MODEL_NAME
    }
