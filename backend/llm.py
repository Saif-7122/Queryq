"""
llm.py — LLM integration logic using LangChain and Groq.
"""

import os
from typing import List, Dict, Generator
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"

def _get_llm(streaming: bool = False):
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name=MODEL_NAME,
        temperature=0.0,
        max_tokens=1024,
        streaming=streaming
    )

def _convert_history(history: List[Dict]) -> List:
    """Convert history dicts to LangChain message objects."""
    messages = []
    if not history:
        return messages
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

def _build_chain(streaming: bool = False):
    llm = _get_llm(streaming=streaming)
    
    system_prompt = (
        "Answer ONLY using the provided context chunks.\n"
        "If the answer is not present, respond exactly: 'I could not find this in the uploaded documents.'\n"
        "Do not use outside knowledge.\n"
        "\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # This chain handles formatting `context` from a list of Documents
    return create_stuff_documents_chain(llm, prompt)


def answer_question(question: str, docs: List[Document], history: List[Dict] = None) -> dict:
    """
    Generate an answer using LangChain.
    """
    chain = _build_chain(streaming=False)
    chat_history = _convert_history(history)
    
    try:
        result = chain.invoke({
            "input": question,
            "context": docs,
            "chat_history": chat_history
        })
        answer = result
    except Exception as e:
        answer = f"Error communicating with LLM: {str(e)}"

    return {"answer": answer, "model_used": MODEL_NAME}


def stream_answer_question(question: str, docs: List[Document], history: List[Dict] = None) -> Generator[str, None, None]:
    """
    Generate a streaming answer using LangChain.
    """
    chain = _build_chain(streaming=True)
    chat_history = _convert_history(history)
    
    try:
        for chunk in chain.stream({
            "input": question,
            "context": docs,
            "chat_history": chat_history
        }):
            if chunk:
                yield chunk
    except Exception as e:
        yield f"\n[Error communicating with LLM: {str(e)}]"
