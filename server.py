import os
import re
import sqlite3
import logging
import json
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from dotenv import load_dotenv
from auth import get_current_user

# --- LOGGING SETUP ---
# ... (logging setup remains the same)

# --- CONFIG ---
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "chroma_db")
SQL_DB_PATH = os.getenv("SQL_DB_PATH")
SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class User(BaseModel):
    username: str
    roles: list[str]

class QueryRequest(BaseModel):
    text: str

class Source(BaseModel):
    source: str
    content: str

class ChatResponse(BaseModel):
    response: str
    sources: list[Source]

# --- ENDPOINTS ---
@app.get("/health")
async def health_check():
    """Simple health check to confirm the server is running."""
    return {"status": "ok"}

@app.post("/auth/verify", response_model=User)
async def verify_api_key(current_user: dict = Depends(get_current_user)):
    """Verifies an API key and returns the user's details."""
    return User(username=current_user["username"], roles=current_user["roles"])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest, current_user: dict = Depends(get_current_user)):
    # ... (chat endpoint logic remains the same)
    raw_query = request.text
    user_roles = current_user.get("roles", [])
    username = current_user.get("username")
    
    safe_query = scrub_sensitive_data(raw_query)
    logger.info(f"Processing secure query for user '{username}' with roles {user_roles}")

    sql_triggers = ["count", "total", "average", "sum", "salary", "employees", "database", "list"]
    
    context = ""
    sources = []
    audit_event = {
        "user": username,
        "roles": user_roles,
        "query": safe_query,
        "data_source": None,
        "status": "Denied"
    }

    can_access_sql = "finance" in user_roles or "admin" in user_roles
    can_access_pdf = "legal" in user_roles or "admin" in user_roles
    is_sql_query = any(trigger in safe_query.lower() for trigger in sql_triggers)

    if is_sql_query:
        if not can_access_sql:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have permission to query the SQL database.")
        source_type = "SQL Database"
        audit_event.update({"data_source": source_type, "status": "Allowed"})
        context = run_secure_sql(safe_query)
        sources.append(Source(source=source_type, content=context))
    elif retriever:
        if not can_access_pdf:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have permission to query PDF documents.")
        source_type = "PDF Documents"
        audit_event.update({"data_source": source_type, "status": "Allowed"})
        docs = retriever.invoke(safe_query)
        context_list = []
        for doc in docs:
            source_file = doc.metadata.get('source', 'Unknown File')
            content = doc.page_content.replace("\n", " ")
            sources.append(Source(source=source_file, content=content))
            context_list.append(f"SOURCEFILE: {source_file} | CONTENT: {content}")
        context = "\n\n".join(context_list)
    else:
        logger.warning("No valid engine to route query to.")
        audit_event["status"] = "No Engine"

    audit_logger.info(json.dumps(audit_event))

    final_prompt = f"""
    You are Project Sentinel, a secure Data Analyst.
    ... (rest of the prompt)
    """
    
    response_text = llm.invoke(final_prompt)
    logger.info("Successfully generated response.")
    return ChatResponse(response=response_text, sources=sources)

# --- INITIALIZATION ---
# ... (engine and database initialization remains the same)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)