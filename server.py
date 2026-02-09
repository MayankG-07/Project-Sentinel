import os
import re
import sqlite3
import logging
import socket
import uuid
import time
import json
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from dotenv import load_dotenv
from auth import get_current_user
import sqlparse
import fitz  # PyMuPDF
from fpdf import FPDF

# --- LOGGING & CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

AUDIT_LOG_FILE = "forensic_audit.log"

def log_forensic_audit(user_data: dict, query: str, sources: list, status: str = "SUCCESS"):
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": str(uuid.uuid4()),
        "user": user_data.get("username"),
        "roles": user_data.get("roles"),
        "query": query,
        "sources_accessed": [s.source for s in sources],
        "status": status
    }
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")

DB_PATH = os.getenv("DB_PATH", "chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OUTPUT_DIR = "/app/client/public/rendered_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are Project Sentinel, a professional, stateless AI agent.
You have access to two data sources:
1. STRUCTURED DATA: Results from a SQL database.
2. UNSTRUCTURED DATA: Text from PDF documents.

Your task is to combine information from BOTH sources to provide a single, concise, and factual answer.
If the data is in the database, use it. If the context is in the documents, use it.
If they conflict, prioritize the database for numbers and the documents for descriptions.
"""

# --- MODELS ---
class User(BaseModel):
    username: str
    roles: list[str]

class Source(BaseModel):
    source: str
    content: str

class QueryRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    sources: list[Source]

# --- INITIALIZE SERVICES ---
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- HELPERS ---
def scrub_sensitive_data(text: str):
    return anonymizer.anonymize(text=text, analyzer_results=analyzer.analyze(text=text, entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], language='en')).text

def clean_db_result(result):
    if not result or "No data" in str(result): return None
    res_str = str(result)
    cleaned = res_str.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("'", "")
    return re.sub(r',\s*$', '', cleaned).strip()

def run_secure_sql(query_text: str, db: SQLDatabase, llm: OllamaLLM):
    try:
        schema = db.get_table_info()
        # Improved prompt to ensure valid SQL
        prompt = f"You are a SQL expert. Write a standard SQL SELECT query to answer this question: {query_text}\n\nSchema:\n{schema}\n\nReturn ONLY the SQL code. No explanation."
        generated_sql = llm.invoke(prompt).strip().replace("```sql", "").replace("```", "")
        
        # Clean up the generated SQL
        generated_sql = generated_sql.split(';')[0] + ';'
        
        logger.info(f"AI Generated SQL: {generated_sql}")
        
        if "SELECT" not in generated_sql.upper():
            logger.warning("AI failed to generate a SELECT query.")
            return None
            
        result = db.run(generated_sql)
        logger.info(f"Database Result: {result}")
        return result
    except Exception as e:
        logger.error(f"SQL Execution Error: {e}")
        return None

# --- INITIALIZE ENGINES ---
llm = OllamaLLM(base_url=OLLAMA_HOST, model=MODEL_NAME, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
sql_db = SQLDatabase.from_uri(os.getenv("SQL_CONNECTION_STRING") or "sqlite:///db_data/company.db")

# --- ENDPOINTS ---
@app.get("/auth/verify", response_model=User)
async def verify_api_key(current_user: dict = Depends(get_current_user)):
    return User(username=current_user["username"], roles=current_user["roles"])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    safe_query = scrub_sensitive_data(request.text)
    sources = []
    
    try:
        # 1. GET DATA FROM SQL
        db_raw = run_secure_sql(safe_query, sql_db, llm)
        db_clean = clean_db_result(db_raw)
        if db_clean:
            sources.append(Source(source="Live Online Database (Supabase)", content=f"Data: {db_clean}"))

        # 2. GET DATA FROM RAG
        docs = retriever.invoke(safe_query)
        doc_context = "\n\n".join([doc.page_content for doc in docs])
        for doc in docs:
            sources.append(Source(source=os.path.basename(doc.metadata.get('source', 'Unknown')), content=doc.page_content))

        # 3. SYNTHESIZE FINAL ANSWER
        synthesis_prompt = f"""
        {SYSTEM_PROMPT}
        USER QUESTION: {safe_query}
        DATABASE DATA: {db_clean if db_clean else "No relevant data found in database."}
        DOCUMENT CONTEXT: {doc_context if doc_context else "No relevant text found in documents."}
        FINAL ANSWER:
        """
        response_text = llm.invoke(synthesis_prompt)

        log_forensic_audit(current_user, request.text, sources, "SUCCESS")
        return ChatResponse(response=response_text, sources=sources)

    except Exception as e:
        logger.error(f"Chat Error: {e}")
        log_forensic_audit(current_user, request.text, [], f"FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
