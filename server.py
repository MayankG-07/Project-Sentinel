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
from fastapi.responses import StreamingResponse
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
    try:
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4()),
            "user": user_data.get("username"),
            "roles": user_data.get("roles"),
            "query": query,
            "sources_accessed": [s if isinstance(s, str) else s.get('source') for s in sources],
            "status": status
        }
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
    except: pass

DB_PATH = os.getenv("DB_PATH", "chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OUTPUT_DIR = "/app/client/public/rendered_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- INITIALIZE SERVICES ---
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- MODELS ---
class User(BaseModel):
    username: str
    roles: list[str]

class QueryRequest(BaseModel):
    text: str

# --- HELPERS ---
def scrub_sensitive_data(text: str):
    return anonymizer.anonymize(text=text, analyzer_results=analyzer.analyze(text=text, entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], language='en')).text

def clean_db_result(result):
    if not result: return None
    res_str = str(result).strip()
    if res_str in ["[]", "", "None", "()"]: return None
    cleaned = res_str.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("'", "")
    return re.sub(r',\s*$', '', cleaned).strip()

def render_pdf_page(doc_path: str, page_num: int) -> str:
    try:
        doc = fitz.open(doc_path)
        if page_num >= len(doc): return ""
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        filename = f"view_{uuid.uuid4().hex[:8]}.png"
        pix.save(os.path.join(OUTPUT_DIR, filename))
        return f"/rendered_pages/{filename}"
    except: return ""

def run_secure_sql(query_text: str, db: SQLDatabase, llm: OllamaLLM):
    try:
        schema = db.get_table_info()
        prompt = f"""
        You are a SQL expert. Write a standard SQL SELECT query to answer this question: {query_text}
        
        DATABASE SCHEMA:
        {schema}
        
        RULES:
        1. Return ONLY the SQL code. No explanation.
        2. Use JOINs to link tables.
        3. Start with 'SELECT'.
        """
        # Use the non-streaming LLM for logic
        raw_response = llm.invoke(prompt).strip()
        
        # Robust SQL Extraction
        sql_match = re.search(r'(SELECT.*)', raw_response, re.IGNORECASE | re.DOTALL)
        if not sql_match:
            logger.warning(f"SQL Generation Failed. Raw response: {raw_response}")
            return None
            
        generated_sql = sql_match.group(1).split(';')[0] + ';'
        logger.info(f"Executing SQL: {generated_sql}")
        
        result = db.run(generated_sql)
        return {"sql": generated_sql, "data": result}
    except Exception as e:
        logger.error(f"SQL Engine Error: {e}")
        return None

# --- INITIALIZE ENGINES ---
# Logic LLM (Non-Streaming) for SQL and reasoning
logic_llm = OllamaLLM(base_url=OLLAMA_HOST, model=MODEL_NAME, temperature=0, streaming=False)
# Stream LLM (Streaming) for the final response
stream_llm = OllamaLLM(base_url=OLLAMA_HOST, model=MODEL_NAME, temperature=0, streaming=True)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Database Connection with Logging
db_uri = os.getenv("SQL_CONNECTION_STRING")
if db_uri:
    logger.info("Initializing connection to LIVE SQL Database...")
    sql_db = SQLDatabase.from_uri(db_uri)
else:
    logger.warning("No SQL_CONNECTION_STRING found. Using local SQLite.")
    sql_db = SQLDatabase.from_uri("sqlite:///db_data/company.db")

# --- ENDPOINTS ---
@app.get("/auth/verify", response_model=User)
async def verify_api_key(current_user: dict = Depends(get_current_user)):
    return User(username=current_user["username"], roles=current_user["roles"])

@app.post("/chat")
async def chat(request: QueryRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    safe_query = scrub_sensitive_data(request.text)
    
    async def generate():
        sources = []
        
        # 1. VISUAL RENDERING
        page_match = re.search(r'page\s+(\d+)\s+of\s+([\w\d\s\._-]+?)(?:\.pdf)?\b', safe_query, re.IGNORECASE)
        if page_match:
            page_num, filename = int(page_match.group(1)), page_match.group(2).strip()
            doc_results = retriever.invoke(filename)
            if doc_results:
                img_url = render_pdf_page(doc_results[0].metadata['source'], page_num - 1)
                if img_url:
                    sources.append({"source": f"{filename} (Page {page_num})", "content": "Visual Render"})
                    yield json.dumps({"sources": sources}) + "\n---\n"
                    yield f"Here is page {page_num} of '{filename}':\n![Page Image]({img_url})"
                    log_forensic_audit(current_user, request.text, sources, "SUCCESS")
                    return

        # 2. DATA GATHERING (Using Logic LLM)
        sql_res = run_secure_sql(safe_query, sql_db, logic_llm)
        db_clean = clean_db_result(sql_res["data"]) if sql_res else None
        
        if db_clean:
            sources.append({"source": "Live Online Database (Supabase)", "content": f"SQL: {sql_res['sql']}\nResult: {db_clean}"})
        else:
            sources.append({"source": "SQL Engine", "content": "No relevant data found in database."})

        docs = retriever.invoke(safe_query)
        doc_context = "\n\n".join([doc.page_content for doc in docs])
        for doc in docs:
            sources.append({"source": os.path.basename(doc.metadata.get('source', 'Unknown')), "content": doc.page_content})

        # 3. AUTHORITATIVE SYNTHESIS (Using Stream LLM)
        synthesis_prompt = f"""
        You are Project Sentinel, a professional AI agent. You have DIRECT ACCESS to internal data.
        
        INTERNAL DATABASE DATA: {db_clean if db_clean else 'No records found.'}
        INTERNAL DOCUMENT CONTEXT: {doc_context if doc_context else 'No text found.'}
        
        USER QUESTION: {safe_query}
        
        INSTRUCTIONS:
        - If DATABASE DATA is provided, it is the ABSOLUTE TRUTH. Use it.
        - If the database is empty, do NOT guess or speculate from the documents.
        - Report facts directly and concisely.
        
        ANSWER:
        """
        
        yield json.dumps({"sources": sources}) + "\n---\n"
        for chunk in stream_llm.stream(synthesis_prompt):
            yield chunk

        log_forensic_audit(current_user, request.text, sources, "SUCCESS")

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
