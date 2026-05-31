from pathlib import Path
import os
from dotenv import load_dotenv
import re
import logging
import uuid
import json
from datetime import datetime
from fastapi import FastAPI, Depends, Request, BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse # JSONResponse for non-streaming errors
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from sqlalchemy.orm import Session

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.core.auth import get_current_user
from backend.core.database import get_db, engine as db_engine
from backend.core.models import Base
from backend.core.security import sanitize_and_validate_sql
from backend.core.logger import log_audit_event # Import the new logger
from backend.api.schemas import QueryRequest, SentinelResponse, SourceCitation # Import API schemas
import fitz  # PyMuPDF

# --- DYNAMIC PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "backend" / "data"
CLIENT_DIR = BASE_DIR / "client"

# Ensure DATA_DIR exists before doing anything else
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file in the project root
load_dotenv(BASE_DIR / ".env")

# --- LOGGING & CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PATH & MODEL CONFIG ---
DB_PATH = os.getenv("SENTINEL_DB_PATH", (DATA_DIR / "chroma_db").as_posix())
MODEL_NAME = os.getenv("SENTINEL_MODEL_NAME", "llama3")
EMBEDDING_MODEL_NAME = os.getenv("SENTINEL_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("SENTINEL_OLLAMA_HOST", "http://localhost:11434")
OUTPUT_DIR = CLIENT_DIR / "public" / "rendered_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- INITIALIZE SERVICES & APP ---
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def on_startup():
    # Create database tables on startup
    Base.metadata.create_all(bind=db_engine)

# --- MODELS (Pydantic models are now in schemas.py) ---
class User(BaseModel): # This User model is for auth, not directly part of SentinelResponse
    username: str
    roles: list[str]
    clearance_level: str = "confidential" # Ensure clearance_level is part of the user model

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
        pix.save(OUTPUT_DIR / filename)
        return f"/rendered_pages/{filename}"
    except Exception as e:
        logger.error(f"Error rendering PDF page: {e}")
        return ""

def generate_and_validate_sql(query_text: str, db: SQLDatabase, llm: OllamaLLM) -> str:
    """
    Generates a SQL query from an LLM and immediately sanitizes/validates it.
    Raises ValueError if the SQL is invalid or not a SELECT statement.
    """
    schema = db.get_table_info()
    prompt = f"You are a SQL expert. Write a standard SQL SELECT query for: {query_text}\nSchema: {schema}\nReturn ONLY SQL."
    raw_response = llm.invoke(prompt).strip()
    
    sql_match = re.search(r'(SELECT.*)', raw_response, re.IGNORECASE | re.DOTALL)
    raw_llm_sql = sql_match.group(1).split(';')[0] + ';' if sql_match else ""
    
    if not raw_llm_sql:
        raise ValueError("The AI failed to generate any SQL.")
        
    sanitized_query = sanitize_and_validate_sql(raw_llm_sql)
    return sanitized_query

# --- RAG EXECUTION FUNCTION WITH METADATA FILTERING ---
def execute_rag(query: str, user_clearance: str, vector_db_instance: Chroma, k: int = 5):
    """
    Primary RAG execution function with metadata filtering for tenant isolation.
    Only retrieves documents matching the user's clearance level.
    """
    # Ensure the filter is applied BEFORE retrieval
    return vector_db_instance.similarity_search(query, k=k, filter={"clearance_level": user_clearance})


# --- INITIALIZE ENGINES ---
logic_llm = OllamaLLM(base_url=OLLAMA_HOST, model=MODEL_NAME, temperature=0, streaming=False)
stream_llm = OllamaLLM(base_url=OLLAMA_HOST, model=MODEL_NAME, temperature=0, streaming=True)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


# --- ENDPOINTS ---
@app.get("/auth/verify", response_model=User)
async def verify_api_key(current_user: dict = Depends(get_current_user)):
    # Ensure the User model returned here includes clearance_level
    return User(username=current_user["username"], roles=current_user["roles"], clearance_level=current_user.get("clearance_level", "confidential"))


@app.post("/v1/query", response_model=SentinelResponse) # response_model for documentation, actual return is StreamingResponse
@limiter.limit("5/minute")
async def query_endpoint(
    request: Request, # Required by slowapi
    request_data: QueryRequest, # Use the new QueryRequest schema
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db) # Inject database session
):
    session_id = request_data.session_id
    prompt = request_data.prompt
    safe_prompt = scrub_sensitive_data(prompt)
    user_clearance = current_user.get("clearance_level", "confidential") # Defaulting to 'confidential' if not set

    sanitized_sql = None
    sql_table_data = None
    final_message_content = ""
    sources_for_audit = [] # To store sources for audit log

    try:
        # --- PRE-STREAM PROCESSING: SQL Generation, Validation & Execution ---
        local_sql_db = SQLDatabase(engine=db_engine) # Use the global engine for Langchain's SQLDatabase
        
        try:
            sanitized_sql = generate_and_validate_sql(safe_prompt, local_sql_db, logic_llm)
            # Execute the sanitized query
            sql_table_data_raw = local_sql_db.run(sanitized_sql)
            sql_table_data = json.loads(sql_table_data_raw) if sql_table_data_raw else [] # Ensure it's a list of dicts
            logger.info(f"SQL executed successfully. Data: {sql_table_data}")
        except ValueError as e:
            # This catches SQL generation/validation errors
            error_msg = f"SQL Error: {e}"
            logger.warning(error_msg)
            log_audit_event(session_id=session_id, prompt=prompt, generated_sql=sanitized_sql, success=False, error_message=error_msg)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
        except Exception as e:
            # Catch any other SQL execution errors
            error_msg = f"SQL Execution Failed: {e}"
            logger.error(error_msg)
            log_audit_event(session_id=session_id, prompt=prompt, generated_sql=sanitized_sql, success=False, error_message=error_msg)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

        # --- PRE-STREAM PROCESSING: RAG Retrieval with Metadata Filtering ---
        docs = execute_rag(safe_prompt, user_clearance, vector_db)
        doc_context = "\n\n".join([doc.page_content for doc in docs])
        source_citations = [
            SourceCitation(document_name=os.path.basename(doc.metadata.get('source', 'Unknown')),
                           page_number=doc.metadata.get('page', -1))
            for doc in docs
        ]
        sources_for_audit = [s.dict() for s in source_citations] # Convert Pydantic models to dict for audit log

        # --- LLM Synthesis Prompt Construction ---
        synthesis_prompt = f"""
        You are Project Sentinel, a professional AI agent. You have DIRECT ACCESS to internal data.
        
        INTERNAL DATABASE DATA: {sql_table_data if sql_table_data else 'No records found.'}
        INTERNAL DOCUMENT CONTEXT: {doc_context if doc_context else 'No text found.'}
        
        USER QUESTION: {safe_prompt}
        
        INSTRUCTIONS:
        - If DATABASE DATA is provided, it is the ABSOLUTE TRUTH. Use it.
        - If the database is empty, do NOT guess or speculate from the documents.
        - Report facts directly and concisely.
        
        ANSWER:
        """

        # --- STREAMING GENERATOR ---
        async def generate_stream():
            nonlocal final_message_content # Allow modification of outer scope variable

            # 1. Stream initial metadata (sources, SQL data)
            initial_metadata = {
                "sql_query": sanitized_sql,
                "table_data": sql_table_data,
                "sources": [s.dict() for s in source_citations] # Convert Pydantic models to dict
            }
            yield f"event: metadata\ndata: {json.dumps(initial_metadata)}\n\n"

            # 2. Stream LLM message chunks
            yield "event: message_start\ndata: {}\n\n"
            for chunk in stream_llm.stream(synthesis_prompt):
                final_message_content += chunk
                yield f"event: message_chunk\ndata: {json.dumps({'chunk': chunk})}\n\n"
            yield "event: message_end\ndata: {}\n\n"

            # 3. Send a final 'complete' event if needed by client
            yield "event: complete\ndata: {}\n\n"

            # 4. Log audit event AFTER the full message has been generated and streamed
            log_audit_event(
                session_id=session_id,
                prompt=prompt,
                generated_sql=sanitized_sql,
                success=True,
                error_message=None # No error if we reached here
            )

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except HTTPException:
        # Re-raise HTTPExceptions (e.g., from SQL validation)
        raise
    except Exception as e:
        # Catch any unexpected errors during pre-stream processing
        error_msg = f"An unexpected server error occurred: {e}"
        logger.error(error_msg)
        log_audit_event(session_id=session_id, prompt=prompt, generated_sql=sanitized_sql, success=False, error_message=error_msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)