import os
import re
import sqlite3
import logging
import socket
from fastapi import FastAPI, Depends, HTTPException, status
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

# --- LOGGING & CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

# --- INITIALIZE SERVICES ---
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- MODELS ---
class User(BaseModel): username: str; roles: list[str]
class Source(BaseModel): source: str; content: str
class QueryRequest(BaseModel): text: str
class ChatResponse(BaseModel): response: str; sources: list[Source]

# --- HELPERS ---
def scrub_sensitive_data(text: str):
    return anonymizer.anonymize(text=text, analyzer_results=analyzer.analyze(text=text, entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], language='en')).text

def validate_sql(query: str):
    parsed = sqlparse.parse(query)
    for statement in parsed:
        if statement.get_type() != 'SELECT':
            raise ValueError("Only SELECT queries are allowed.")
    return query

def clean_db_result(result):
    if not result or result == "No data found.": return "No data found."
    res_str = str(result)
    cleaned = res_str.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("'", "")
    cleaned = re.sub(r',\s*$', '', cleaned).strip()
    return cleaned if cleaned else "No data found."

def run_secure_sql(query_text: str, db: SQLDatabase):
    try:
        schema = db.get_table_info()
        prompt = f"You are a SQL expert. Given the schema below, write a SQL query to answer the question. Return ONLY the SQL code.\n\nSchema:\n{schema}\n\nQuestion: {query_text}\n\nSQL:"
        generated_sql = llm.invoke(prompt).strip().replace("```sql", "").replace("```", "")
        safe_sql = validate_sql(generated_sql)
        logger.info(f"Executing SQL: {safe_sql}")
        return db.run(safe_sql)
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return f"Error: {str(e)}"

# --- INITIALIZE ENGINES ---
llm = OllamaLLM(base_url=OLLAMA_HOST, model=MODEL_NAME, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# --- DATABASE CONNECTION ---
sql_db = None
db_uri = os.getenv("SQL_CONNECTION_STRING")
db_source_name = "Local Offline Database"

if db_uri:
    try:
        if "sslmode" not in db_uri:
            db_uri += ("&" if "?" in db_uri else "?") + "sslmode=require"
        sql_db = SQLDatabase.from_uri(db_uri)
        db_source_name = "Live Online Database (Supabase)"
    except Exception as e:
        logger.error(f"Supabase Connection Failed: {e}")
        raise RuntimeError(f"Database Connection Failed: {e}")
else:
    sql_db = SQLDatabase.from_uri("sqlite:///db_data/company.db")

# --- IMPROVED AI ROUTING ---
def get_query_route(query: str) -> str:
    # 1. Broadened Keyword Check
    sql_keywords = ["employee", "salary", "paid", "department", "count", "how many", "list all", "everyone", "who is", "names of"]
    if any(word in query.lower() for word in sql_keywords):
        return "sql_database"
    
    # 2. LLM Double Check for ambiguous queries
    routing_prompt = f"""
    Analyze if this question is asking for specific data from a database (like names, numbers, lists) or general information from a document.
    Question: "{query}"
    Respond with ONLY 'sql_database' or 'document_search'.
    Tool:"""
    return llm.invoke(routing_prompt).strip().lower()

# --- ENDPOINTS ---
@app.get("/auth/verify", response_model=User)
async def verify_api_key(current_user: dict = Depends(get_current_user)):
    return User(username=current_user["username"], roles=current_user["roles"])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest, current_user: dict = Depends(get_current_user)):
    safe_query = scrub_sensitive_data(request.text)
    route = get_query_route(safe_query)
    logger.info(f"Route chosen: {route}")
    
    if "sql" in route:
        if any(role in current_user.get("roles", []) for role in ["finance", "admin"]):
            raw_result = run_secure_sql(safe_query, sql_db)
            clean_data = clean_db_result(raw_result)
            format_prompt = f"The database returned: {clean_data}\n\nAnswer the user's question based ONLY on this data: {safe_query}"
            response_text = llm.invoke(format_prompt)
            sources = [Source(source=db_source_name, content=str(raw_result))]
        else:
            raise HTTPException(status_code=403, detail="Unauthorized for SQL access.")
    else:
        docs = retriever.invoke(safe_query)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [Source(source=os.path.basename(doc.metadata.get('source', 'Unknown')), content=doc.page_content) for doc in docs]
        final_prompt = f"Answer using ONLY the CONTEXT provided.\n\nCONTEXT: {context}\n\nQUESTION: {safe_query}\n\nANSWER:"
        response_text = llm.invoke(final_prompt)

    return ChatResponse(response=response_text, sources=sources)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
