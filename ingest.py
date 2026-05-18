import os
import sys
from dotenv import load_dotenv

# --- CHROMA_DB COMPATIBILITY FIX ---
# We use a custom filename 'sentinel.env' to prevent ChromaDB from auto-scanning it.
load_dotenv("sentinel.env")

import logging
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz  # PyMuPDF

# --- CONFIG & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("SENTINEL_DATA_PATH", "Data")
DB_PATH = os.getenv("SENTINEL_DB_PATH", "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("SENTINEL_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("SENTINEL_CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("SENTINEL_CHUNK_OVERLAP", 200))

def load_pdf_with_pymupdf(file_path: str):
    """Loads a PDF using PyMuPDF for high-quality text extraction."""
    try:
        doc = fitz.open(file_path)
        documents = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num}
                ))
        # Add a final metadata document for the whole file
        documents.append(Document(
            page_content=f"This document '{os.path.basename(file_path)}' has {len(doc)} pages.",
            metadata={"source": file_path, "type": "metadata", "total_pages": len(doc)}
        ))
        return documents
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(file_path)} with PyMuPDF: {e}")
        return []

def process_documents(file_paths):
    """Loads and splits a list of documents in parallel."""
    pdf_files = [p for p in file_paths if p.lower().endswith(".pdf")]
    
    with Pool(processes=min(cpu_count(), len(pdf_files))) as pool:
        all_docs = []
        with tqdm(total=len(pdf_files), desc="Loading Documents") as pbar:
            for docs in pool.imap_unordered(load_pdf_with_pymupdf, pdf_files):
                all_docs.extend(docs)
                pbar.update()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(all_docs)
    return chunks

def ingest_data():
    logger.info(f"--- Scanning {DATA_PATH} for documents ---")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        logger.info(f"Created {DATA_PATH} folder. Please add your documents there.")
        return

    all_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH)]
    if not all_files:
        logger.warning(f"No files found in '{DATA_PATH}'. Nothing to ingest.")
        return

    logger.info(f"Found {len(all_files)} file(s) to process.")
    
    chunks = process_documents(all_files)
    if not chunks:
        logger.error("No content could be extracted from the documents.")
        return

    logger.info(f"\n--- Vectorizing {len(chunks)} document chunks ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Ensure the directory is clean before ingestion
    # Note: This may fail if the directory is locked by another process
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            logger.info(f"Removed existing database at {DB_PATH}")
        except Exception as e:
            logger.warning(f"Could not remove {DB_PATH}: {e}. Proceeding with update.")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    logger.info("--- Ingestion Complete. AI Memory has been successfully updated. ---")

if __name__ == "__main__":
    ingest_data()
