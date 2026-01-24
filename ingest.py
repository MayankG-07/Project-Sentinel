import os
import pypdf
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from dotenv import load_dotenv

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "Data")
DB_PATH = os.getenv("DB_PATH", "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


def load_and_split_pdf(pdf_path):
    """Loads a single PDF and splits it into chunks."""
    try:
        logger.debug(f"Processing PDF: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return []

def ingest_data():
    logger.info(f"--- Scanning {DATA_PATH} for PDFs ---")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        logger.info(f"Created {DATA_PATH} folder. Please add your PDF documents there.")
        return

    pdf_files = [os.path.join(DATA_PATH, file) for file in os.listdir(DATA_PATH) if file.endswith(".pdf")]

    if not pdf_files:
        logger.warning(f"No PDF files found in '{DATA_PATH}' folder. Nothing to ingest.")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to process.")

    with Pool(cpu_count()) as pool:
        all_chunks = []
        with tqdm(total=len(pdf_files), desc="Loading and Splitting PDFs") as pbar:
            for chunks in pool.imap_unordered(load_and_split_pdf, pdf_files):
                all_chunks.extend(chunks)
                pbar.update()

    if not all_chunks:
        logger.error("No content could be extracted from the PDFs. Please check the file formats and content.")
        return

    logger.info(f"\n--- Vectorizing {len(all_chunks)} document chunks (This may take a moment) ---")
    # Initialize HuggingFaceEmbeddings with num_workers for parallel embedding
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={'num_workers': cpu_count()}
    )

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    logger.info("--- Ingestion Complete. AI Memory has been successfully updated. ---")

if __name__ == "__main__":
    ingest_data()
