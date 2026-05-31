import os
from langchain_ollama.llms import OllamaLLM

OLLAMA_HOST = os.getenv("SENTINEL_OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("SENTINEL_MODEL_NAME", "llama3")

router_llm = OllamaLLM(base_url=OLLAMA_HOST, model=MODEL_NAME, temperature=0, streaming=False)

def classify_intent(prompt: str) -> str:
    system_prompt = """You are a strict query classifier. Evaluate the user's input and classify it into EXACTLY ONE of the following categories:
- "SQL": For structured data queries, database queries, employee details, or metrics.
- "RAG": For unstructured questions, document searches, policies, or general knowledge retrieved from documents.
- "GENERAL": For conversational greetings, fallback, or non-informational chatting.

Output ONLY the category name ("SQL", "RAG", or "GENERAL") and nothing else. Do not provide explanations."""

    full_prompt = f"{system_prompt}\n\nUser Input: {prompt}\nClassification:"
    response = router_llm.invoke(full_prompt).strip().upper()

    if "SQL" in response:
        return "SQL"
    elif "RAG" in response:
        return "RAG"
    else:
        return "GENERAL"

