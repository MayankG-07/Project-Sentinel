from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    """
    Defines the contract for an incoming query to the Sentinel API.
    """
    prompt: str = Field(..., min_length=1, description="The user's query or question.")
    session_id: str = Field(..., description="A unique identifier for the user's session.")

class SourceCitation(BaseModel):
    """
    Represents a single source document cited in the response.
    """
    document_name: str = Field(..., description="The name of the source document.")
    page_number: int = Field(..., description="The page number within the document that was cited.")

class SentinelResponse(BaseModel):
    """
    Defines the contract for a response from the Sentinel API.
    """
    message: str = Field(..., description="The primary text response from the AI.")
    sql_query: Optional[str] = Field(None, description="The SQL query that was executed, if any.")
    table_data: Optional[List[Dict[str, Any]]] = Field(None, description="The data returned from the SQL query, if any.")
    sources: Optional[List[SourceCitation]] = Field(None, description="A list of source documents cited in the response.")
