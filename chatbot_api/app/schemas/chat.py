from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Use a common Base class if you have shared configurations
# from app.schemas import Base

class ChatInputUnauthenticated(BaseModel):
    user_message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    # client_id will come from header

class ChatResponseUnauthenticated(BaseModel):
    ai_response: str
    session_id: str 

class ChatInput(BaseModel):
    user_message: str = Field(..., min_length=1)
    session_id: Optional[str] = None # Client sends existing ID or None/empty for new chat
    # Optional: Client can send relevant health data snapshot per request for demo
    health_data_snapshot: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    ai_response: str
    session_id: str # Return the session ID used/created 