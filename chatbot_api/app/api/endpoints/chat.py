from fastapi import APIRouter, Depends, HTTPException, Body, Header
from typing import Optional, Annotated, Any
import uuid

# Explicitly import the Pydantic models needed
from app.schemas.chat import ChatInput, ChatResponse
# Removed: from app import schemas 
from app.services.chat_service import process_chat_message
from app.core.config import settings
from app.state import model

router = APIRouter()

# Use the directly imported ChatResponse model
@router.post("/", response_model=ChatResponse)
async def handle_chat_message(
    *,
    # Use the directly imported ChatInput model
    chat_input: ChatInput, # Request body
):
    """
    Handles incoming chat messages using in-memory state.
    Relies on session_id (and optionally health_data_snapshot) in the body.
    """
    if not model:
         # Check if the Gemini model was initialized successfully
         raise HTTPException(status_code=503, detail="AI service is not initialized. Check API key and logs.")

    # Generate session ID if not provided (start of a new chat)
    current_session_id = chat_input.session_id or str(uuid.uuid4())

    try:
        # Call the updated service function
        ai_response_content = await process_chat_message(
            session_id=current_session_id,
            user_message=chat_input.user_message,
            health_data_snapshot=chat_input.health_data_snapshot # Pass snapshot
        )

        # Use the directly imported ChatResponse model
        return ChatResponse(
            ai_response=ai_response_content,
            session_id=current_session_id # Return the session_id used/created
        )

    except ValueError as ve: # Handle specific known errors from service layer
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as http_exc: # Re-raise HTTP exceptions from service layer (like API errors)
        raise http_exc
    except Exception as e: # General error handling
        # TODO: Improve logging
        print(f"Error processing chat in session {current_session_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the chat message.") 