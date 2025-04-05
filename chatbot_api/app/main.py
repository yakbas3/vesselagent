from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import uuid
import logging

# Load environment variables from .env file BEFORE importing other modules
load_dotenv()

# Now import modules that might depend on environment variables (like state which initializes Gemini)
from app.api.endpoints import chat as chat_router
from app.core.config import settings
from app.state import chat_histories, model
from app import schemas
from app.services.chat_service import call_gemini_api

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if model initialization was successful (already happens in state.py)
if not model:
    logger.warning("Gemini model not initialized. AI features may be limited or disabled.")

# --- FastAPI App Instance ---
app = FastAPI(title="Hackathon Health Chatbot API (Web UI)", version="0.3.0")

# --- Mount Static Files Directory ---
# Serve files from the 'static' directory under the '/static' URL path
# The directory path is relative to where uvicorn is run.
# Since we run from inside 'chatbot_api', the path is just 'static'.
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Updated Gemini API Call Logic (moved from service for simplicity here) ---
MAX_HISTORY_LENGTH = 20 # Defined here, was 20 turns

async def call_gemini_api_with_history(session_id: str, user_message: str) -> str:
    """Calls the Gemini API using conversation history stored in memory."""
    if not model:
        logger.error("call_gemini_api_with_history called but model is not initialized.")
        return "AI Service not configured or API key missing."

    # Get or initialize history, ensure it's in Gemini format
    session_history = chat_histories.setdefault(session_id, [])

    # Append the new user message to the history *before* potentially trimming
    # Gemini format: list of {"role": "user/model", "parts": ["content"]}
    current_user_message_formatted = {"role": "user", "parts": [user_message]}
    session_history.append(current_user_message_formatted)

    # Trim history if it gets too long (based on total messages, not turns)
    if len(session_history) > MAX_HISTORY_LENGTH:
        # Keep the last MAX_HISTORY_LENGTH messages
        session_history = session_history[-MAX_HISTORY_LENGTH:]
        logger.info(f"Trimmed history for session {session_id} to {len(session_history)} messages.")

    # Update the stored history (important after trimming and before API call)
    chat_histories[session_id] = session_history

    # Prepare history for the API call (exclude the latest user message for start_chat)
    history_for_api = session_history[:-1]

    logger.debug(f"Session {session_id} History sending to API (len {len(history_for_api)}): {history_for_api}")
    logger.debug(f"Session {session_id} Current message: {current_user_message_formatted}")

    try:
        # Start chat with the history *excluding* the latest user turn
        chat = model.start_chat(history=history_for_api)
        
        # Send the *latest* user message's content
        response = await chat.send_message_async(current_user_message_formatted['parts'])
        ai_response_content = response.text
        logger.info(f"Gemini Response received for session {session_id}")

        # Append AI response to history *after* successful call
        ai_response_formatted = {"role": "model", "parts": [ai_response_content]}
        session_history.append(ai_response_formatted)

        # Update the global store again with the AI response included
        # Trim again *after* adding AI response if needed (though trimming before user msg is common)
        if len(session_history) > MAX_HISTORY_LENGTH:
             session_history = session_history[-MAX_HISTORY_LENGTH:]
        chat_histories[session_id] = session_history
        logger.debug(f"Session {session_id} History updated with AI response (len {len(session_history)})")

        return ai_response_content

    except Exception as e:
        logger.error(f"Error calling Gemini API for session {session_id}: {e}")
        # Attempt to remove the user message we added optimistically if the API call failed
        if session_history and session_history[-1]["role"] == "user":
             logger.warning(f"Removing last user message for session {session_id} due to API error.")
             session_history.pop()
             chat_histories[session_id] = session_history
        # Return a user-friendly error message
        return f"Sorry, I encountered an error processing your request. Please try again. (Error: {str(e)[:100]}...)"

# --- API Endpoints ---

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def get_chat_ui(request: Request):
    """Serves the main chat interface HTML file."""
    # Correct path relative to running uvicorn from inside chatbot_api
    html_file_path = "static/index.html"
    try:
        with open(html_file_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        logger.error(f"{html_file_path} not found! Ensure it exists relative to where you run uvicorn.")
        raise HTTPException(status_code=404, detail="Chat interface file (index.html) not found.")
    except Exception as e:
        logger.error(f"Error reading {html_file_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not load chat interface.")

# Handle chat messages from the frontend
@app.post("/chat", response_model=schemas.chat.ChatResponse)
async def handle_chat_post(chat_input: schemas.chat.ChatInput):
    """
    Handles chat messages from the UI, maintains in-memory context,
    and interacts with the Gemini AI.
    """
    session_id = chat_input.session_id or str(uuid.uuid4())
    logger.info(f"Received message for session: {session_id}")

    # Use the refined API call logic directly
    ai_response_content = await call_gemini_api_with_history(session_id, chat_input.user_message)

    return schemas.chat.ChatResponse(
        ai_response=ai_response_content,
        session_id=session_id # Return the session_id used/created
    )

# --- Optional: Endpoint to clear history for testing ---
@app.post("/clear_history/{session_id}")
async def clear_session_history(session_id: str):
    if session_id in chat_histories:
        del chat_histories[session_id]
        logger.info(f"Cleared history for session: {session_id}")
        return {"message": f"History cleared for session {session_id}"}
    else:
        logger.warning(f"Attempted to clear non-existent session: {session_id}")
        raise HTTPException(status_code=404, detail="Session ID not found")

@app.get("/active_sessions", tags=["Debugging"])
async def get_active_sessions():
    """Returns a list of active session IDs and their history length (for debugging)."""
    session_info = {sid: len(hist) for sid, hist in chat_histories.items()}
    return {"active_sessions": session_info}

# The original chat router is no longer needed if we handle /chat here
# app.include_router(chat_router.router, prefix="/chat", tags=["Chat API"])

# Remove the old root endpoint as '/' now serves the HTML
# @app.get("/", tags=["Root"])
# async def read_root(): ... 

# To run locally:
# pip install -r requirements.txt
# Ensure .env file exists with GEMINI_API_KEY="YOUR_KEY"
# uvicorn chatbot_api.app.main:app --reload --host 0.0.0.0 --port 8000 