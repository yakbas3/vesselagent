from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import uuid
import logging
from typing import List, Tuple, Optional
import json
import pandas as pd

# Load environment variables from .env file BEFORE importing other modules
load_dotenv()

# Now import modules that might depend on environment variables (like state which initializes Gemini)
from app.api.endpoints import chat as chat_router
from app.core.config import settings
from app.state import chat_histories, model, user_data_df, date_col_found, user_data_summary
from app import schemas

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
# Ensure the 'static' directory exists at the same level as this main.py or adjust path
# IMPORTANT: Create the 'static' folder in the 'chatbot_api' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- System Prompt Definition ---
SYSTEM_PROMPT = (
    "You are a helpful and knowledgeable health assistant. "
    "Your goal is to analyze the user's biomarker data and conversation history to answer questions, identify trends, and provide concise, data-driven insights. "
    "For questions about specific data points (e.g., metrics on a certain date), relevant data will be retrieved and provided in the context prefixed with \'Specifically retrieved data...\'. "
    "You MUST prioritize and use this specifically retrieved data to answer the question it pertains to. State the data found clearly in your answer. "
    "If the retrieved information indicates \'No data found\' or an error, explain that to the user. "
    "If no specific data is retrieved for the query, you can use the general data summary (if provided previously) or state that you need more information or cannot answer from the summary alone. "
    "Focus on accuracy. Do not provide medical diagnoses. Be friendly."
)
# ------------------------------

# --- Helper Function for LLM-based Data Querying --- 
def query_user_data_with_llm(user_query: str, df_columns: List[str]) -> Tuple[Optional[str], bool]:
    """
    Uses the LLM to parse the user query and executes a predefined, safe query.
    Returns a tuple: (formatted result string or info message, is_actual_data_found)
    is_actual_data_found is True only if specific rows were successfully retrieved.
    """
    is_actual_data_found = False # Flag to track if we got real data vs info message
    if user_data_df is None or date_col_found is None:
        logger.info("query_user_data_with_llm: DataFrame or date column not available.")
        return ("Data source is not loaded or has no date column.", is_actual_data_found)
    
    if not model:
        logger.error("query_user_data_with_llm: Gemini model not available for parsing.")
        return ("Error: Could not parse query because the AI model is unavailable.", is_actual_data_found)

    parsing_prompt = f"""
    Analyze the user query about health data. Columns: {df_columns}. Date column: '{date_col_found}'.
    Extract: metric (column name), date (YYYY-MM-DD), query_type ('lookup', 'average', 'other').
    Query: "{user_query}"
    Return ONLY JSON: {{"metric": "...", "date": "...", "query_type": "..."}} (use null if not found).
    Standardize date found to YYYY-MM-DD.
    JSON response:
    """

    logger.info(f"Sending query to LLM for parsing: {user_query}")
    parsed_params = None
    try:
        parsing_response = model.generate_content(parsing_prompt)
        llm_output = parsing_response.text.strip()
        logger.info(f"LLM parsing response: {llm_output}")
        
        if llm_output.startswith("```json"):
            llm_output = llm_output.strip("`json\n \t")
        elif llm_output.startswith("```"):
             llm_output = llm_output.strip("`\n \t")
             
        parsed_params = json.loads(llm_output)
        parsed_params = {
            "metric": parsed_params.get("metric"),
            "date": parsed_params.get("date"),
            "query_type": parsed_params.get("query_type")
        }
        
    except json.JSONDecodeError:
        logger.warning(f"LLM response was not valid JSON: {llm_output}")
        return ("I had trouble understanding the structure of your request for data.", is_actual_data_found)
    except Exception as e:
        logger.error(f"Error during LLM query parsing: {e}")
        return ("An error occurred while trying to understand your data request.", is_actual_data_found)

    # --- Execute Predefined Query --- 
    query_type = parsed_params.get("query_type")
    metric = parsed_params.get("metric")
    date_str = parsed_params.get("date")

    if query_type == "lookup" and date_str: # Allow lookup even without specific metric (show whole row)
        logger.info(f"Executing lookup for metric '{metric or 'all'}' on date '{date_str}'")
        try:
            query_date = pd.to_datetime(date_str)
            mask = user_data_df[date_col_found].dt.date == query_date.date()
            results_df = user_data_df[mask]

            if not results_df.empty:
                # If a specific metric was requested and exists, select it
                if metric and metric in results_df.columns:
                    results_df = results_df[[date_col_found, metric]]
                elif metric: # Metric requested but not found
                     return (f"I found data for {query_date.strftime('%Y-%m-%d')}, but the specific metric '{metric}' wasn't in the columns: {list(user_data_df.columns)}", is_actual_data_found)

                result_string = f"Data found for {query_date.strftime('%Y-%m-%d')}:\n"
                result_string += results_df.to_string(index=False)
                is_actual_data_found = True # Mark that we found data rows
                return (result_string, is_actual_data_found)
            else:
                return (f"No specific data found for the date {query_date.strftime('%Y-%m-%d')} in the dataset.", is_actual_data_found)

        except ValueError:
            return (f"I couldn't understand the date '{date_str}' from your query.", is_actual_data_found)
        except Exception as e:
            logger.error(f"Error during data lookup on {date_str}: {e}")
            return ("An error occurred while trying to retrieve the specific data.", is_actual_data_found)
            
    elif query_type in ["average", "trend", "comparison"]:
        return (f"I understand you're asking about {query_type}, but that feature isn't implemented yet.", is_actual_data_found)
        
    else: # 'other', null, or unhandled type
         logger.info(f"LLM classified query as '{query_type or 'other/null'}', no specific data retrieval triggered.")
         return (None, is_actual_data_found) # No specific data/info to add

# --- Updated Gemini API Call Logic --- 
MAX_HISTORY_LENGTH = 20 # Max number of *messages* (user + model) to keep in history buffer

async def call_gemini_api_with_history(session_id: str, user_message: str) -> str:
    """Calls the Gemini API using conversation history and RAG context."""
    if not model:
        logger.error("call_gemini_api_with_history called but model is not initialized.")
        return "AI Service not configured or API key missing."

    # 1. Perform RAG query using LLM parser
    retrieved_info_str, specific_data_was_found = query_user_data_with_llm(
        user_message, 
        user_data_df.columns.tolist() if user_data_df is not None else []
    )

    # 2. Get current session history
    session_history = chat_histories.get(session_id, [])

    # 3. Construct the context for the API
    # Always include the base system prompt
    context_messages = [
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["Okay, I understand my role and will use the provided data context to answer questions about the user's health data."]}
    ]

    # Add specifically retrieved data/info *if* it was generated
    if retrieved_info_str:
        context_messages.extend([
            # Use a clear prefix indicating this is direct data/info for the query
            # Instruct the AI on HOW to use it and what NOT to output
            # Use a single triple-quoted f-string for multi-line content
            {"role": "user", "parts": [
                f"""INTERNAL CONTEXT ONLY (DO NOT mention this phrase or the raw data format to the user):
Based on the user's query, the following relevant data/information was retrieved:
```
{retrieved_info_str}
```
TASK: Formulate your response to the user based ONLY on this retrieved information and the conversation history.
IMPORTANT: DO NOT repeat the raw data format above in your response. State the key findings in a natural sentence."""
                ]},
            # Model's acknowledgment (can potentially be omitted if instructions are clear enough)
            {"role": "model", "parts": ["Okay, I will use the provided internal context to formulate a natural response to the user without mentioning the retrieval process or showing the raw data format."]}
        ])
    # Conditionally add general summary if NO specific data was retrieved this turn
    elif not specific_data_was_found: # Add summary only if no specific data found
        context_messages.extend([
            {"role": "user", "parts": [f"General Data Summary:\n```\n{user_data_summary}\n```"]},
            {"role": "model", "parts": ["Understood. I have the general data summary available if needed."]}
        ])
    
    # Combine context, history, and the new user message
    full_history_for_api = context_messages + session_history
    current_user_message_formatted = {"role": "user", "parts": [user_message]}
    full_history_for_api.append(current_user_message_formatted)

    # 4. Trim the combined history (if needed) 
    if len(full_history_for_api) > MAX_HISTORY_LENGTH:
        num_context_messages = len(context_messages) 
        num_history_to_keep = MAX_HISTORY_LENGTH - num_context_messages - 1 
        if num_history_to_keep < 0: num_history_to_keep = 0 
        start_index = len(session_history) - num_history_to_keep
        if start_index < 0: start_index = 0
        trimmed_session_history = session_history[start_index:]
        full_history_for_api = context_messages + trimmed_session_history + [current_user_message_formatted]
        logger.info(f"Trimmed full history for session {session_id} before API call.")

    # 5. Prepare history for the start_chat method 
    history_for_start_chat = full_history_for_api[:-1]
    
    logger.debug(f"Session {session_id} History for start_chat (len {len(history_for_start_chat)}): {history_for_start_chat}")
    logger.debug(f"Session {session_id} Current message parts: {current_user_message_formatted['parts']}")

    try:
        # 6. Call Gemini API 
        chat = model.start_chat(history=history_for_start_chat)
        response = await chat.send_message_async(current_user_message_formatted['parts'])
        ai_response_content = response.text
        logger.info(f"Gemini Response received for session {session_id}")

        # 7. Update *session* history 
        session_history.append(current_user_message_formatted)
        ai_response_formatted = {"role": "model", "parts": [ai_response_content]}
        session_history.append(ai_response_formatted)

        if len(session_history) > MAX_HISTORY_LENGTH:
            session_history = session_history[-MAX_HISTORY_LENGTH:]
            logger.info(f"Trimmed stored session history for {session_id} to {len(session_history)} messages.")

        chat_histories[session_id] = session_history 

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