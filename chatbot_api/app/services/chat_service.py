from typing import Any, Tuple, Dict, List, Optional
from fastapi import HTTPException

# Import the shared state (in-memory history and Gemini model)
from app.state import chat_histories, model

# --- Configuration --- 
MAX_HISTORY_LEN = 20 # Max number of turns (user + assistant messages) to keep

# --- Helper Functions --- 
def get_demo_health_context(snapshot: Optional[Dict[str, Any]]) -> str:
    """Generates a context string from health data (static or snapshot)."""
    if snapshot:
        # Example: format data sent from the app
        parts = []
        hr = snapshot.get("heart_rate")
        if hr: parts.append(f"Heart Rate={hr} bpm")
        steps = snapshot.get("steps_today")
        if steps: parts.append(f"Steps={steps}")
        sleep = snapshot.get("sleep_last_night")
        if sleep: parts.append(f"Sleep={sleep}h")

        if not parts:
            return "No specific health data provided in this request."
        return f"User's latest data: {', '.join(parts)}."
    else:
        # Example: Static fallback demo data if snapshot is None
        return "User's demo data: Heart Rate=75 bpm, Steps=5200, Sleep=7h."

async def call_gemini_api(prompt_messages: List[Dict[str, str]]) -> str:
    """Calls the configured Gemini API asynchronously."""
    if not model:
        print("Error: Gemini model not initialized.")
        raise HTTPException(status_code=503, detail="AI Service not configured.")

    try:
        # Reformat for Gemini's content structure (list of Content objects)
        # We need role and parts (list of strings)
        gemini_formatted_history = []
        for msg in prompt_messages:
            # Map 'assistant' role to 'model' for Gemini API
            api_role = 'model' if msg['role'] == 'assistant' else msg['role']
            gemini_formatted_history.append({"role": api_role, "parts": [msg["content"]]})

        # The last message is the current user query
        # Separate it as Gemini `start_chat` expects history, not the current query
        if not gemini_formatted_history:
             raise ValueError("Cannot send an empty query to the AI.")

        current_query_content = gemini_formatted_history.pop()
        # History should not include the latest user query for start_chat
        chat_history_for_init = gemini_formatted_history

        # Start chat with history
        chat = model.start_chat(history=chat_history_for_init)
        
        # Send the current message (ensure it's in the correct format)
        response = await chat.send_message_async(current_query_content['parts'])
        return response.text

    except Exception as e:
        print(f"Error calling Gemini API: {e}") # Log the specific error
        # You might want to inspect 'e' for specific Gemini API errors
        raise HTTPException(status_code=500, detail=f"Error communicating with AI service: {e}")

async def process_chat_message(
    session_id: str,
    user_message: str,
    health_data_snapshot: Optional[Dict[str, Any]]
) -> str:
    """
    Processes chat message using in-memory state:
    1. Retrieves/Initializes conversation history for session_id.
    2. Gets health context (demo/snapshot).
    3. Formats the complete prompt.
    4. Calls the Gemini API.
    5. Updates the in-memory history.
    6. Returns the AI response content.
    """
    # 1. Get or initialize history for this session
    # Use setdefault for atomicity (though less critical without concurrency)
    session_history = chat_histories.setdefault(session_id, [])

    # 2. Get Health Context
    health_context_str = get_demo_health_context(health_data_snapshot)
    # Prepare it as a message to include in the prompt context
    # Note: Placing context optimally depends on the model. Sometimes near the user query is good.
    health_context_msg = {"role": "user", "content": f"[Use this Health Context: {health_context_str}]"}

    # 3. Prepare messages for AI
    system_prompt = {"role": "system", "content": "You are a helpful health assistant analyzing provided data and conversation history. Keep responses concise and focus on the provided information."}
    user_msg = {"role": "user", "content": user_message}

    # Construct the list of messages for the API call
    # Order: System Prompt (if supported/useful), Health Context, History, Current User Message
    messages_for_gemini = []
    if system_prompt["content"]: # Optional system prompt
         messages_for_gemini.append(system_prompt)
    if health_context_msg["content"]: # Optional health context
        messages_for_gemini.append(health_context_msg)
    
    messages_for_gemini.extend(session_history) # Add past conversation
    messages_for_gemini.append(user_msg) # Add current user message

    # Filter out system prompt if Gemini handles it via a different mechanism or if empty
    # The current google-generativeai library uses the history mechanism for roles 'user' and 'model'
    messages_for_gemini_filtered = [m for m in messages_for_gemini if m['role'] != 'system']

    # 4. Call AI
    ai_response_content = await call_gemini_api(messages_for_gemini_filtered)

    # 5. Update In-Memory History
    session_history.append(user_msg) # Append the actual user message
    session_history.append({"role": "assistant", "content": ai_response_content}) # Append AI response

    # --- Optional: Simple History Length Control --- 
    if len(session_history) > MAX_HISTORY_LEN * 2:
        # Keep only the last MAX_HISTORY_LEN turns (user + assistant messages)
        chat_histories[session_id] = session_history[-(MAX_HISTORY_LEN * 2):]
        print(f"Trimmed history for session {session_id}")
    # ---------------------------------------------

    # 6. Return AI response
    return ai_response_content 