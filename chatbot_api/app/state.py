from typing import Dict, List, Any
import os
import google.generativeai as genai
from app.core.config import settings

# --- Global In-Memory Store ---
# Warning: This data is lost if the server restarts!
# Key: session_id, Value: List of chat messages ({role: str, content: str})
chat_histories: Dict[str, List[Dict[str, str]]] = {}
# -----------------------------

# --- Configure Gemini --- 
# Use the API key from settings
genai.configure(api_key=settings.GEMINI_API_KEY)

try:
    # Initialize the model
    model = genai.GenerativeModel('gemini-1.5-flash') # Or choose another appropriate model like gemini-1.5-flash
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    print("Ensure GEMINI_API_KEY is set correctly in your .env file.")
    model = None # Set model to None if initialization fails

print(f"Gemini Model Initialized: {'Yes' if model else 'No'}") 