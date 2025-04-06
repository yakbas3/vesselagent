from typing import Dict, List, Any
import os
import google.generativeai as genai
from app.core.config import settings
import pandas as pd
import io
import logging

# --- Global In-Memory Store ---
# Warning: This data is lost if the server restarts!
# Key: session_id, Value: List of chat messages ({role: str, content: str})
chat_histories: Dict[str, List[Dict[str, str]]] = {}
# -----------------------------

# --- Global Data Store ---
data_file_path = "../data.csv" # USE data.csv instead of data2.csv
user_data_df = None
date_col_found = None # Initialize globally
user_data_summary = "No user data loaded."

# --- Attempt to load and process data on startup ---
logger = logging.getLogger(__name__) # Use logger if available, otherwise print

try:
    logger.info(f"Attempting to load data from: {data_file_path}")
    user_data_df = pd.read_csv(data_file_path)
    logger.info(f"Successfully loaded DataFrame with shape: {user_data_df.shape}")

    # --- Data Preprocessing: Find and parse date column ---
    potential_date_cols = ['Date', 'date', 'Timestamp', 'timestamp', 'Time', 'time']
    for col in potential_date_cols:
        if col in user_data_df.columns:
            try:
                original_dtype = user_data_df[col].dtype
                # Attempt to convert to datetime, handle potential errors
                parsed_dates = pd.to_datetime(user_data_df[col], errors='coerce')
                # Check if parsing was successful for at least some rows
                if not parsed_dates.isnull().all(): 
                    user_data_df[col] = parsed_dates # Assign back if successful
                    date_col_found = col
                    logger.info(f"Parsed column '{col}' (original dtype: {original_dtype}) as datetime.")
                    break # Stop after finding and parsing the first valid date column
                else:
                    logger.warning(f"Column '{col}' could be parsed but resulted in all NaT values.")
            except Exception as date_e:
                logger.warning(f"Could not parse column '{col}' as datetime: {date_e}")
    
    if date_col_found:
         logger.info(f"Identified '{date_col_found}' as the primary date column.")
    else:
        logger.warning("No standard date column found or parsed successfully.")
    # -----------------------------------------------------

    # --- Create Data Summary --- 
    summary_buffer = io.StringIO()
    summary_buffer.write(f"User Data Overview (from {os.path.basename(data_file_path)}):\n")
    summary_buffer.write("Columns: " + ", ".join(user_data_df.columns) + "\n")
    summary_buffer.write(f"Total Rows: {len(user_data_df)}\n")
    if date_col_found:
        summary_buffer.write(f"Date Range: {user_data_df[date_col_found].min()} to {user_data_df[date_col_found].max()}\n")
    summary_buffer.write("\nFirst 3 rows:\n")
    user_data_df.head(3).to_string(buf=summary_buffer)
    summary_buffer.write("\n\nLast 3 rows:\n")
    user_data_df.tail(3).to_string(buf=summary_buffer)
    user_data_summary = summary_buffer.getvalue()
    logger.info(f"Successfully created data summary.")
    # --------------------------

except FileNotFoundError:
    logger.error(f"Fatal Error: Data file not found at {data_file_path}. Ensure the file exists relative to the app directory.")
    user_data_summary = "Error: User data file could not be loaded (Not Found)."
except Exception as e:
    logger.error(f"Fatal Error loading or processing data from {data_file_path}: {e}", exc_info=True)
    user_data_summary = f"Error: Could not process user data file. Error: {e}"
# ---------------------------------------------------

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