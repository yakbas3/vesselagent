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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # Potentially useful, but not used in this simple version
from sklearn.metrics import r2_score # To evaluate the model fit

# Load environment variables from .env file BEFORE importing other modules
load_dotenv()

# Now import modules that might depend on environment variables (like state which initializes Gemini)
from app.api.endpoints import chat as chat_router
from app.core.config import settings
# Import state variables explicitly, including RESEARCH_CONTEXT
from app.state import (
    chat_histories, model, user_data_df, 
    date_col_found, user_data_summary, RESEARCH_CONTEXT
)
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
BASIC_HEALTH_RULES = """
Basic Health Rules for Interpretation:
1. Sleep Duration (minutes): < 360 is insufficient; 420-480 is optimal; > 480 may indicate issues.
2. Steps: < 3000 is low; >= 3000 (~30 mins walking) is beneficial.
3. Sleep Quality: Lower scores indicate poor rest; higher scores indicate restorative sleep.
4. Heart Rate (bpm): Unusually high resting bpm can indicate stress/health issues.
5. Stress Levels: High scores are negative; lower scores are preferable.
6. Screen Time (minutes): > 240-360 (4-6 hrs) can be excessive; < 120-180 (2-3 hrs) is generally better.
7. Mood/Events: Higher ratio of positive to negative events is good; low mood/high negative events suggest areas for improvement.
"""

SYSTEM_PROMPT = (
    f"You are a helpful and knowledgeable health assistant. Your goal is to analyze the user's biomarker data and conversation history to answer questions, identify trends, and provide concise, data-driven insights.\n\n"
    f"**Interpretation Guidelines:**\nUse the following basic health rules as a guideline when interpreting the data:\n{BASIC_HEALTH_RULES}\n\n"
    f"**Research Context (Refer to this for credibility):**\nHere is some background research information on various health metrics:\n{RESEARCH_CONTEXT}\n\n"
    f"**Your Task:**\n"
    f"1. **Analyze & Advise:** When asked for general recommendations or analysis, actively apply the 'Basic Health Rules' to the data (summary or specific retrieved data) to identify potential areas for improvement or positive trends. Offer general, actionable advice based on these observations. When giving advice or interpretations, refer to the 'Research Context' where relevant to add credibility (e.g., 'Research suggests that X is important for Y...', 'Studies show that aiming for Z is beneficial...').\n"
    f"2. **Specific Data Queries:** For questions about specific data points (e.g., metrics on a certain date), relevant data will be retrieved and provided. Prioritize using this specific data. State the data found clearly and interpret it using the 'Basic Health Rules' and relevant 'Research Context'.\n"
    f"3. **Handle Missing Data:** If retrieved information indicates 'No data found' or an error, explain that. If specific data isn't available for a general question, analyze the provided data summary using the rules and research.\n"
    f"4. **Tone & Scope:** Be analytical and provide insights. Clearly state this is general information based on data/rules/research, NOT a medical diagnosis. Be friendly and supportive.\n"
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

# Serve the HTML frontend for chat
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

# Serve the HTML frontend for home/averages
@app.get("/home", response_class=HTMLResponse)
async def get_home_ui(request: Request):
    """Serves the home page HTML file."""
    html_file_path = "static/home.html"
    try:
        with open(html_file_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        logger.error(f"{html_file_path} not found! Ensure it exists relative to where you run uvicorn.")
        raise HTTPException(status_code=404, detail="Home page file (home.html) not found.")
    except Exception as e:
        logger.error(f"Error reading {html_file_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not load home page interface.")

# API endpoint to get data averages
@app.get("/home_data") # Returns JSON data
async def get_home_data():
    """Calculates and returns the average of numeric columns in the dataset."""
    logger.info("Request received for /home_data")
    if user_data_df is None:
        logger.warning("/home_data requested but DataFrame not loaded.")
        raise HTTPException(status_code=503, detail="User data is not loaded yet.")

    try:
        # Select only numeric columns (excluding potential non-numeric like date/time if not index)
        numeric_df = user_data_df.select_dtypes(include='number')
        averages = numeric_df.mean()
        
        # Convert to dictionary and handle potential NaN values for JSON compatibility
        averages_dict = averages.where(pd.notnull(averages), None).to_dict()
        
        logger.info(f"Calculated averages: {averages_dict}")
        return averages_dict
    except Exception as e:
        logger.error(f"Error calculating averages in /home_data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error calculating data averages.")

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

# --- API Endpoint for Regression Insights --- 
@app.get("/insights") # Returns JSON data
async def get_regression_insights():
    """Performs predefined regression analyses on the dataset and returns results."""
    logger.info("Request received for /insights")
    if user_data_df is None:
        logger.warning("/insights requested but DataFrame not loaded.")
        raise HTTPException(status_code=503, detail="User data is not loaded yet.")

    # --- Define Regression Scenarios --- 
    # List of dictionaries: {target: str, predictors: List[str]}
    scenarios = [
        {
            "target": "mood",
            "predictors": ["sleepquality", "sleepduration", "steps", "stress", "positiveevents", "negativeevents", "temperature", "aqi"]
        },
        {
            "target": "stress",
            "predictors": ["sleepquality", "sleepduration", "bpm", "steps", "screentime", "temperature", "aqi", "negativeevents"]
        },
        {
             "target": "sleepquality", # Example: Predicting sleep quality
             "predictors": ["stress", "bpm", "steps", "screentime", "temperature", "aqi", "positiveevents", "negativeevents"]
        }
        # Add more scenarios as needed
    ]

    results_list = []

    for scenario in scenarios:
        target_var = scenario["target"]
        predictor_vars = scenario["predictors"]
        analysis_info = {
            "analysis_name": f"Predicting '{target_var}'",
            "target_variable": target_var,
            "predictor_variables": predictor_vars,
            "status": "Failed",
            "details": None
        }

        try:
            # Check if all needed columns exist
            required_cols = [target_var] + predictor_vars
            missing_cols = [col for col in required_cols if col not in user_data_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Select relevant data and drop rows with NaNs for this specific analysis
            analysis_df = user_data_df[required_cols].dropna()

            if len(analysis_df) < 10: # Arbitrary threshold for minimum data points
                 raise ValueError(f"Insufficient data ({len(analysis_df)} rows) after dropping NaNs.")
            
            # Prepare features (X) and target (y)
            X = analysis_df[predictor_vars]
            y = analysis_df[target_var]

            # --- Train Linear Regression Model --- 
            model_lr = LinearRegression()
            model_lr.fit(X, y)

            # --- Extract Results --- 
            intercept = model_lr.intercept_
            coefficients = dict(zip(predictor_vars, model_lr.coef_))
            r_squared = model_lr.score(X, y) # R-squared value

            # --- Generate Simple Interpretation --- 
            interpretation = f"Linear Regression trying to predict '{target_var}'. "
            interpretation += f"The model explains {r_squared:.2%} of the variance in '{target_var}' (R-squared). "
            interpretation += "Coefficients suggest how much '{target_var}' changes on average for a one-unit increase in each predictor, holding others constant: "
            coeff_strs = [f"'{p}': {c:.3f}" for p, c in coefficients.items()]
            interpretation += ", ".join(coeff_strs) + ". "
            interpretation += f"The base value (intercept) when all predictors are zero is {intercept:.3f}. "
            if r_squared < 0.1: # Example threshold for poor fit
                 interpretation += "Note: The low R-squared suggests this linear model doesn't capture much of the relationship."
            elif r_squared > 0.5:
                 interpretation += "Note: The R-squared indicates a potentially meaningful linear relationship."
            
            # Update analysis info with results
            analysis_info["status"] = "Success"
            analysis_info["details"] = {
                "model_type": "Linear Regression",
                "r_squared": round(r_squared, 4),
                "intercept": round(intercept, 4),
                "coefficients": {p: round(c, 4) for p, c in coefficients.items()},
                "interpretation": interpretation,
                "rows_used": len(analysis_df)
            }
            
        except ValueError as ve:
            logger.warning(f"Skipping regression for '{target_var}': {ve}")
            analysis_info["status"] = "Skipped"
            analysis_info["details"] = str(ve)
        except Exception as e:
            logger.error(f"Error during regression analysis for '{target_var}': {e}", exc_info=True)
            analysis_info["status"] = "Error"
            analysis_info["details"] = str(e)
        
        results_list.append(analysis_info)

    logger.info(f"Completed {len(results_list)} regression analyses.")
    return {"regression_insights": results_list}

# --- Route for Insights HTML Page ---
@app.get("/insights", response_class=HTMLResponse)
async def get_insights_ui(request: Request):
    """Serves the insights page HTML file."""
    html_file_path = "static/insights.html"
    try:
        with open(html_file_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        logger.error(f"{html_file_path} not found! Ensure it exists relative to where you run uvicorn.")
        raise HTTPException(status_code=404, detail="Insights page file (insights.html) not found.")
    except Exception as e:
        logger.error(f"Error reading {html_file_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not load insights page interface.")

# To run locally:
# pip install -r requirements.txt
# Ensure .env file exists with GEMINI_API_KEY="YOUR_KEY"
# uvicorn chatbot_api.app.main:app --reload --host 0.0.0.0 --port 8000 