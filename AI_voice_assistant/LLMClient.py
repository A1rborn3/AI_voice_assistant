
import os
import json
import logging
import APIManagerMain

# Import Groq from the library
try:
    from groq import Groq
except ImportError:
    Groq = None

# Import dotenv to load environment variables
try:
    from dotenv import load_dotenv
    # Load .env file from the current directory or parent
    # Explicitly looking in the current folder just in case
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
except ImportError:
    pass # dotenv might not be installed, we assume vars are set otherwise

# Set up logger
logger = logging.getLogger(__name__)

# NOTE: You must set the GROQ_API_KEY environment variable.
if Groq:
    try:
        # Initialize Groq client. It automatically looks for GROQ_API_KEY in env.
        # If not found in env, we try to see if it's in the .env file we just loaded
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key.startswith("gsk_paste"):
             print("\n\n WARNING: GROQ_API_KEY is not set or is still the placeholder in .env! \n\n")
        
        client = Groq(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        client = None
else:
    logger.error("Groq library not installed.")
    client = None

# Tool Definitions (OpenAI Format for Groq)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_data",
            "description": "Get the weather data for the current location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forecast": {
                        "type": "boolean",
                        "description": "True to get 3-day forecast, False for today only."
                    }
                },
                "required": ["forecast"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Set a reminder for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer"},
                    "month": {"type": "integer"},
                    "year": {"type": "integer"},
                    "hour": {"type": "integer"},
                    "minute": {"type": "integer"},
                    "context": {"type": "string"}
                },
                "required": ["day", "month", "year", "hour", "minute", "context"]
            }
        }
    }
]

def handle_tool_calls(tool_calls, messages, response_message, model):
    """
    Execute tool calls and send results back to Groq.
    """
    # Append the assistant's message with tool calls to history
    # Converting message to dict to be safe for appending
    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        
        # Parse arguments
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            function_args = {}

        logger.info(f"Calling tool: {function_name} with args: {function_args}")
        print(f"DEBUG: Executing tool {function_name}")
        
        tool_response = None
        
        try:
            if function_name == "get_weather_data":
                tool_response = APIManagerMain.get_weather_data(**function_args)
            elif function_name == "create_reminder":
                tool_response = APIManagerMain.create_reminder(**function_args)
            else:
                tool_response = "Error: Unknown function."
        except Exception as e:
            tool_response = f"Error executing tool: {e}"
        
        # Append the tool result to history
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": str(tool_response) # Groq expects string content
        })

    # Second turn: Get the final answer from Groq
    try:
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return second_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq Follow-up Error: {e}")
        return "I completed the action but failed to generate a final response."

def askLLM(prompt, model="llama-3.3-70b-versatile"):
    """
    Query Groq LLM.
    """
    if not client:
        return "Groq client not initialized. Is the library installed and GROQ_API_KEY set?"
    
    # Simple history management: just the prompt if string, or passed list
    messages = []
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024
        )

        response_message = completion.choices[0].message
        
        # Check for tool calls
        if response_message.tool_calls:
            return handle_tool_calls(response_message.tool_calls, messages, response_message, model)
        
        return response_message.content

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Groq API Error: {error_msg}")
        if "401" in error_msg:
             return "Authentication failed. Please check your GROQ_API_KEY."
        elif "429" in error_msg:
             return "I've hit the Groq rate limit. Please wait a moment."
        else:
             import traceback
             traceback.print_exc()
             return f"I encountered an error with Groq: {error_msg}"
