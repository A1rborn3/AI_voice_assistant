
import os
import json
import logging
import APIManagerMain

# Import OpenAI from the library for local models
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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

# Flag to keep microphone open without wake word
EXPECTS_RESPONSE = False

# Initialize local OpenAI client
if OpenAI:
    try:
        # Initializing for a local instance. It does not require a real API key.
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="local-placeholder")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        client = None
else:
    logger.error("OpenAI library not installed.")
    client = None

# Tool Definitions (OpenAI Format)
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
    },
    {
        "type": "function",
        "function": {
            "name": "continue_conversation",
            "description": "Call this function to keep the microphone open without waiting for the wake word again. Use this when you ask the user a question or expect them to respond or ask a follow up question.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def handle_tool_calls(tool_calls, messages, response_message, model):
    """
    Execute tool calls and send results back to the LLM.
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
            elif function_name == "continue_conversation":
                global EXPECTS_RESPONSE
                EXPECTS_RESPONSE = True
                tool_response = "Success. The microphone will remain open for the user to reply immediately."
            else:
                tool_response = "Error: Unknown function."
        except Exception as e:
            tool_response = f"Error executing tool: {e}"
        
        # Append the tool result to history
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": str(tool_response)
        })

    # Second turn: Get the final answer from the LLM
    try:
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return second_response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Follow-up Error: {e}")
        return "I completed the action but failed to generate a final response."

#for easy copy paste
#alt model 1:qwen2.5-1.5b-instruct-q4_k_m
#alt model 2:Qwen3.5 0.8B

def askLLM(prompt, model="Qwen3.5 0.8B"):
    """
    Query Local LLM.
    """
    if not client:
        return "Local client not initialized. Is the openai library installed?"
    
    # Simple history management: just the prompt if string, or passed list
    messages = []
    if isinstance(prompt, str):
        # Fallback if a raw string is passed
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": str(prompt)}]

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
        logger.error(f"Local API Error: {error_msg}")
        return f"I encountered an error with the local model: {error_msg}"

