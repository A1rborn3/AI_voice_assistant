import os
import json
import logging
import re
import APIManagerMain

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# LLM_BASE_URL: address of your llama.cpp / LM Studio server (set in .env)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.2-3b-instruct")

# Temperature controls how creative/varied the spoken responses are.
# 0.0 = fully deterministic (used internally for routing/JSON extraction)
# 0.7 = balanced, natural-sounding speech (good default for conversation)
# 1.0+ = very creative but less predictable
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

if OpenAI:
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key="local-placeholder")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        client = None
else:
    logger.error("OpenAI library not installed.")
    client = None

# ─── Tool Registry ────────────────────────────────────────────────────────────
# This is the canonical list of tools available in the system.
# Each entry maps a display name (for the router) to a callable and its schema.
TOOLS = {
    "create_reminder": {
        "description": "Set a reminder or alarm for the user at a specific date and time.",
        "schema": {
            "day":     {"type": "integer", "description": "Day of the month"},
            "month":   {"type": "integer", "description": "Month (1-12)"},
            "year":    {"type": "integer", "description": "4-digit year"},
            "hour":    {"type": "integer", "description": "Hour in 24-hour format"},
            "minute":  {"type": "integer", "description": "Minute (0-59)"},
            "context": {"type": "string",  "description": "What the reminder is for"}
        },
        "required": ["day", "month", "year", "hour", "minute", "context"],
        "callable": APIManagerMain.create_reminder,
    },
    "list_reminders": {
        "description": "List all active reminders and alarms.",
        "schema": {},
        "required": [],
        "callable": APIManagerMain.list_reminders,
    },
    "delete_reminder": {
        "description": "Delete an active reminder or alarm using its ID.",
        "schema": {
            "reminder_id": {"type": "string", "description": "The ID of the reminder to delete"}
        },
        "required": ["reminder_id"],
        "callable": APIManagerMain.delete_reminder,
    },
    "get_weather_data": {
        "description": "Get current weather or a 3-day forecast for the user's location.",
        "schema": {
            "forecast": {"type": "boolean", "description": "True for 3-day forecast, False for today only"}
        },
        "required": ["forecast"],
        "callable": APIManagerMain.get_weather_data,
    },
    "end_conversation": {
        "description": "End the conversation if the user indicates they are finished or wants to say goodbye.",
        "schema": {},
        "required": [],
        "callable": APIManagerMain.end_conversation,
    },
}


def _chat(messages: list, max_tokens: int = 256, temperature: float = 0.1) -> str:
    """Low-level helper: send messages to LLM and return content string."""
    if not client:
        return "[Error: LLM client not initialized]"
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return f"[Error calling LLM: {e}]"


# ─── Stage 1: Router ──────────────────────────────────────────────────────────
def route_intent(user_text: str) -> str:
    """
    Classify the user's intent as TOOL or CHAT.
    Returns 'TOOL' or 'CHAT'.
    """
    # Build a concise tool summary for the router so it knows what's available
    tool_summary = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier. Your only job is to decide if the user's message "
                "requires calling a backend tool, or if it is just normal conversation.\n\n"
                "Available tools:\n"
                f"{tool_summary}\n\n"
                "Examples:\n"
                "- \"set an alarm for 7am\": TOOL\n"
                "- \"what's the weather?\": TOOL\n"
                "- \"that's all, thanks\": TOOL\n"
                "- \"goodbye\": TOOL\n"
                "- \"how are you?\": CHAT\n"
                "- \"tell me a joke\": CHAT\n\n"
                "If the user's message matches the PURPOSE of any tool above, respond with exactly: TOOL\n"
                "Otherwise respond with exactly: CHAT\n"
                "Output only one word. No explanation."
            )
        },
        {"role": "user", "content": user_text}
    ]

    result = _chat(messages, max_tokens=10, temperature=0.0)
    # Normalize – capture the full output to match against tool names
    result_clean = result.strip().upper()
    
    # Check if the model said "TOOL" OR mentioned any of our tool names specifically
    tool_names = [name.upper() for name in TOOLS.keys()]
    is_tool_by_llm = "TOOL" in result_clean or any(name in result_clean for name in tool_names)
    
    # MANUAL KEYWORD OVERRIDE: If STT hears these words, force TOOL path
    # This fixes issues where STT typos like "sit and alarm" confuse the router.
    trigger_words = ["alarm", "reminder", "weather", "forecast"]
    is_tool_by_keyword = any(word in user_text.lower() for word in trigger_words)
    
    intent = "TOOL" if (is_tool_by_llm or is_tool_by_keyword) else "CHAT"
    
    # Print to console for user visibility
    print(f"[Router] {intent} (LLM said: '{result.strip()}', Keyword Match: {is_tool_by_keyword})")
    return intent


# ─── Stage 2a: Tool Executor ──────────────────────────────────────────────────
def execute_tool(user_text: str, now_str: str) -> str:
    """
    Ask the LLM to select & parameterize the correct tool, then execute it.
    Returns a plain-text result string (tool output).
    """
    # Build a concise schema block for each tool
    schema_blocks = []
    for name, info in TOOLS.items():
        params = ", ".join(
            f"{k} ({v['type']}): {v['description']}"
            for k, v in info["schema"].items()
        ) if info["schema"] else "(no parameters required)"
        schema_blocks.append(f"Tool: {name}\n  Description: {info['description']}\n  Parameters: {params}")

    tool_docs = "\n\n".join(schema_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool selector and parameter extractor. "
                "Given the user's request, select the most appropriate tool and output "
                "ONLY a single-line JSON object with this structure:\n"
                '{"tool": "<tool_name>", "args": {<parameter key-value pairs>}}\n\n'
                "Rules:\n"
                "- Output ONLY the raw JSON. No explanation, no extra text.\n"
                "- Use 24-hour time for hours (e.g., 7:30 PM is hour: 19, minute: 30).\n"
                "- Match the date to the current date provided below unless specified otherwise.\n"
                "- If a tool takes no parameters, use: {\"tool\": \"<name>\", \"args\": {}}\n"
                f"- Current date/time: {now_str}\n\n"
                "Available tools:\n"
                f"{tool_docs}"
            )
        },
        {"role": "user", "content": user_text}
    ]

    raw = _chat(messages, max_tokens=120, temperature=0.0)
    logger.info(f"[Tool Executor] LLM raw output: {raw}")

    # Extract JSON from the response (handles markdown code blocks too)
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return f"[Error: could not parse tool JSON from: {raw}]"

    try:
        parsed = json.loads(json_match.group())
        tool_name = parsed.get("tool", "")
        args = parsed.get("args", {})
    except json.JSONDecodeError as e:
        return f"[Error: JSON decode failed: {e} | Raw: {raw}]"

    if tool_name not in TOOLS:
        return f"[Error: unknown tool '{tool_name}']"

    # Execute the tool
    try:
        print(f"\n[EXEC] Running tool: {tool_name} with {args}")
        func = TOOLS[tool_name]["callable"]
        if args:
            result = func(**args)
        else:
            result = func()
        logger.info(f"[Tool Executor] Result: {result}")
        return str(result)
    except Exception as e:
        return f"[Error executing tool '{tool_name}': {e}]"


# ─── Stage 2b: Conversational Response ───────────────────────────────────────
def ask_conversational(user_text: str, context_messages: list, now_str: str,
                       user_personality: str = "", assistant_context: str = "Samantha") -> str:
    """
    Generate a spoken, personality-rich conversational reply using Samantha's persona.
    `context_messages` is the list of prior {role, content} dicts (conversation history).
    """
    system_prompt = (
        "You are Samantha, a witty, charming, and highly capable personal AI assistant.\n"
        "You possess a distinct personality: helpful, calm, concise, and humorous. "
        "You love chatting naturally and always stay in character.\n\n"
        "Rules:\n"
        "1. Output a SINGLE line of plain text formatted for Text-to-Speech.\n"
        "2. Use natural pauses with commas or periods for TTS readability.\n"
        "3. Never use emojis, markdown, numbered lists, or symbols.\n"
        "4. Always use the metric system (Celsius, kilometres).\n"
        "5. Treat every user request as a fresh opportunity to be helpful. Even if a user repeats a question, answer it fully and politely as if it's the first time.\n"
        "6. CRITICAL: Never say 'I already told you', 'as I mentioned before', 'like I said', or anything dismissive. Do not lecture the user on what you have previously discussed.\n"
        f"\nCurrent date/time: {now_str}\n"
        f"Assistant persona: {assistant_context}\n"
        f"User context: {user_personality}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(context_messages)
    messages.append({"role": "user", "content": user_text})

    return _chat(messages, max_tokens=150, temperature=LLM_TEMPERATURE)


# ─── Tool Result → Conversational Confirmation ────────────────────────────────
def confirm_tool_result(tool_result: str, original_request: str, context_messages: list,
                        now_str: str, user_personality: str = "",
                        assistant_context: str = "Samantha") -> str:
    """
    Take a tool execution result and generate a friendly spoken confirmation using the Samantha persona.
    """
    synthesis_text = (
        f"You just performed an action for the user. Here is the result:\n"
        f"\"{tool_result}\"\n\n"
        f"The user originally said: \"{original_request}\"\n\n"
        "Confirm that you just completed the action in one natural, friendly, spoken sentence. "
        "Do not list previous actions or recap the conversation history. Just confirm this specific result."
    )

    return ask_conversational(synthesis_text, context_messages, now_str, user_personality, assistant_context)
