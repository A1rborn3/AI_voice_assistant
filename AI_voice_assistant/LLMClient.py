"""
LLM client — single-call native tool calling
=============================================
`respond()` sends the user's message with the tool definitions in one request.
If the model asks for tools, they are executed and their results are sent back
so the model can phrase the spoken reply (one extra call only on the tool path).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import APIManagerMain
import AssistantTools
import HazardModule
import PeopleMemory
import settings

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

LLM_BASE_URL    = settings.LLM_BASE_URL
LLM_MODEL       = settings.LLM_MODEL
LLM_TEMPERATURE = settings.LLM_TEMPERATURE

# Rounds of tool calls allowed per request. After the last round the model is
# asked again WITHOUT tools to phrase the result: with tools still attached,
# small local models (llama-3.2-3b) answer "anything else?" instead of
# reporting what the tool returned. Raise this for models that chain tools well.
MAX_TOOL_ROUNDS = 1

FALLBACK_REPLY = "Sorry, I can't reach my language model right now."
EMPTY_REPLY = "Sorry, I didn't quite get that. Could you say it again?"
TOOL_RESULT_NUDGE = (
    "The tool results above are real, current data. Using them, answer the "
    "user's last request in one or two short spoken sentences."
)

if OpenAI:
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=settings.LLM_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        client = None
else:
    logger.error("OpenAI library not installed.")
    client = None

# ─── Tool Registry ────────────────────────────────────────────────────────────
# Canonical list of tools. `schema` holds JSON-schema properties; it is turned
# into the OpenAI `tools=` format and also used to validate arguments.
# `group` decides when the tool is offered (see TOOL_GROUPS / select_tools).
#
# Notes for small local models:
# * keep parameters few and simple (strings/numbers); llama-3.2-3b drops
#   fields from bigger schemas (e.g. day/month/year/hour/minute integers)
# * never leave a schema empty: it then emits malformed calls like
#   {"name": "x",} that the server drops. Empty schemas get the optional
#   PLACEHOLDER_ARG automatically.
PLACEHOLDER_ARG = "reason"

TOOLS = {
    # ── reminders ──
    "create_reminder": {
        "group": "reminders",
        "description": "Set a reminder or alarm. Alarms and reminders are the same thing. Use in_minutes for 'in 20 minutes', otherwise time (+ optional date).",
        "schema": {
            "context":    {"type": "string", "description": "What the reminder is for"},
            "time":       {"type": "string", "description": "Time in 24-hour HH:MM, e.g. '08:30' or '19:00'"},
            "date":       {"type": "string", "description": "YYYY-MM-DD, 'today' or 'tomorrow'. Omit for the next time that clock time comes round."},
            "repeat":     {"type": "string", "description": "none, daily, weekdays or weekly. Default none."},
            "in_minutes": {"type": "number", "description": "For relative reminders: minutes from now"},
        },
        "required": ["context"],
        "callable": APIManagerMain.create_reminder,
    },
    "list_reminders": {
        "group": "reminders",
        "description": "List the user's active reminders and alarms with their numbers.",
        "schema": {
            "search": {"type": "string", "description": "Optional word to filter by, e.g. 'gym'. Use an empty string for all reminders."}
        },
        "required": [],
        "callable": APIManagerMain.list_reminders,
    },
    "delete_reminder": {
        "group": "reminders",
        "description": "Delete an active reminder or alarm by its number. Call list_reminders first if you don't know the number.",
        "schema": {
            "reminder_id": {"type": "integer", "description": "The reminder's number, e.g. 2"}
        },
        "required": ["reminder_id"],
        "callable": APIManagerMain.delete_reminder,
    },
    "snooze_reminder": {
        "group": "reminders",
        "description": "Snooze the reminder or alarm that just went off.",
        "schema": {"minutes": {"type": "number", "description": "Minutes to snooze, default 10"}},
        "required": [],
        "callable": APIManagerMain.snooze_reminder,
    },
    # ── weather ──
    "get_weather_data": {
        "group": "weather",
        "description": "Get current weather or a 3-day forecast for the user's location.",
        "schema": {
            "forecast": {"type": "boolean", "description": "True for 3-day forecast, False for today only"}
        },
        "required": ["forecast"],
        "callable": APIManagerMain.get_weather_data,
    },
    # ── kitchen safety ──
    "hazard_status": {
        "group": "hazards",
        "description": "Is the stove on? Current kitchen safety status from the camera: stove state, how long it's been on, whether anyone is in the kitchen.",
        "schema": {},
        "required": [],
        "callable": HazardModule.hazard_status,
    },
    "check_stove": {
        "group": "hazards",
        "description": "Take a fresh look at the stove with the camera right now.",
        "schema": {},
        "required": [],
        "callable": HazardModule.check_stove,
    },
    "snooze_hazard_alerts": {
        "group": "hazards",
        "description": "Pause repeated stove safety alerts for some minutes (max 60).",
        "schema": {"minutes": {"type": "number", "description": "Minutes to snooze, default 15; 0 re-enables alerts"}},
        "required": [],
        "callable": HazardModule.snooze_hazard_alerts,
    },
    "recent_events": {
        "group": "hazards",
        "description": "What happened in the kitchen recently: stove on/off times and safety alerts.",
        "schema": {"hours": {"type": "number", "description": "How many hours back to look, default 12"}},
        "required": [],
        "callable": HazardModule.recent_events,
    },
    # ── timers ──
    "set_timer": {
        "group": "timers",
        "description": "Start a countdown timer, e.g. a 10 minute pasta timer.",
        "schema": {
            "minutes": {"type": "number", "description": "Length in minutes"},
            "seconds": {"type": "number", "description": "Extra seconds"},
            "label":   {"type": "string", "description": "Optional name, e.g. 'pasta'"},
        },
        "required": [],
        "callable": AssistantTools.set_timer,
    },
    "timer_status": {
        "group": "timers",
        "description": "How much time is left on the running timers.",
        "schema": {},
        "required": [],
        "callable": AssistantTools.timer_status,
    },
    "cancel_timer": {
        "group": "timers",
        "description": "Cancel a timer by its number or label.",
        "schema": {
            "timer_id": {"type": "integer", "description": "Timer number"},
            "label":    {"type": "string", "description": "Timer name, e.g. 'pasta'"},
        },
        "required": [],
        "callable": AssistantTools.cancel_timer,
    },
    # ── lists ──
    "add_to_list": {
        "group": "lists",
        "description": "Add items to a list (shopping list by default).",
        "schema": {
            "items":     {"type": "string", "description": "Comma-separated items, e.g. 'milk, eggs'"},
            "list_name": {"type": "string", "description": "shopping, to-do, or any name. Default shopping."},
        },
        "required": ["items"],
        "callable": AssistantTools.add_to_list,
    },
    "remove_from_list": {
        "group": "lists",
        "description": "Remove an item from a list.",
        "schema": {
            "item":      {"type": "string", "description": "The item to remove"},
            "list_name": {"type": "string", "description": "Default shopping"},
        },
        "required": ["item"],
        "callable": AssistantTools.remove_from_list,
    },
    "read_list": {
        "group": "lists",
        "description": "Read out what's on a list.",
        "schema": {"list_name": {"type": "string", "description": "Default shopping"}},
        "required": [],
        "callable": AssistantTools.read_list,
    },
    "clear_list": {
        "group": "lists",
        "description": "Empty a whole list. Only set confirm=true after the user has explicitly confirmed.",
        "schema": {
            "list_name": {"type": "string", "description": "Default shopping"},
            "confirm":   {"type": "boolean", "description": "True only once the user said yes to clearing it"},
        },
        "required": [],
        "callable": AssistantTools.clear_list,
    },
    # ── notes ──
    "remember_note": {
        "group": "notes",
        "description": "Save a fact the user wants remembered, e.g. 'the spare key is in the blue pot'.",
        "schema": {"text": {"type": "string", "description": "The fact, as a full sentence"}},
        "required": ["text"],
        "callable": AssistantTools.remember_note,
    },
    "recall_note": {
        "group": "notes",
        "description": "Search the user's saved notes, e.g. 'where is the spare key'.",
        "schema": {"query": {"type": "string", "description": "What to look for"}},
        "required": ["query"],
        "callable": AssistantTools.recall_note,
    },
    "delete_note": {
        "group": "notes",
        "description": "Delete a saved note by its number.",
        "schema": {"note_id": {"type": "integer", "description": "Note number"}},
        "required": ["note_id"],
        "callable": AssistantTools.delete_note,
    },
    # ── people memory ──
    "recall_person": {
        "group": "people",
        "description": "Say what you remember about a person, e.g. 'what do you know about Sarah' or 'what do you know about me'.",
        "schema": {"name": {"type": "string", "description": "The person's name, or 'me' for the user"}},
        "required": [],
        "callable": PeopleMemory.recall_person,
    },
    "forget_fact": {
        "group": "people",
        "description": "Forget something remembered about a person, e.g. 'forget that I like jazz'. Use 'everything' to forget all of it.",
        "schema": {
            "what":   {"type": "string", "description": "The fact to forget, in the user's words"},
            "person": {"type": "string", "description": "Whose fact it is; omit or 'me' for the user"},
        },
        "required": ["what"],
        "callable": PeopleMemory.forget_fact,
    },
    # ── maths ──
    "calculate": {
        "group": "maths",
        "description": "Work out arithmetic exactly. Always use this instead of doing maths yourself.",
        "schema": {"expression": {"type": "string", "description": "The sum in the user's terms, e.g. '(12.5 * 4) + 3', 'sqrt(81)', '17% of 240'"}},
        "required": ["expression"],
        "callable": AssistantTools.calculate,
    },
    "convert_units": {
        "group": "maths",
        "description": "Convert a value between units: length, weight, cooking volumes, speed, time, temperature.",
        "schema": {
            "value":     {"type": "number", "description": "The amount"},
            "from_unit": {"type": "string", "description": "e.g. 'cups', 'lb', 'fahrenheit'"},
            "to_unit":   {"type": "string", "description": "e.g. 'ml', 'kg', 'celsius'"},
        },
        "required": ["value", "from_unit", "to_unit"],
        "callable": AssistantTools.convert_units,
    },
    # ── device ──
    "set_volume": {
        "group": "volume",
        "description": "Set how loud the assistant speaks, 0 to 10.",
        "schema": {"level": {"type": "number", "description": "0 (silent) to 10 (loudest)"}},
        "required": ["level"],
        "callable": AssistantTools.set_volume,
    },
    "end_conversation": {
        "group": "core",
        "description": "End the conversation when the user says they are finished or says goodbye.",
        "schema": {},
        "required": [],
        "callable": APIManagerMain.end_conversation,
    },
}

# Tools whose raw results (JSON, long lists) need the LLM to turn them into
# speech. Every other tool returns a finished spoken sentence, which is said
# as-is: no second LLM call, and no chance for a small model to misreport it
# (seen live: "3.361 kilograms" for 1361 g, a volume change described as an
# alarm). Errors are spoken in fixed wording (see spoken_error).
PHRASED_BY_LLM = set()   # every tool now speaks for itself; kept as the hook for future raw-data tools

# Destructive or conversation-ending tools only run when the user's own words
# ask for it. Seen live from llama-3.2-3b: "where is the spare key" ->
# delete_note (the note was deleted), "yes, clear it" -> end_conversation.
_YES = r"yes|yeah|yep|sure|ok(?:ay)?|go ahead|do it|confirm"
WORD_GATES = {
    "end_conversation": re.compile(r"\b(?:bye|goodbye|good ?night|see you|that'?s (?:all|it|everything)|"
                                   r"that'?ll be all|thanks|thank you|cheers|no more|nothing else|"
                                   r"i'?m done|we'?re done|stop)\b", re.IGNORECASE),
    "delete_note":      re.compile(r"\b(?:delete|remove|forget|erase)\b", re.IGNORECASE),
    "forget_fact":      re.compile(r"\b(?:forget|delete|remove|erase)\b", re.IGNORECASE),
    "delete_reminder":  re.compile(r"\b(?:delete|remove|cancel|forget|get rid of|turn off)\b", re.IGNORECASE),
    "cancel_timer":     re.compile(r"\b(?:cancel|stop|delete|remove|kill|turn off)\b", re.IGNORECASE),
    "clear_list":       re.compile(rf"\b(?:clear|empty|wipe|delete|{_YES})\b", re.IGNORECASE),
}

# Keyword patterns that make a tool group available for a request. Sending a
# 3B model ~25 tool definitions every time wastes context and confuses it,
# so each request only gets the groups it plausibly needs ("core" always).
TOOL_GROUPS = {
    "reminders": r"alarm|remind|wake me|snooze|appointment",
    "weather":   r"weather|forecast|rain|sunny|sunshine|cloud|wind|cold|hot|warm|temperature|umbrella|jacket|snow|storm",
    "hazards":   r"stove|oven|cooker|hob|burner|kitchen|hazard|safe|safety|camera|fire|smoke|left (?:it |the \w+ )?on|turned off|snooze",
    "timers":    r"timer|countdown|time left|how long (?:is )?left",
    "lists":     r"\blists?\b|shopping|grocer|to-?do|\bbuy\b|\badd\b.*\bto\b|\bremove\b|\bclear\b",
    "notes":     r"remember|\bnotes?\b|where(?:'s| is| are| did| do)|recall|forget",
    "people":    r"know about|what do you (?:know|remember)|tell me about (?:me|myself)|who is|who's|forget (?:that|about|what|everything)",
    "maths":     r"calculat|\d\s*[-+*/x×÷^%]\s*\d|plus|minus|times|divided|percent|square root|convert|how many|"
                 r"celsius|fahrenheit|kilo|pound|ounce|gram|litre|liter|cups?\b|tablespoon|teaspoon|inch|feet|foot|mile|metre|meter",
    "volume":    r"volume|louder|quieter|softer|turn (?:it |yourself )?(?:up|down)|too loud|can't hear|mute",
}
_GROUP_RES = {g: re.compile(p, re.IGNORECASE) for g, p in TOOL_GROUPS.items()}


def select_tools(user_text: str, history: list | None = None) -> list:
    """Names of the tools to offer for this request."""
    # For short follow-ups the assistant's last reply counts too, so "yes, clear it"
    # still finds clear_list. Not for full sentences: tools offered only because of
    # the last reply prime the model to talk about them ("my cat is sick" -> kitchen safety)
    last_reply = ""
    if len(user_text.split()) <= 5:
        last_reply = next((m.get("content") or "" for m in reversed(history or []) if m.get("role") == "assistant"), "")
    text = f"{user_text}\n{last_reply}"
    groups = {"core"} | {g for g, rx in _GROUP_RES.items() if rx.search(text)}
    return [name for name, info in TOOLS.items() if info["group"] in groups]


def tool_specs(names: list | None = None) -> list:
    """TOOLS (or just `names`) in the OpenAI `tools=` format."""
    specs = []
    for name, info in TOOLS.items():
        if names is not None and name not in names:
            continue
        properties = info["schema"] or {
            PLACEHOLDER_ARG: {"type": "string", "description": "Optional short reason for calling this"}}
        specs.append({
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": {"type": "object", "properties": properties, "required": info["required"]},
            },
        })
    return specs


@dataclass
class AssistantReply:
    text: str
    ended: bool = False          # end_conversation was called
    error: bool = False          # the LLM could not be reached
    tools_used: list = field(default_factory=list)


# ─── Low-level helpers ────────────────────────────────────────────────────────
def _complete(messages: list, max_tokens: int = 256, temperature: float = 0.1, tools: list | None = None,
              tool_choice: str = "auto"):
    """Send messages to the LLM; return the response message, or None on error."""
    if not client:
        logger.error("LLM client not initialized")
        return None
    kwargs = {}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return None


def _chat(messages: list, max_tokens: int = 256, temperature: float = 0.1) -> str:
    """Send messages to the LLM and return the content string ('' on error)."""
    msg = _complete(messages, max_tokens=max_tokens, temperature=temperature)
    return (msg.content or "").strip() if msg else ""


def ask_raw(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Send a single user prompt with no persona or history (used for internal tasks)."""
    return _chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature)


def extract_json_object(raw: str) -> dict | None:
    """
    Return the first JSON object found in `raw` (tolerates markdown fences and
    surrounding text), or None if there isn't one.
    """
    if not raw:
        return None
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(raw, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = raw.find("{", start + 1)
    return None


# ─── Argument validation ──────────────────────────────────────────────────────
_TRUE_STRINGS = {"true", "yes", "1", "y"}
_FALSE_STRINGS = {"false", "no", "0", "n", ""}


def _coerce_arg(value, expected_type: str):
    """Convert an LLM-supplied argument to the type declared in the tool schema."""
    if expected_type == "integer":
        if isinstance(value, bool):
            raise ValueError(f"expected integer, got boolean {value}")
        return int(float(value)) if isinstance(value, str) else int(value)
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise ValueError(f"expected boolean, got {value!r}")
    if expected_type == "number":
        if isinstance(value, bool):
            raise ValueError(f"expected number, got boolean {value}")
        if isinstance(value, str):
            # "10 minutes" -> 10.0
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if not match:
                raise ValueError(f"expected number, got {value!r}")
            return float(match.group())
        return float(value)
    if expected_type == "string":
        return str(value)
    return value


def _validate_tool_args(tool_name: str, args) -> dict:
    """
    Keep only the arguments the tool declares, coerce their types and check
    that every required one is present. Raises ValueError on bad input.
    """
    if not isinstance(args, dict):
        raise ValueError(f"args must be an object, got {type(args).__name__}")
    schema = TOOLS[tool_name]["schema"]
    unknown = set(args) - set(schema) - {PLACEHOLDER_ARG}
    if unknown:
        logger.warning(f"[Tools] Dropping unknown args for {tool_name}: {sorted(unknown)}")

    clean = {}
    for key, spec in schema.items():
        value = args.get(key)
        # Small models fill unused optional fields with "" — treat that as "not given"
        if value is None or (spec["type"] != "string" and isinstance(value, str) and not value.strip()):
            continue
        if key in args:
            try:
                clean[key] = _coerce_arg(args[key], spec["type"])
            except (TypeError, ValueError) as e:
                if key == "confirm":
                    clean[key] = False   # unreadable confirmation = not confirmed (the safe side)
                    continue
                raise ValueError(f"invalid value for '{key}': {e}") from e

    missing = [k for k in TOOLS[tool_name]["required"] if k not in clean]
    if missing:
        raise ValueError(f"missing required argument(s): {', '.join(missing)}")
    return clean


def execute_tool_call(tool_name: str, arguments) -> str:
    """Validate and run one tool call. Errors come back as text for the model to explain."""
    if tool_name not in TOOLS:
        return f"Error: unknown tool '{tool_name}'."
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return f"Error: arguments were not valid JSON ({e})."
    try:
        args = _validate_tool_args(tool_name, arguments or {})
    except ValueError as e:
        return f"Error: bad arguments for {tool_name}: {e}."

    logger.info(f"[Tools] Running {tool_name} with {args}")
    try:
        result = str(TOOLS[tool_name]["callable"](**args))
    except Exception as e:
        logger.exception(f"[Tools] {tool_name} failed")
        return f"Error: {tool_name} failed ({e})."
    logger.info(f"[Tools] {tool_name} -> {result}")
    return result


# ─── Prompting ────────────────────────────────────────────────────────────────
def build_system_prompt(now_str: str, user_profile_text: str = "",
                        assistant_profile: dict | None = None, with_tools: bool = False,
                        speaker: str = "", tool_groups=None) -> str:
    """`user_profile_text` holds the remembered facts relevant to this request (PeopleMemory)."""
    """Persona comes from memory.json's `assistant` block; only the speech/tool rules live here."""
    profile = assistant_profile if isinstance(assistant_profile, dict) else {}
    name = profile.get("name") or "Samantha"
    persona = profile.get("personality") or "Helpful, calm, concise and humorous."
    purpose = profile.get("purpose") or "Personal voice assistant."
    requirements = profile.get("requirements") or ""

    try:
        now = datetime.strptime(now_str, "%Y-%m-%d %H:%M")
        weekday = now.strftime("%A")
        tomorrow = (now + timedelta(days=1)).strftime("%A %Y-%m-%d")
    except ValueError:
        weekday = tomorrow = ""

    lines = [
        f"You are {name}, a voice assistant.",
        f"Personality: {persona}",
        f"Purpose: {purpose}",
    ]
    if requirements:
        lines.append(f"Guidelines: {requirements}")
    lines += [
        "",
        "Speech rules (your reply is read aloud by text-to-speech):",
        "1. Reply in plain spoken sentences on a single line.",
        "2. Never use emojis, markdown, lists, code, JSON or symbols.",
        "3. Use the metric system (Celsius, kilometres).",
        "4. Answer repeated questions fully and politely; never say 'as I mentioned before'.",
        "5. Never promise to remind, watch, alert or do anything later unless a tool did it in this turn.",
    ]
    if with_tools:
        # Topic-specific rules only when those tools are offered: always mentioning
        # kitchen safety made the model bring it up for "my cat has been sick"
        groups = set(tool_groups) if tool_groups is not None else set(TOOL_GROUPS) | {"core"}
        lines += [
            "",
            "Tool rules:",
            "- Use a tool whenever the request needs one of your tools; for normal conversation just reply.",
            "- Call tools only through the tool-calling interface. Never write a function call or JSON in your reply.",
            "- Never say you set, added, deleted or checked something unless a tool result in this turn confirms it.",
        ]
        if "hazards" in groups:
            lines.append("- Never guess about the stove or kitchen safety: use the kitchen safety tools.")
        if "maths" in groups:
            lines.append("- For arithmetic and unit conversions always use calculate or convert_units.")
        if groups & {"reminders", "timers"}:
            lines += [
                "- Speech recognition makes mistakes: 'sit an alarm' means 'set an alarm'.",
                "- Resolve relative times like 'tomorrow at 7' against the current date/time; give times as 24-hour HH:MM.",
            ]
        if groups & {"reminders", "timers", "notes"}:
            lines.append("- Reminders, timers and notes are identified by number; 'alarm number one' means reminder_id 1.")
    lines += [
        "",
        f"Current date/time: {now_str} {weekday}".rstrip(),
        *([f"Tomorrow is {tomorrow}."] if tomorrow else []),
        f"You are talking to {speaker}." if speaker else "",
        "Things you remember that may be relevant (use only if they help; never mention this list):",
        user_profile_text or "- nothing relevant",
    ]
    return "\n".join(lines)


# When the user's own words are unambiguous, they win over the model's
# argument (llama-3.2-3b turns "every weekday" into repeat="weekly").
ARG_HINTS = {
    "create_reminder": [
        ("repeat", re.compile(r"\b(?:every )?week ?days?\b|\bmonday to friday\b", re.I), "weekdays"),
        ("repeat", re.compile(r"\bevery ?day\b|\bdaily\b|\beach day\b", re.I), "daily"),
        ("repeat", re.compile(r"\bevery week\b|\bweekly\b", re.I), "weekly"),
    ],
}
# Filled in only when the model left the argument out ("set a pasta timer for
# 10 minutes" sometimes arrives as set_timer(label="pasta") with no length).
ARG_FILLS = {
    "set_timer": [("minutes", re.compile(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", re.I)),
                  ("seconds", re.compile(r"(\d+)\s*(?:seconds?|secs?)\b", re.I))],
    "create_reminder": [("in_minutes", re.compile(r"\bin\s+(\d+)\s*(?:minutes?|mins?)\b", re.I))],
}


def _apply_arg_hints(name: str, args, user_text: str):
    if not isinstance(args, dict):
        return args
    args = dict(args)
    for key, pattern, value in ARG_HINTS.get(name, []):
        if pattern.search(user_text):
            if args.get(key) != value:
                logger.info(f"[Tools] {name}.{key}: {args.get(key)!r} -> {value!r} (from the user's words)")
            args[key] = value
            break
    for key, pattern in ARG_FILLS.get(name, []):
        if args.get(key) in (None, "", 0, "0"):
            match = pattern.search(user_text)
            if match:
                logger.info(f"[Tools] {name}.{key} filled from the user's words: {match.group(1)}")
                args[key] = match.group(1)
    if name == "create_reminder":
        # Only repeat when the user said so (the model added repeat="daily" to "in 20 minutes")
        if not any(p.search(user_text) for k, p, _ in ARG_HINTS["create_reminder"] if k == "repeat"):
            args["repeat"] = "none"
        # Relative time only when the user gave one; "in 2 hours" -> 120 minutes
        hours = re.search(r"\bin\s+(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b", user_text, re.I)
        if hours:
            args["in_minutes"] = str(float(hours.group(1)) * 60)
        elif not re.search(r"\bin\s+\d+\s*(?:minutes?|mins?)\b", user_text, re.I):
            args.pop("in_minutes", None)
    if name == "set_timer":
        # "5 seconds" arrived as minutes=5, seconds=5: drop units the user never said
        if args.get("minutes") not in (None, "", 0, "0") and not re.search(r"\b(?:minutes?|mins?|hours?|hrs?)\b", user_text, re.I):
            logger.info("[Tools] set_timer: dropping minutes the user didn't ask for")
            args.pop("minutes")
        if args.get("seconds") not in (None, "", 0, "0") and not re.search(r"\b(?:seconds?|secs?)\b", user_text, re.I):
            args.pop("seconds")
    return args


def _infer_tool_from_args(args: dict, offered: list | None) -> str | None:
    """The single offered tool whose schema contains all these keys and its required ones."""
    keys = set(args) - {PLACEHOLDER_ARG}
    if not keys:
        return None
    candidates = [
        name for name in (offered or TOOLS)
        if TOOLS[name]["schema"] and keys <= set(TOOLS[name]["schema"])
        and set(TOOLS[name]["required"]) <= keys
    ]
    return candidates[0] if len(candidates) == 1 else None


_CALL_RE = re.compile(r"^`*\s*(\w+)\s*\((.*)\)\s*\.?`*$", re.DOTALL)          # tool(a=1, b="x")
# Everything else seen live: `tool a=1`, `tool, a=1`, `'tool', a=1`, `tool: a=1`, `tool shopping`
_LOOSE_RE = re.compile(r"^[`'\"]*\s*(\w+)[`'\"]*\s*(?:[,:]\s*|\s+|$)(.*?)[`.]*\s*$", re.DOTALL)
_KV_RE = re.compile(r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s,]+)")


def _literal(text: str):
    import ast
    text = text.strip()
    try:
        return ast.literal_eval({"true": "True", "false": "False", "null": "None"}.get(text, text))
    except (ValueError, SyntaxError):
        return text.strip("\"'")


def _parse_code_call(content: str):
    """
    Parse a tool call written as code, e.g. `recall_note(query="spare key")`,
    `read_list("shopping")` or `read_list list_name="shopping"`.
    Returns (name, args) or None.
    """
    import ast
    text = content.strip()
    match = _CALL_RE.match(text)
    # Tool names are matched case-insensitively (llama wrote "Read_list(...)")
    if match and match.group(1).lower() in TOOLS:
        name, inner = match.group(1).lower(), match.group(2)
        normalised = re.sub(r"\b(true|false|null)\b",
                            lambda m: {"true": "True", "false": "False", "null": "None"}[m.group(1)], inner)
        try:
            call = ast.parse(f"f({normalised})", mode="eval").body
            order = list(TOOLS[name]["schema"])
            args = {order[i]: ast.literal_eval(a) for i, a in enumerate(call.args) if i < len(order)}
            args.update({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg})
            return name, args
        except (SyntaxError, ValueError):
            return name, {k: _literal(v) for k, v in _KV_RE.findall(inner)}
    match = _LOOSE_RE.match(text)
    if match and match.group(1).lower() in TOOLS:
        name, rest = match.group(1).lower(), match.group(2).strip()
        pairs = _KV_RE.findall(rest)
        if pairs:
            return name, {k: _literal(v) for k, v in pairs}
        if not rest:
            return name, {}
        # Positional values ("read_list shopping"); only for snake_case names so
        # a sentence that happens to start with "calculate" isn't mistaken for a call
        if "_" in name and len(rest) <= 80 and "=" not in rest:
            values = [v.strip(" '\"") for v in rest.split(",") if v.strip(" '\"")]
            order = list(TOOLS[name]["schema"])
            return name, {order[i]: _literal(v) for i, v in enumerate(values) if i < len(order)}
    return None


# A reply that claims an action or reports data, made without any tool having
# run, is almost certainly invented (seen live: "The shopping list has been
# cleared" with nothing cleared).
_CLAIM_RE = re.compile(
    r"\b(?:added|cleared|removed|deleted|saved|noted|recorded|snoozed|started|cancell?ed|adjusted|done|"
    r"updated|stored|created|changed|scheduled|"   # qwen3-4b: "Shopping list updated: milk, eggs, bread."
    r"clearing|adding|removing|deleting|saving|setting|cancell?ing|"
    r"turned (?:up|down|off|on)|set (?:for|to|a|an|the)|has been|have been|is now|running|"
    r"on your (?:\w+ )?list|\w+'? list(?=\s*:)|you have \d|you have (?:one|two|three|no)|left on|minutes? left|"
    r"you told me|stoves?|alarms?|alerts?|reminders?|timers?|burners?|"
    # promises and self-run confirmations, seen from qwen3-4b with history in context:
    # "I'll remember that ...", "Are you sure you want to clear the shopping list?"
    r"i'?ll|i will|i'?ve|remember(?:ed)?|are you sure|confirm|"
    # talking about the weather instead of fetching it ("I don't have current weather data")
    r"weather|forecast|degrees|rain)\b", re.IGNORECASE)


# Promises of future action ("I'll make sure to remind you", "I'll keep an eye on it")
# ("I'll remember that" is allowed: people memory really does remember.)
_PROMISE_RE = re.compile(
    r"\b(?:i'?ll|i will|i'?m going to|let me)\s+(?:make sure|remind|keep an eye|monitor|watch|alert|notify|"
    r"let you know|check|set|add|track|look into|look up|double-check)\b", re.IGNORECASE)
# Added to the system prompt for a fresh retry. (A system message placed after the
# assistant's turn makes qwen3-4b return an empty reply, so the retry starts over.)
NO_PROMISES_NUDGE = ("For this reply: you have no tool for what they said, so do not offer or promise to check, "
                     "set up, look up, monitor or remind. Respond warmly and naturally in one or two sentences, "
                     "like a friend would. You may say you'll remember it.")


def _drop_promises(text: str) -> str:
    """Last resort: remove sentences that promise an action nothing performed."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if not _PROMISE_RE.search(s)]
    return " ".join(kept) if kept else "Thanks for letting me know."


def _claims_action(text: str, offered: list | None = None) -> bool:
    if _CLAIM_RE.search(text):
        return True
    # A number in the answer while maths tools are on offer: the model did the sum
    # itself (qwen3-4b: "12 times 34 is 408") instead of using the calculator
    if offered and any(TOOLS[n]["group"] == "maths" for n in offered) and re.search(r"\d", text):
        return True
    return False


def _looks_like_json(text: str) -> bool:
    """Raw JSON or a written-out tool call: never something to read aloud."""
    stripped = text.strip().strip("`").strip()
    if stripped.startswith("{") and stripped.rstrip(".").endswith("}"):
        return True
    return _parse_code_call(text) is not None


def spoken_error(result: str) -> str:
    """
    A tool error in fixed wording. Not handed to the LLM: a small model was seen
    turning an error into "Reminder number two set" when nothing was set.
    """
    message = result[len("Error:"):].strip() if result.startswith("Error:") else result
    if message.startswith("bad arguments") or message.startswith("arguments were not valid"):
        return "Sorry, I didn't catch all the details for that. Could you say it again?"
    if message.startswith("unknown tool") or " failed (" in message:
        return "Sorry, something went wrong while doing that."
    if not message.startswith(("I ", "I'")):
        message = message[:1].lower() + message[1:]
    return f"Sorry, {message}"


def _tool_calls_from_message(msg, offered: list | None = None) -> list:
    """
    Return [(id, name, arguments)] for the tool calls in a response message.
    Small local models sometimes write the call as JSON text instead of using
    the native field, so that form is accepted too.
    """
    calls = []
    for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
        calls.append((tc.id or f"call_{i}", tc.function.name, tc.function.arguments))
    if calls:
        return calls

    # Text-form calls only count for tools that were actually offered
    text_calls = _text_tool_calls((msg.content or "").strip(), offered)
    allowed = [c for c in text_calls if offered is None or c[1] in offered]
    if text_calls and not allowed:
        logger.info(f"[Tools] Ignoring text-form call to a tool that wasn't offered: {text_calls[0][1]}")
    return allowed


def _text_tool_calls(content: str, offered: list | None) -> list:
    calls = []
    # Just the bare name of a tool that needs no arguments, e.g. "end_conversation"
    bare = content.strip("`'\". ").lower()
    if bare in TOOLS and not TOOLS[bare]["required"]:
        logger.info(f"[Tools] Parsed bare tool name: {bare}")
        return [("call_text_0", bare, {})]
    code_call = _parse_code_call(content)
    if code_call:
        logger.info(f"[Tools] Parsed code-style tool call: {code_call[0]}")
        return [("call_text_0", code_call[0], code_call[1])]
    if content.startswith("{") or content.startswith("```"):
        obj = extract_json_object(content)
        if obj and not any(k in obj for k in ("name", "tool", "function")):
            # Arguments only, e.g. {"items": "milk", "list_name": "shopping"}
            name = _infer_tool_from_args(obj, offered)
            if name:
                logger.info(f"[Tools] Inferred tool from bare arguments: {name}")
                return [("call_text_0", name, obj)]
        if obj:
            # Seen from llama-3.2-3b: {"name": ...}, {"tool": ...} and, after a few
            # turns, {"function": "clear_list", "parameters": {...}}
            name = obj.get("name") or obj.get("tool") or obj.get("function")
            if isinstance(name, dict):   # {"function": {"name": ..., "arguments": ...}}
                obj = {**obj, **name}
                name = name.get("name")
            if name in TOOLS:
                args = obj.get("parameters", obj.get("arguments", obj.get("args", {})))
                logger.info(f"[Tools] Parsed text-form tool call: {name}")
                calls.append(("call_text_0", name, args))
    return calls


def respond(user_text: str, history: list, now_str: str,
            user_profile_text: str = "", assistant_profile: dict | None = None,
            speaker: str = "") -> AssistantReply:
    """
    Main entry point: one request with tools attached. Normal conversation is
    answered in that single call; tool requests get one follow-up call to
    phrase the result.
    """
    # Small models get worse at tool calling as context grows; keep recent turns only
    history = history[-2 * settings.LLM_HISTORY_TURNS:] if settings.LLM_HISTORY_TURNS > 0 else []
    offered = select_tools(user_text, history)
    logger.debug(f"[Tools] Offering: {offered}")
    messages = [{"role": "system", "content": build_system_prompt(
        now_str, user_profile_text, assistant_profile, with_tools=True, speaker=speaker,
        tool_groups={TOOLS[n]["group"] for n in offered})}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    tools_used = []
    retried = False
    force_tool = False
    round_no = 0
    while round_no <= MAX_TOOL_ROUNDS:
        # Final round has no tools, which forces a spoken answer
        tools = tool_specs(offered) if round_no < MAX_TOOL_ROUNDS else None
        temperature = 0.0 if retried else LLM_TEMPERATURE
        msg = _complete(messages, max_tokens=200, temperature=temperature, tools=tools,
                        tool_choice="required" if force_tool else "auto")
        if msg is None:
            return AssistantReply(FALLBACK_REPLY, error=True, tools_used=tools_used)

        calls = _tool_calls_from_message(msg, offered) if tools else []
        blocked = {name for _, name, _ in calls if name in WORD_GATES and not WORD_GATES[name].search(user_text)}
        if blocked:
            logger.warning(f"[LLM] Blocked {sorted(blocked)}: the user's words don't ask for that; retrying without")
            offered = [n for n in offered if n not in blocked]
            calls = [c for c in calls if c[1] not in blocked]
            if not calls:
                if not retried and any(TOOLS[n]["group"] != "core" for n in offered):
                    retried = True
                    continue
                return AssistantReply(EMPTY_REPLY, error=True, tools_used=tools_used)
        if not calls:
            text = (msg.content or "").strip()
            # Only force a tool when the user's OWN words point at one; groups offered
            # because of the previous reply ("kitchen safety") made "my cat has been
            # sick" force a stove-status check
            has_real_tools = any(TOOLS[n]["group"] != "core" for n in select_tools(user_text))
            if (round_no == 0 and not retried and has_real_tools and text
                    and not _looks_like_json(text) and _claims_action(text, offered)):
                # Claimed an action / reported data without calling anything: make it use a tool
                logger.warning(f"[LLM] Reply claims an action without a tool call, forcing a tool: {text!r}")
                retried = force_tool = True
                # Keep only the last exchange (follow-ups like "yes, clear it" need it)
                messages = [messages[0], *messages[-3:]] if len(messages) >= 4 else [messages[0], messages[-1]]
                continue
            if _looks_like_json(text) and not retried:
                # Garbled tool call or JSON instead of speech: retry once,
                # deterministically and without the (distracting) history
                logger.warning(f"[LLM] Reply was JSON, retrying: {text}")
                retried = True
                if round_no == 0:
                    messages = [messages[0], messages[-1]]
                continue
            if not text or _looks_like_json(text):
                # e.g. the server dropped a malformed tool call; don't go silent
                logger.warning(f"[LLM] No usable reply: {text!r}")
                return AssistantReply(EMPTY_REPLY, error=True, tools_used=tools_used)
            if not tools_used and _PROMISE_RE.search(text):
                # "I'll make sure to remind you..." with nothing set: ask once for an honest reply
                logger.warning(f"[LLM] Reply promises an action no tool performed, regenerating: {text!r}")
                redo = _complete([{"role": "system", "content": messages[0]["content"] + "\n\n" + NO_PROMISES_NUDGE},
                                  *messages[1:]],
                                 max_tokens=200, temperature=LLM_TEMPERATURE)
                better = (redo.content or "").strip() if redo else ""
                if better and not _looks_like_json(better):
                    text = better
                if _PROMISE_RE.search(text):
                    text = _drop_promises(text)
            return AssistantReply(text, tools_used=tools_used)

        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": cid, "type": "function", "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                }}
                for cid, name, args in calls
            ],
        })
        results = []
        for cid, name, args in calls:
            tools_used.append(name)
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    pass   # execute_tool_call reports it
            result = execute_tool_call(name, _apply_arg_hints(name, args, user_text))
            if result == APIManagerMain.CONVERSATION_ENDED:
                return AssistantReply("", ended=True, tools_used=tools_used)
            results.append((name, result))
            messages.append({"role": "tool", "tool_call_id": cid, "content": result})

        errors = [result for _, result in results if result.startswith("Error")]
        if errors:
            return AssistantReply(spoken_error(errors[0]), tools_used=tools_used)
        if all(name not in PHRASED_BY_LLM for name, _ in results):
            return AssistantReply(" ".join(result for _, result in results), tools_used=tools_used)
        # Anchor the next answer to the results; with conversation history in
        # context, small models otherwise drift back to earlier turns (and
        # e.g. describe a timer as an alarm).
        names = ", ".join(dict.fromkeys(name for _, name, _ in calls))
        messages.append({"role": "system", "content": f"Tool used: {names}. {TOOL_RESULT_NUDGE}"})
        round_no += 1

    return AssistantReply(FALLBACK_REPLY, error=True, tools_used=tools_used)


def ask_conversational(user_text: str, context_messages: list, now_str: str,
                       user_profile_text: str = "", assistant_profile: dict | None = None,
                       speaker: str = "") -> str:
    """Persona reply without tools (used for proactive alerts)."""
    messages = [{"role": "system", "content": build_system_prompt(
        now_str, user_profile_text, assistant_profile, with_tools=False, speaker=speaker)}]
    messages.extend(context_messages)
    messages.append({"role": "user", "content": user_text})
    return _chat(messages, max_tokens=150, temperature=LLM_TEMPERATURE)
