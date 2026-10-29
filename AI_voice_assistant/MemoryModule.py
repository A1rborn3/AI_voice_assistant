import copy
import json
import threading
from pathlib import Path
from collections import deque
import logging

import settings

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = settings.MEMORY_PATH

# Used when neither memory.json nor memory.example.json exists
EMPTY_PROFILE = {
    "identity": {
        "name": "",
        "prefered_name": "",
        "age": "",
        "location": "",
        "education": "",
        "field_of_study": "",
        "work": "",
        "skills": [],
        "other_details": []
    },
    "interests": [],
    "hobbies": [],
    "projects": [],
    "technical_preferences": [],
    "assistant_expectations": [
        "maintain context",
        "accurate explanations"
    ]
}

DEFAULT_DATA = {
    "user": {
        "name": "",
        "location": "",
        "personality": EMPTY_PROFILE,
        "devices": []
    },
    "assistant": {
        "name": "Samantha",
        "personality": "Helpful, calm, concise, humorous",
        "purpose": "Smart home assistant that watches for safety hazards around the house",
        "requirements": "Provide accurate information, maintain context and assist with tasks. Keep responses clear and concise."
    },
    "meta": {
        "counter": 0
    }
}

# Process-wide shared instance (see get_memory)
_memory_instance = None
_memory_instance_lock = threading.Lock()


def get_memory(path=DEFAULT_MEMORY_PATH):
    """Return the shared Memory instance, creating it on first call."""
    global _memory_instance
    with _memory_instance_lock:
        if _memory_instance is None:
            _memory_instance = Memory(path)
        return _memory_instance


class ConversationQueue:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)

    def add(self, user_msg, assistant_msg):
        """Store a pair of user + assistant messages."""
        self.queue.append((user_msg, assistant_msg))

    def get_history(self):
        """Return formatted text for LLM context."""
        formatted = ""
        for user, assistant in self.queue:
            formatted += f"User: {user}\nAssistant: {assistant}\n"
        return formatted.strip()

    def load_from_list(self, conv):
        """Load conversation history from a list of pairs."""
        for user_msg, assistant_msg in conv:
            self.add(user_msg, assistant_msg)

    def to_list(self):
        """Export conversation history as a list of pairs."""
        return list(self.queue)


def _normalise_profile(value) -> dict:
    """
    The user profile is always stored as a dict. Older memory files kept it as
    a JSON string (or free text); convert those so nothing is lost.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        profile = copy.deepcopy(EMPTY_PROFILE)
        profile["identity"]["other_details"].append(value.strip())
        return profile
    return copy.deepcopy(EMPTY_PROFILE)


class Memory:
    def __init__(self, path, example_path=None):
        # store path as a Path object for reliable filesystem ops
        self.path = Path(path)
        self.example_path = Path(example_path) if example_path else self.path.with_name("memory.example.json")
        self.conversation_path = self.path.with_name(self.path.stem + ".conversation.json")
        self.conversation = ConversationQueue(max_size=10)
        # counter cycles 0..10 and is incremented each time a conversation pair is added
        self.counter = 0
        # Guards data/conversation/counter and file writes; the personality update
        # runs in a background thread while the main loop keeps adding messages.
        self._lock = threading.RLock()
        self._update_running = False
        self._load()

    def _initial_data(self) -> dict:
        """Starting memory for a fresh install: the committed example file, else built-in defaults."""
        try:
            with self.example_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return copy.deepcopy(DEFAULT_DATA)

    def _load(self):
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            logger.info(f"No memory file at {self.path}; creating one from {self.example_path.name}")
            self.data = self._initial_data()
            self._ensure_shape()
            self.save()
            return  # skip loading conversation
        except json.JSONDecodeError:
            # Back up corrupted file to avoid data loss, then create defaults
            try:
                corrupt_backup = self.path.with_suffix(self.path.suffix + ".corrupt")
                self.path.replace(corrupt_backup)
                logger.warning("Corrupted memory file moved to %s", corrupt_backup)
            except Exception:
                logger.exception("Failed to move corrupted memory file; it will be overwritten.")
            self.data = self._initial_data()
            self._ensure_shape()
            self.save()
            return

        if self._ensure_shape():
            self.save()

        # If file contained a persisted counter, restore it
        try:
            self.counter = int(self.data["meta"].get("counter", 0))
        except (TypeError, ValueError):
            self.counter = 0

        # Load conversation history if available
        try:
            with self.conversation_path.open("r", encoding="utf-8") as f:
                conv = json.load(f)
                if isinstance(conv, list):
                    self.conversation.load_from_list(conv)
        except FileNotFoundError:
            pass  # no conversation file yet - fine
        except json.JSONDecodeError:
            logger.warning(f"Ignoring corrupted conversation file {self.conversation_path}")

    def _ensure_shape(self) -> bool:
        """Fill in missing sections and migrate old formats. Returns True if anything changed."""
        before = json.dumps(self.data, sort_keys=True, default=str)
        for section in ("user", "assistant", "meta"):
            if not isinstance(self.data.get(section), dict):
                self.data[section] = copy.deepcopy(DEFAULT_DATA[section])
        for key, value in DEFAULT_DATA["assistant"].items():
            self.data["assistant"].setdefault(key, value)
        self.data["user"]["personality"] = _normalise_profile(self.data["user"].get("personality"))
        return json.dumps(self.data, sort_keys=True, default=str) != before

    def save(self):
        with self._lock:
            parent = self.path.parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

            # persist counter in a meta section so it survives restarts
            self.data.setdefault("meta", {})["counter"] = self.counter

            # Save profiles (atomic write)
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            tmp.replace(self.path)

            # Save conversation (trim then atomic replace)
            conv_list = self.conversation.to_list()[-self.conversation.max_size:]
            tmpc = self.conversation_path.with_suffix(".tmp")
            with tmpc.open("w", encoding="utf-8") as f:
                json.dump(conv_list, f, indent=4, ensure_ascii=False)
            tmpc.replace(self.conversation_path)

    def get_user_profile(self):
        return self.data["user"]

    def get_assistant_profile(self):
        return self.data["assistant"]

    def get_user_profile_text(self) -> str:
        """The learned user profile as compact JSON for prompts."""
        with self._lock:
            return json.dumps(self.data["user"].get("personality", {}), ensure_ascii=False)

    def get_history_messages(self) -> list:
        """Recent conversation as OpenAI-style {role, content} messages."""
        with self._lock:
            pairs = self.conversation.to_list()
        messages = []
        for user_msg, assistant_msg in pairs:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    def add_message_pair(self, user_input, assistant_output):
        """Store recent conversation pair and advance the 0..10 counter."""
        with self._lock:
            # increment and wrap back to 0 after reaching 10
            self.counter = (self.counter + 1) % 11
            self.conversation.add(user_input, assistant_output)
            # Trigger personality update when the counter reaches 10,
            # unless a previous update is still running
            start_update = self.counter == 10 and not self._update_running
            if start_update:
                self._update_running = True
        if start_update:
            # Run in a daemon thread so it doesn't block the main program
            t = threading.Thread(target=self._run_update_personality_safe, daemon=True)
            t.start()

    def _run_update_personality_safe(self):
        """Wrapper to catch exceptions in the background thread."""
        try:
            self.update_user_personality()
        except Exception:
            logger.exception("update_user_personality failed in background thread")
        finally:
            with self._lock:
                self._update_running = False

    def get_recent_conversation(self):
        with self._lock:
            return self.conversation.get_history()

    def update_user_personality(self):
        """
        Generate an updated user profile from recent conversation via the LLM.
        This function imports LLMClient lazily to avoid potential circular imports.
        Only a valid JSON object replaces the stored profile.
        """
        with self._lock:
            conversation_history = self.get_recent_conversation()
            personality = json.dumps(self.data["user"].get("personality", {}), indent=2, ensure_ascii=False)
        prompt = f"""Update the user's long-term profile using the recent conversation.
Do NOT describe personality types such as "outgoing," "inquisitive,"
"calm," etc. Focus ONLY on stable, factual, practical information
that is useful for future interactions.

Output strictly as JSON in this structure:

{json.dumps(EMPTY_PROFILE, indent=2)}

Rules:
- Keep entries short, factual, and verifiable from the conversation.
- If something is unknown, leave it empty, null without change.
- Merge new facts with the existing profile without removing correct ones.
- Only include information clearly mentioned in the conversation.
- Avoid assumptions or personality descriptors.
- Keep the final output under 200 words total.

Recent conversation:
{conversation_history}

Existing profile:
{personality}
"""
        try:
            # lazy import to reduce circular import risk
            import LLMClient
            raw = LLMClient.ask_raw(prompt, max_tokens=400)
            # Only accept a real JSON profile; LLM errors come back as plain
            # strings and must never overwrite the stored profile.
            parsed = LLMClient.extract_json_object(raw)
            if parsed:
                with self._lock:
                    self.data.setdefault("user", {})["personality"] = parsed
                    self.save()
                logger.info("User personality updated by LLM.")
            else:
                logger.warning(f"Personality update skipped; LLM output was not a JSON profile: {raw!r}")
        except Exception:
            logger.exception("Failed to update personality via LLM.")
