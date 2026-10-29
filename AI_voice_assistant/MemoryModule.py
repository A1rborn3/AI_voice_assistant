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
    """
    Assistant persona + recent conversation (memory.json / .conversation.json).
    Facts about people live in PeopleMemory (config/people.db): they are
    learned from the conversation every FACT_EXTRACTION_EVERY exchanges and
    only the relevant ones are added to each request (see memory_context).
    """

    def __init__(self, path, example_path=None, people=None):
        # store path as a Path object for reliable filesystem ops
        self.path = Path(path)
        self.example_path = Path(example_path) if example_path else self.path.with_name("memory.example.json")
        self.conversation_path = self.path.with_name(self.path.stem + ".conversation.json")
        self.conversation = ConversationQueue(max_size=10)
        # exchanges added since facts were last learned from the conversation
        self.counter = 0
        # Guards data/conversation/counter and file writes; fact learning runs
        # in a background thread while the main loop keeps adding messages.
        self._lock = threading.RLock()
        self._update_running = False
        self._people = people
        self._load()

    @property
    def people(self):
        """The PeopleMemory store (the shared one unless given); migrates the old profile on first use."""
        if self._people is None:
            import PeopleMemory
            self._people = PeopleMemory.get_people_memory()
        self._migrate_profile()
        return self._people

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
        if "personality" in self.data["user"]:
            # Legacy profile: only kept until it has been imported into PeopleMemory
            self.data["user"]["personality"] = _normalise_profile(self.data["user"].get("personality"))
        return json.dumps(self.data, sort_keys=True, default=str) != before

    def _migrate_profile(self):
        """One-off: turn the old JSON profile into facts, then drop it from memory.json."""
        with self._lock:
            if self.data["meta"].get("profile_migrated") or self._people is None:
                return
            profile = self.data["user"].get("personality")
            self.data["meta"]["profile_migrated"] = True
        try:
            added = self._people.import_profile(self.get_speaker_name(), profile) if profile else 0
            logger.info(f"Imported {added} facts from the old JSON profile into people memory")
            with self._lock:
                self.data["user"].pop("personality", None)
                self.save()
        except Exception:
            logger.exception("Profile migration failed; will retry next start")
            with self._lock:
                self.data["meta"]["profile_migrated"] = False

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

    def get_speaker_name(self) -> str:
        """Who the assistant is talking to (the main user until speaker recognition exists)."""
        return (self.data["user"].get("name") or "").strip() or "The user"

    def memory_context(self, query: str) -> str:
        """Only the remembered facts relevant to `query`, for the prompt."""
        try:
            return self.people.context_for(query)
        except Exception:
            logger.exception("People memory lookup failed")
            return ""

    def get_user_profile_text(self) -> str:
        """A few recent facts about the speaker (for prompts with no user request, e.g. alerts)."""
        try:
            facts = self.people.facts_about(self.get_speaker_name())[-5:]
        except Exception:
            logger.exception("People memory lookup failed")
            return ""
        return "\n".join(f"- {f}" for f in facts)

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
        """Store a conversation pair; every FACT_EXTRACTION_EVERY pairs, learn facts in the background."""
        with self._lock:
            self.counter += 1
            self.conversation.add(user_input, assistant_output)
            start_update = self.counter >= max(settings.FACT_EXTRACTION_EVERY, 1) and not self._update_running
            if start_update:
                self._update_running = True
                pending = self.conversation.to_list()[-min(self.counter, self.conversation.max_size):]
                self.counter = 0
        if start_update:
            # Run in a daemon thread so it doesn't block the main program
            t = threading.Thread(target=self._run_learn_safe, args=(pending,), daemon=True)
            t.start()

    def _run_learn_safe(self, pairs):
        """Wrapper to catch exceptions in the background thread."""
        try:
            self.learn_facts(pairs)
        except Exception:
            logger.exception("Learning facts failed in background thread")
        finally:
            with self._lock:
                self._update_running = False

    def get_recent_conversation(self):
        with self._lock:
            return self.conversation.get_history()

    def learn_facts(self, pairs=None) -> list:
        """Extract lasting facts from these (user, assistant) pairs into people memory."""
        import PeopleMemory
        if pairs is None:
            with self._lock:
                pairs = self.conversation.to_list()
        store = self.people
        store.backfill_embeddings()
        learned = PeopleMemory.extract_and_store(store, pairs, self.get_speaker_name())
        if learned:
            logger.info(f"Learned {len(learned)} fact(s): {learned}")
        return learned
