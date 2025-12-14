import json
from pathlib import Path
from collections import deque
import logging

logger = logging.getLogger(__name__)

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


class Memory:
    def __init__(self, path):
        # store path as a Path object for reliable filesystem ops
        self.path = Path(path)
        self.conversation_path = self.path.with_name(self.path.stem + ".conversation.json")
        self.conversation = ConversationQueue(max_size=10)
        # counter cycles 0..10 and is incremented each time a conversation pair is added
        self.counter = 0
        self._load()

    def _load(self):   
        dataDefault = {
                "user": {
                    "personality": {
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
                    },
                    "devices": [
                        "Placeholder",
                        "Placeholder"
                    ]
                },
                "assistant": {
                    "name": "Samantha",
                    "personality": "Helpful, calm, concise , humorous",
                    "purpose": "Smart home control and general voice assistant",
                    "requirements": "provide accurate information, maintain context, assist with tasks. Responses should be clear and concise, limiting to a few sentences unless more detail is requested. You can ask questions and make requests to further help."
                },
                "meta": {
                    "counter": 0
                }
            }
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            # Default hardcoded memory
            self.data = dataDefault
            # ensure counter persisted alongside default data
            # self.counter is already initialized in __init__
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
            self.data = dataDefault
            self.save()
            return

        # If file contained a persisted counter, restore it
        meta = self.data.get("meta")
        if isinstance(meta, dict):
            try:
                self.counter = int(meta.get("counter", 0))
            except (TypeError, ValueError):
                self.counter = 0
        else:
            self.counter = 0

        # Load conversation history if available
        try:
            with self.conversation_path.open("r", encoding="utf-8") as f:
                conv = json.load(f)
                if isinstance(conv, list):
                    self.conversation.load_from_list(conv)
        except FileNotFoundError:
            # no conversation file yet - fine
            pass
        # (handle JSONDecodeError by backing up corrupted file if desired)

    def save(self):
        parent = self.path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        # persist counter in a meta section so it survives restarts
        self.data.setdefault("meta", {})["counter"] = self.counter

        # Save profiles (atomic write optional)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        tmp.replace(self.path)  # atomic replace

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

    def add_message_pair(self, user_input, assistant_output):
        """Store recent conversation pair and advance the 0..10 counter."""
        # increment and wrap back to 0 after reaching 10
        self.counter = (self.counter + 1) % 11
        self.conversation.add(user_input, assistant_output)
        # Trigger personality update when the counter reaches 10
        if self.counter == 10:
            # Run in a daemon thread so it doesn't block the main program
            import threading
            t = threading.Thread(target=self._run_update_personality_safe, daemon=True)
            t.start()

    def _run_update_personality_safe(self):
        """Wrapper to catch exceptions in the background thread."""
        try:
            self.update_user_personality()
        except Exception:
            logger.exception("update_user_personality failed in background thread")

    def get_recent_conversation(self):
        return self.conversation.get_history()

    # placeholder for future automated memory wrapper
    def auto_update_from_text(self, user_input, assistant_output):
        pass
    
    def update_user_personality(self):
        """
        Generate an updated user personality from recent conversation via the LLM.
        This function imports LLMClient lazily to avoid potential circular imports.
        The returned text (if any) is stored and persisted.
        """
        conversation_history = self.get_recent_conversation()
        personality = self.data["user"].get("personality", "")
        prompt = f"""Update the user's long-term profile using the recent conversation.  
Do NOT describe personality types such as "outgoing," "inquisitive," 
"calm," etc. Focus ONLY on stable, factual, practical information 
that is useful for future interactions.

Output strictly as JSON in this structure:

{{
  "identity": {{
    "name": "",
    "prefered_name": "",
    "age": "",
    "location": "",
    "education": "",
    "field_of_study": "",
    "work": "",
    "skills": [],  
    "other_details": []
  }},
  "interests": [],
  "hobbies": [],
  "projects": [],
  "technical_preferences": [],
  "assistant_expectations": []
}}

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
            updated = LLMClient.askLLM(prompt)
            if isinstance(updated, str):
                updated = updated.strip()
            if updated:
                # store concise personality update
                self.data.setdefault("user", {})["personality"] = updated
                self.save()
                logger.info("User personality updated by LLM.")
            else:
                logger.debug("LLM returned empty personality update.")
        except Exception:
            logger.exception("Failed to update personality via LLM.")
