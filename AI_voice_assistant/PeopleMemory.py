"""
People memory (small RAG).
==========================
Instead of pasting one big profile into every prompt, the assistant keeps
short facts about people in SQLite, e.g. "Sarah is allergic to peanuts".

* learning  — every FACT_EXTRACTION_EVERY exchanges the LLM pulls lasting
              facts the user stated out of the recent conversation
              (extract_and_store); near-duplicates are merged and facts
              that change an older one replace it
* retrieval — each request gets only the few facts relevant to what was
              said (context_for), found by embedding similarity
              (nomic-embed-text via the LLM server), plus facts about any
              known person named in the request. Without embeddings it
              falls back to keyword matching.
* tools     — "what do you know about Sarah?", "forget that I like jazz"
"""

import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path

import settings

logger = logging.getLogger(__name__)

MAX_FACT_CHARS = 160
MAX_FACTS_PER_EXTRACTION = 8
REPLACE_SIMILARITY = 0.6          # how close an old fact must be to count as the one being replaced
PERSON_MENTION_LIMIT = 3          # facts added for each known person named in the request

# What the model must never store as a "fact" (requests, not facts)
_COMMAND_RE = re.compile(r"\b(?:set|remind|timer|alarm|add .* to|turn (?:up|down)|what'?s the weather|tell me a)\b",
                         re.IGNORECASE)


def _key(person: str) -> str:
    return " ".join(person.strip().lower().split())


def _display(person: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in person.strip().split())


class PeopleMemory:
    def __init__(self, path, embed=None):
        import AssistantTools
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._embed = embed or AssistantTools._embed
        self._cosine = AssistantTools._cosine
        self._keywords = AssistantTools._keywords
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("""CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person TEXT NOT NULL, person_key TEXT NOT NULL, fact TEXT NOT NULL,
            embedding TEXT, created REAL NOT NULL, updated REAL NOT NULL)""")
        self._db.execute("CREATE INDEX IF NOT EXISTS facts_person ON facts(person_key)")
        self._db.commit()

    # ── storage ──
    def _rows(self, person: str | None = None) -> list:
        sql = "SELECT id, person, fact, embedding, created, updated FROM facts"
        args = ()
        if person is not None:
            sql += " WHERE person_key = ?"
            args = (_key(person),)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY updated", args).fetchall()
        return [{"id": r[0], "person": r[1], "fact": r[2], "embedding": json.loads(r[3]) if r[3] else None,
                 "created": r[4], "updated": r[5]} for r in rows]

    def _similarity(self, a_text, a_vec, b) -> float:
        if a_vec and b["embedding"]:
            return self._cosine(a_vec, b["embedding"])
        a_words, b_words = self._keywords(a_text), self._keywords(b["fact"])
        return len(a_words & b_words) / max(len(a_words | b_words), 1)

    def add_fact(self, person: str, fact: str, replaces: str = "") -> str:
        """Store a fact. Returns 'added', 'updated' (near-duplicate refreshed) or 'rejected'."""
        person, fact = (person or "").strip(), " ".join((fact or "").split())
        if not person or len(person) > 40 or not (5 <= len(fact) <= MAX_FACT_CHARS) or _COMMAND_RE.search(fact):
            logger.info(f"[PeopleMemory] Rejected fact: {person!r}: {fact!r}")
            return "rejected"
        vector = self._embed(fact, "search_document")
        existing = self._rows(person)

        if replaces:
            old_vec = self._embed(replaces, "search_document")
            best = max(existing, key=lambda f: self._similarity(replaces, old_vec, f), default=None)
            if best and self._similarity(replaces, old_vec, best) >= REPLACE_SIMILARITY:
                self._delete([best["id"]])
                logger.info(f"[PeopleMemory] Replaced {best['fact']!r} with {fact!r}")
                existing = [f for f in existing if f["id"] != best["id"]]

        now = time.time()
        for f in existing:
            same = f["fact"].lower() == fact.lower() or (
                vector and f["embedding"] and self._cosine(vector, f["embedding"]) >= settings.MEMORY_DUPLICATE_SIMILARITY)
            if same:
                embedding = vector or f["embedding"]
                with self._lock:
                    self._db.execute("UPDATE facts SET fact = ?, embedding = ?, updated = ? WHERE id = ?",
                                     (fact, json.dumps(embedding) if embedding else None, now, f["id"]))
                    self._db.commit()
                return "updated"

        with self._lock:
            self._db.execute("INSERT INTO facts (person, person_key, fact, embedding, created, updated) "
                             "VALUES (?, ?, ?, ?, ?, ?)",
                             (_display(person), _key(person), fact, json.dumps(vector) if vector else None, now, now))
            self._db.commit()
        logger.info(f"[PeopleMemory] Learned: {fact}")
        return "added"

    def backfill_embeddings(self) -> int:
        """Embed facts stored while the embedding model was unavailable. Returns how many were fixed."""
        fixed = 0
        for row in self._rows():
            if row["embedding"]:
                continue
            vector = self._embed(row["fact"], "search_document")
            if not vector:
                break   # still unavailable; try again next time
            with self._lock:
                self._db.execute("UPDATE facts SET embedding = ? WHERE id = ?", (json.dumps(vector), row["id"]))
                self._db.commit()
            fixed += 1
        return fixed

    def _delete(self, ids: list):
        with self._lock:
            self._db.executemany("DELETE FROM facts WHERE id = ?", [(i,) for i in ids])
            self._db.commit()

    def people(self) -> list:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT DISTINCT person FROM facts ORDER BY person")]

    def facts_about(self, person: str) -> list:
        return [f["fact"] for f in self._rows(person)]

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # ── retrieval ──
    def mentioned_people(self, text: str) -> list:
        lowered = f" {text.lower()} "
        return [p for p in self.people() if re.search(rf"\b{re.escape(p.lower())}\b", lowered)]

    def search(self, query: str, k: int | None = None, min_similarity: float | None = None) -> list:
        """Facts most relevant to `query`, best first."""
        k = settings.MEMORY_FACTS_K if k is None else k
        threshold = settings.MEMORY_MIN_SIMILARITY if min_similarity is None else min_similarity
        rows = self._rows()
        if not rows or k <= 0:
            return []
        q_vec = self._embed(query, "search_query") if any(r["embedding"] for r in rows) else None
        scored = []
        for r in rows:
            if q_vec and r["embedding"]:
                score = self._cosine(q_vec, r["embedding"])
                if score >= threshold:
                    scored.append((score, r))
            else:
                overlap = len(self._keywords(query) & self._keywords(r["fact"]))
                if overlap:
                    scored.append((overlap / 10, r))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [r for _, r in scored[:k]]

    def context_for(self, query: str) -> str:
        """The facts to put in the prompt for this request (empty if none are relevant)."""
        chosen, seen = [], set()
        for person in self.mentioned_people(query):
            for r in self._rows(person)[-PERSON_MENTION_LIMIT:]:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    chosen.append(r["fact"])
        for r in self.search(query):
            if r["id"] not in seen:
                seen.add(r["id"])
                chosen.append(r["fact"])
        return "\n".join(f"- {fact}" for fact in chosen)

    def forget(self, person: str, what: str) -> list:
        """Delete the person's fact that best matches `what` (or all of them for 'everything')."""
        rows = self._rows(person)
        if not rows:
            return []
        if what.strip().lower() in ("everything", "all", "all of it", "it all"):
            self._delete([r["id"] for r in rows])
            return [r["fact"] for r in rows]
        vec = self._embed(what, "search_query") if any(r["embedding"] for r in rows) else None
        best = max(rows, key=lambda r: self._similarity(what, vec, r))
        if self._similarity(what, vec, best) < (0.45 if vec else 0.15):
            return []
        self._delete([best["id"]])
        return [best["fact"]]

    # ── migration from the old profile JSON ──
    def import_profile(self, name: str, profile: dict) -> int:
        """Turn the old memory.json 'personality' profile into facts. Returns how many were added."""
        name = (name or "").strip() or "The user"
        identity = profile.get("identity", {}) if isinstance(profile, dict) else {}
        templates = {
            "prefered_name": "{n} prefers to be called {v}", "age": "{n} is {v} years old",
            "location": "{n} lives in {v}", "education": "{n}'s education: {v}",
            "field_of_study": "{n} studies {v}", "work": "{n} works as {v}",
        }
        candidates = []
        for field, template in templates.items():
            value = str(identity.get(field) or "").strip()
            if value:
                candidates.append(template.format(n=name, v=value))
        list_templates = {
            "skills": "{n} is skilled in {v}", "interests": "{n} is interested in {v}",
            "hobbies": "{n}'s hobbies include {v}", "projects": "{n} is working on {v}",
            "technical_preferences": "{n} prefers {v}", "assistant_expectations": "{n} wants the assistant to {v}",
        }
        sources = dict(profile) if isinstance(profile, dict) else {}
        sources["skills"] = identity.get("skills", [])
        # Placeholder values from the old template aren't facts about anyone
        template_defaults = {"maintain context", "accurate explanations"}
        sentence_fields = {"assistant_expectations", "technical_preferences"}
        for field, template in list_templates.items():
            for value in sources.get(field) or []:
                value = str(value).strip().rstrip(".")
                if not value or value.lower() in template_defaults:
                    continue
                if field in sentence_fields:   # "Provide accurate forecasts" reads as a clause
                    value = value[:1].lower() + value[1:]
                candidates.append(template.format(n=name, v=value))
        for value in identity.get("other_details") or []:
            if str(value).strip():
                candidates.append(f"{name}: {str(value).strip()}")
        return sum(self.add_fact(name, fact) == "added" for fact in candidates)

    def close(self):
        with self._lock:
            self._db.close()


# ─── Learning from conversation ───────────────────────────────────────────────
# One short call per user message: given a whole mixed conversation chunk,
# qwen3-4b returned {"facts": []} even for "she's allergic to peanuts", but it
# extracts reliably from a single statement. Assistant replies are never sent.
EXTRACTION_PROMPT = """Extract lasting personal facts from something {speaker} just said.
A lasting fact will still be true next week: names and relationships, pets, where someone lives,
work or study, likes and dislikes, allergies or health, birthdays, routines.
Facts mentioned in passing count. Requests, questions and one-off plans (tonight, tomorrow) are not facts.
Write each fact as a short sentence that starts with the person's name. Use "{speaker}" for the speaker.
If a fact changes a known fact (for example someone moved), put the old fact in "replaces", otherwise "".

Examples:
Said: "my brother Tom just started working at the fire station"
{{"facts": [{{"person": "Tom", "fact": "Tom is {speaker}'s brother", "replaces": ""}}, {{"person": "Tom", "fact": "Tom works at the fire station", "replaces": ""}}]}}
Said: "set a timer for 10 minutes"
{{"facts": []}}
Said: "Priya's coming round later, she doesn't eat meat"
{{"facts": [{{"person": "Priya", "fact": "Priya is vegetarian", "replaces": ""}}]}}

Known facts:
{known}

Said: "{statement}"
Output only JSON."""

# User messages that are plainly commands or questions to the assistant
_NOT_A_STATEMENT = re.compile(
    r"^\s*(?:hey |ok(?:ay)? |please |can you |could you )*"
    r"(?:set|remind|add|remove|delete|cancel|clear|snooze|turn|stop|start|check|read|list|"
    r"tell me a|what'?s the|what is the|how long|how many|how much|is the stove|what time|"
    r"what do you know|forget|calculate|convert)\b", re.IGNORECASE)


def _is_statement(text: str) -> bool:
    return len(text.split()) >= 3 and not _NOT_A_STATEMENT.search(text)


def extract_and_store(store: PeopleMemory, pairs: list, speaker: str) -> list:
    """Learn facts from the user's side of these (user, assistant) pairs. Returns the facts added/updated."""
    import LLMClient
    stored = []
    for user_text, _ in pairs:
        if not _is_statement(user_text):
            continue
        # Existing facts about whoever is mentioned, so a change can name what it replaces
        people = {speaker, *store.mentioned_people(user_text)}
        known = [f for p in people for f in store.facts_about(p)][-12:]
        prompt = EXTRACTION_PROMPT.format(speaker=speaker, statement=user_text.replace('"', "'"),
                                          known="\n".join(f"- {f}" for f in known) or "- none")
        raw = LLMClient.ask_raw(prompt, max_tokens=300)
        parsed = LLMClient.extract_json_object(raw)
        if not parsed or not isinstance(parsed.get("facts"), list):
            logger.warning(f"[PeopleMemory] No usable facts in extraction output: {raw!r}")
            continue
        said = user_text.lower()
        speaker_words = store._keywords(speaker)
        for item in parsed["facts"][:MAX_FACTS_PER_EXTRACTION]:
            if not isinstance(item, dict):
                continue
            person = str(item.get("person") or speaker)
            fact = str(item.get("fact") or "")
            # Keep only facts grounded in what the user actually said (stops the
            # prompt's examples or the model's own ideas being stored)
            words = store._keywords(fact) - store._keywords(person) - speaker_words - {f"{w}'s" for w in speaker_words}
            if words and not any(w.rstrip("s") in said for w in words):
                logger.info(f"[PeopleMemory] Skipped fact not grounded in the user's words: {fact!r}")
                continue
            if store.add_fact(person, fact, str(item.get("replaces") or "")) != "rejected":
                stored.append(fact)
    return stored


# ─── Shared instance + voice tools ────────────────────────────────────────────
_store = None
_store_lock = threading.Lock()


def get_people_memory() -> PeopleMemory:
    global _store
    with _store_lock:
        if _store is None:
            _store = PeopleMemory(settings.PEOPLE_DB_PATH)
        return _store


def set_people_memory(store: PeopleMemory | None):
    global _store
    with _store_lock:
        _store = store


def _speaker() -> str:
    import MemoryModule
    return MemoryModule.get_memory().get_speaker_name()


def _resolve_person(name: str) -> str:
    name = (name or "").strip()
    if not name or name.lower() in ("me", "i", "myself", "user", "the user", "you know about me"):
        return _speaker()
    return name


def recall_person(name=""):
    """Everything remembered about a person (the user if no name)."""
    person = _resolve_person(name)
    facts = get_people_memory().facts_about(person)
    if not facts:
        known = get_people_memory().people()
        hint = f" I do know a little about {', '.join(known)}." if known else ""
        return f"I don't know anything about {person} yet.{hint}"
    shown = facts[-8:]
    return f"Here's what I know about {person}: " + " ".join(f.rstrip(".") + "." for f in shown)


def forget_fact(what, person=""):
    """Forget one remembered fact about a person (or 'everything')."""
    who = _resolve_person(person)
    removed = get_people_memory().forget(who, str(what or ""))
    if not removed:
        return f"Error: I couldn't find anything about {who} matching {what}."
    if len(removed) == 1:
        return f"Okay, I've forgotten that {removed[0].rstrip('.')}."
    return f"Okay, I've forgotten everything I knew about {who}."
