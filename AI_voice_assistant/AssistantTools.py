"""
Everyday assistant tools: timers, lists, notes, maths and unit conversion,
volume. Each public function is a voice tool registered in LLMClient.TOOLS;
they return plain text for the LLM to phrase.
"""

import ast
import json
import logging
import math
import operator
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import settings

logger = logging.getLogger(__name__)


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s and not h:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " and ".join(parts) or "0 seconds"


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Ignoring unreadable {path}")
        return default


# ─── Timers ───────────────────────────────────────────────────────────────────
class TimerStore:
    """Countdown timers (kitchen-style, in memory). Numbered 1, 2, 3 like reminders."""

    def __init__(self, on_done=None, clock=time.monotonic):
        self._on_done = on_done or self._announce_done
        self._clock = clock
        self._lock = threading.Lock()
        self._timers = {}   # id -> {"label", "ends_at", "cancel": Event}

    @staticmethod
    def _announce_done(label: str, t_id: int | None = None):
        import AnnouncerModule
        if label:
            text = f"Your {label} timer is done!"
        else:
            # Numbered, so two unnamed timers are told apart
            text = f"Timer number {t_id} is done!" if t_id else "Your timer is done!"
        AnnouncerModule.announce(text, AnnouncerModule.NORMAL)

    def add(self, seconds: float, label: str = "") -> int:
        with self._lock:
            t_id = 1
            while t_id in self._timers:
                t_id += 1
            cancel = threading.Event()
            self._timers[t_id] = {"label": label, "ends_at": self._clock() + seconds, "cancel": cancel}
        threading.Thread(target=self._worker, args=(t_id, seconds, cancel), daemon=True).start()
        return t_id

    def _worker(self, t_id, seconds, cancel):
        if cancel.wait(timeout=seconds):
            return
        with self._lock:
            timer = self._timers.pop(t_id, None)
        if timer is not None:
            if self._on_done == self._announce_done:
                self._announce_done(timer["label"], t_id)
            else:
                self._on_done(timer["label"])

    def cancel(self, t_id: int) -> dict | None:
        with self._lock:
            timer = self._timers.pop(t_id, None)
        if timer:
            timer["cancel"].set()
        return timer

    def items(self) -> list:
        now = self._clock()
        with self._lock:
            return [(t_id, t["label"], max(t["ends_at"] - now, 0)) for t_id, t in sorted(self._timers.items())]

    def find(self, label: str) -> list:
        label = label.strip().lower()
        return [t for t in self.items() if label and label in t[1].lower()]


_timers = None


def get_timer_store() -> TimerStore:
    global _timers
    if _timers is None:
        _timers = TimerStore()
    return _timers


def set_timer(minutes=0, seconds=0, label=""):
    """Start a countdown timer."""
    try:
        total = float(minutes or 0) * 60 + float(seconds or 0)
    except (TypeError, ValueError):
        return "Error: minutes and seconds must be numbers."
    if total <= 0:
        return "Error: the timer needs a length, like 10 minutes."
    if total > 24 * 3600:
        return "Error: timers can be at most 24 hours. Use a reminder for longer."
    label = (label or "").strip()
    t_id = get_timer_store().add(total, label)
    name = f"'{label}' timer" if label else "Timer"
    return f"{name} number {t_id} started for {_duration(total)}."


def timer_status():
    """How long is left on each running timer."""
    items = get_timer_store().items()
    if not items:
        return "No timers are running."
    return "Running timers: " + "; ".join(
        f"number {t_id}{f' ({label})' if label else ''} has {_duration(left)} left" for t_id, label, left in items)


def cancel_timer(timer_id=None, label=""):
    """Cancel a timer by number or label (or the only timer if there's just one)."""
    store = get_timer_store()
    items = store.items()
    if not items:
        return "There are no timers to cancel."
    if timer_id is None and not label:
        if len(items) > 1:
            return "Error: more than one timer is running. " + timer_status() + ". Which one should I cancel?"
        timer_id = items[0][0]
    if timer_id is None:
        matches = store.find(label)
        if len(matches) != 1:
            return f"Error: {'no' if not matches else 'several'} timers match '{label}'. " + timer_status()
        timer_id = matches[0][0]
    try:
        timer = store.cancel(int(timer_id))
    except (TypeError, ValueError):
        timer = None
    if not timer:
        return f"Error: no timer number {timer_id}."
    label_text = f" ({timer['label']})" if timer["label"] else ""
    return f"Cancelled timer number {timer_id}{label_text}."


# ─── Lists (shopping, to-do, ...) ─────────────────────────────────────────────
class ListStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict:
        data = _read_json(self.path, {})
        return data if isinstance(data, dict) else {}

    def add(self, name: str, items: list) -> list:
        with self._lock:
            data = self._load()
            existing = data.setdefault(name, [])
            added = [i for i in items if i.lower() not in (e.lower() for e in existing)]
            existing.extend(added)
            _write_json(self.path, data)
            return added

    def remove(self, name: str, item: str) -> str | None:
        with self._lock:
            data = self._load()
            entries = data.get(name, [])
            needle = item.strip().lower()
            match = next((e for e in entries if e.lower() == needle), None) or \
                next((e for e in entries if needle and needle in e.lower()), None)
            if match is None:
                return None
            entries.remove(match)
            _write_json(self.path, data)
            return match

    def get(self, name: str) -> list:
        with self._lock:
            return list(self._load().get(name, []))

    def names(self) -> list:
        with self._lock:
            return [n for n, items in self._load().items() if items]

    def clear(self, name: str) -> int:
        with self._lock:
            data = self._load()
            count = len(data.get(name, []))
            data[name] = []
            _write_json(self.path, data)
            return count


def _list_name(list_name) -> str:
    name = (list_name or "shopping").strip().lower()
    name = re.sub(r"\s*list$", "", name)          # "shopping list" -> "shopping"
    name = {"todo": "to-do", "to do": "to-do", "grocery": "shopping", "groceries": "shopping"}.get(name, name)
    # A small model sometimes stuffs junk in here ('function requires confirm...',
    # '{"list_name": "shopping'). Real list names are a word or two.
    if not name or not re.fullmatch(r"[a-z][a-z0-9 '-]{0,24}", name) or len(name.split()) > 3:
        logger.info(f"Unusable list name {list_name!r}; using the shopping list")
        return "shopping"
    return name


def _split_items(items) -> list:
    if isinstance(items, list):
        parts = items
    else:
        parts = re.split(r",|\band\b", str(items or ""))
    return [p.strip(" .") for p in map(str, parts) if p.strip(" .")]


def get_list_store() -> ListStore:
    return ListStore(settings.LISTS_PATH)


def add_to_list(items, list_name="shopping"):
    """Add one or more comma-separated items to a list."""
    name = _list_name(list_name)
    wanted = _split_items(items)
    if not wanted:
        return "Error: nothing to add."
    added = get_list_store().add(name, wanted)
    skipped = [w for w in wanted if w not in added]
    msg = f"Added {', '.join(added)} to the {name} list." if added else ""
    if skipped:
        msg += f" Already on the list: {', '.join(skipped)}."
    return msg.strip()


def remove_from_list(item, list_name="shopping"):
    """Remove an item from a list."""
    name = _list_name(list_name)
    removed = get_list_store().remove(name, str(item or ""))
    if removed is None:
        return f"'{item}' isn't on the {name} list."
    return f"Removed {removed} from the {name} list."


def read_list(list_name="shopping"):
    """Read out a list."""
    name = _list_name(list_name)
    items = get_list_store().get(name)
    if not items:
        others = get_list_store().names()
        hint = f" Lists with items: {', '.join(others)}." if others else ""
        return f"The {name} list is empty.{hint}"
    return f"The {name} list has {len(items)} item{'s' if len(items) != 1 else ''}: {', '.join(items)}."


# list name -> time.monotonic() until which a "yes" will clear it. The two-step
# confirmation is enforced here, not left to the model (which was seen passing
# confirm=true on the very first request).
_pending_clear = {}
CONFIRM_WINDOW_S = 120


def clear_list(list_name="shopping", confirm=False):
    """Empty a whole list. Always asks first; a second call with confirm=true clears it."""
    name = _list_name(list_name)
    count = len(get_list_store().get(name))
    if count == 0:
        _pending_clear.pop(name, None)
        return f"The {name} list is already empty."
    now = time.monotonic()
    if confirm and _pending_clear.get(name, 0) > now:
        _pending_clear.pop(name, None)
        get_list_store().clear(name)
        return f"Done, I cleared the {name} list. {count} item{'s' if count != 1 else ''} removed."
    _pending_clear[name] = now + CONFIRM_WINDOW_S
    return f"Your {name} list has {count} item{'s' if count != 1 else ''}. Are you sure you want to clear it?"


# ─── Notes ("remember that...") ───────────────────────────────────────────────
_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {"the", "a", "an", "is", "are", "was", "in", "on", "at", "of", "to", "my", "i", "where", "what",
              "did", "do", "it", "that", "for", "and", "or", "me", "you", "remember", "put", "left", "keep"}


def _keywords(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed(text: str, kind: str) -> list | None:
    """Embedding via the LLM server (nomic needs a task prefix). None if unavailable."""
    try:
        import LLMClient
        if LLMClient.client is None:
            return None
        resp = LLMClient.client.embeddings.create(model=settings.EMBEDDING_MODEL, input=f"{kind}: {text}")
        return list(resp.data[0].embedding)
    except Exception as e:
        logger.info(f"Embeddings unavailable, using keyword search ({e})")
        return None


class NotesStore:
    def __init__(self, path, embed=_embed):
        self.path = Path(path)
        self._embed = embed
        self._lock = threading.Lock()

    def _load(self) -> list:
        data = _read_json(self.path, [])
        return data if isinstance(data, list) else []

    def add(self, text: str) -> int:
        vector = self._embed(text, "search_document")
        with self._lock:
            notes = self._load()
            n_id = max((n["id"] for n in notes), default=0) + 1
            notes.append({"id": n_id, "text": text, "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                          "embedding": vector})
            _write_json(self.path, notes)
            return n_id

    def delete(self, n_id: int) -> dict | None:
        with self._lock:
            notes = self._load()
            note = next((n for n in notes if n["id"] == n_id), None)
            if note:
                notes.remove(note)
                _write_json(self.path, notes)
            return note

    def all(self) -> list:
        with self._lock:
            return self._load()

    def search(self, query: str, limit: int = 3) -> list:
        notes = self.all()
        if not notes:
            return []
        q_words = _keywords(query)
        q_vec = self._embed(query, "search_query") if any(n.get("embedding") for n in notes) else None
        scored = []
        for n in notes:
            overlap = len(q_words & _keywords(n["text"])) / max(len(q_words), 1)
            sim = _cosine(q_vec, n["embedding"]) if q_vec and n.get("embedding") else 0.0
            score = max(sim, overlap) if q_vec else overlap
            # nomic similarities for unrelated text sit around 0.3-0.45
            if (q_vec and sim >= 0.55) or overlap > 0:
                scored.append((score, n))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [n for _, n in scored[:limit]]


def get_notes_store() -> NotesStore:
    return NotesStore(settings.NOTES_PATH)


def remember_note(text):
    """Save a fact or note to recall later."""
    text = str(text or "").strip()
    if not text:
        return "Error: there's nothing to remember."
    n_id = get_notes_store().add(text)
    return f"Got it, I'll remember that. It's note number {n_id}."


def recall_note(query):
    """Search saved notes."""
    matches = get_notes_store().search(str(query or ""))
    if not matches:
        # Facts learned about people count too ("where does Sarah work?")
        try:
            import PeopleMemory
            facts = PeopleMemory.get_people_memory().search(str(query or ""), k=2)
        except Exception:
            logger.exception("People memory search failed")
            facts = []
        if facts:
            return "You told me: " + " ".join(f["fact"].rstrip(".") + "." for f in facts)
        return f"I don't have any notes about {query}."
    if len(matches) == 1:
        return f"You told me: {matches[0]['text'].rstrip('.')}."
    return f"I found {len(matches)} notes. " + " ".join(
        f"Note {n['id']}: {n['text'].rstrip('.')}." for n in matches)


def delete_note(note_id):
    """Delete a saved note by number."""
    try:
        note = get_notes_store().delete(int(note_id))
    except (TypeError, ValueError):
        note = None
    return f"Deleted note {note_id}: {note['text']}" if note else f"Error: there is no note number {note_id}."


# ─── Calculator ───────────────────────────────────────────────────────────────
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max,
          "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log10, "ln": math.log}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
    raise ValueError("unsupported expression")


def calculate(expression):
    """Exact arithmetic, so the LLM never has to do maths in its head."""
    expr = str(expression or "")
    expr = re.sub(r"\s*(?:%|percent)\s*of\s*", "/100*", expr, flags=re.IGNORECASE)   # "17% of 240"
    expr = (expr.replace("×", "*").replace("÷", "/").replace("^", "**")
            .replace(" x ", " * ").replace("%", "/100"))
    if len(expr) > 200:
        return "Error: expression too long."
    try:
        value = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        return f"Error: I couldn't work out {expression}."
    if isinstance(value, float):
        value = round(value, 6)
        if value.is_integer():
            value = int(value)
    return f"{expression} is {value}."


# ─── Unit conversion ──────────────────────────────────────────────────────────
# unit -> (dimension, factor to the base unit)
_UNITS = {
    # length (metre)
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1.0), "km": ("length", 1000.0),
    "in": ("length", 0.0254), "ft": ("length", 0.3048), "yd": ("length", 0.9144), "mi": ("length", 1609.344),
    # mass (gram)
    "mg": ("mass", 0.001), "g": ("mass", 1.0), "kg": ("mass", 1000.0), "oz": ("mass", 28.349523125),
    "lb": ("mass", 453.59237), "st": ("mass", 6350.29318),
    # volume (millilitre) — metric cooking measures
    "ml": ("volume", 1.0), "l": ("volume", 1000.0), "tsp": ("volume", 5.0), "tbsp": ("volume", 15.0),
    "cup": ("volume", 250.0), "floz": ("volume", 29.5735), "pint": ("volume", 568.261), "gal": ("volume", 3785.41),
    # speed (metre/second)
    "m/s": ("speed", 1.0), "km/h": ("speed", 1 / 3.6), "mph": ("speed", 0.44704), "knot": ("speed", 0.514444),
    # time (second)
    "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0), "day": ("time", 86400.0), "week": ("time", 604800.0),
}
_ALIASES = {
    "millimetre": "mm", "millimeter": "mm", "centimetre": "cm", "centimeter": "cm", "metre": "m", "meter": "m",
    "kilometre": "km", "kilometer": "km", "inch": "in", "inches": "in", "foot": "ft", "feet": "ft", "yard": "yd",
    "mile": "mi", "milligram": "mg", "gram": "g", "kilogram": "kg", "kilo": "kg", "ounce": "oz", "pound": "lb",
    "lbs": "lb", "stone": "st", "millilitre": "ml", "milliliter": "ml", "litre": "l", "liter": "l",
    "teaspoon": "tsp", "tablespoon": "tbsp", "fluid ounce": "floz", "fl oz": "floz", "gallon": "gal",
    "kph": "km/h", "kmh": "km/h", "kilometres per hour": "km/h", "kilometers per hour": "km/h",
    "miles per hour": "mph", "metres per second": "m/s", "knots": "knot", "second": "s", "sec": "s",
    "minute": "min", "hour": "h", "hr": "h", "days": "day", "weeks": "week",
    "celsius": "c", "centigrade": "c", "°c": "c", "fahrenheit": "f", "°f": "f", "kelvin": "k",
}
_TEMPS = {"c", "f", "k"}


def _norm_unit(unit: str) -> str:
    u = str(unit or "").strip().lower().replace("degrees ", "").replace("degree ", "")
    if u in _UNITS or u in _TEMPS:
        return u
    if u in _ALIASES:
        return _ALIASES[u]
    if u.endswith("es") and u[:-2] in _ALIASES:
        return _ALIASES[u[:-2]]
    if u.endswith("s") and (u[:-1] in _ALIASES or u[:-1] in _UNITS):
        return _ALIASES.get(u[:-1], u[:-1])
    return u


def convert_units(value, from_unit, to_unit):
    """Convert between units (length, weight, volume, speed, time, temperature)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"Error: '{value}' is not a number."
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    if f in _TEMPS or t in _TEMPS:
        if not (f in _TEMPS and t in _TEMPS):
            return f"Error: I can't convert {from_unit} to {to_unit}."
        celsius = {"c": v, "f": (v - 32) * 5 / 9, "k": v - 273.15}[f]
        result = {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[t]
    else:
        if f not in _UNITS or t not in _UNITS:
            unknown = from_unit if f not in _UNITS else to_unit
            return f"Error: I don't know the unit {unknown}."
        if _UNITS[f][0] != _UNITS[t][0]:
            return f"Error: I can't convert {from_unit} to {to_unit}."
        result = v * _UNITS[f][1] / _UNITS[t][1]
    pretty = f"{result:.4g}" if abs(result) < 1e6 else f"{result:,.0f}"
    return f"{v:g} {from_unit} is {pretty} {to_unit}."


# ─── Volume ───────────────────────────────────────────────────────────────────
def set_volume(level):
    """Set the speaking volume from 0 (silent) to 10 (loudest)."""
    try:
        level = float(level)
    except (TypeError, ValueError):
        return "Error: volume must be a number from 0 to 10."
    import TTSModule
    applied = TTSModule.set_volume(min(max(level, 0), 10) / 10)
    return f"Volume set to {round(applied * 10)} out of 10."
