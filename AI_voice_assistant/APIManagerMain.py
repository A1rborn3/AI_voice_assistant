import requests
import json
import re
import threading
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import settings

logger = logging.getLogger(__name__)

CONVERSATION_ENDED = "CONVERSATION_ENDED"
TIME_FORMAT = "%Y-%m-%d %H:%M"
REPEATS = ("none", "daily", "weekdays", "weekly")
SNOOZE_WINDOW = timedelta(minutes=30)   # "snooze" only applies to a reminder that fired this recently


def spoken_when(when: datetime, now: datetime | None = None) -> str:
    """'today at 22:49', 'tomorrow at 06:45', 'Thursday 29 October at 07:00'."""
    now = now or datetime.now()
    days = (when.date() - now.date()).days
    if days == 0:
        day = "today"
    elif days == 1:
        day = "tomorrow"
    elif 1 < days < 7:
        day = when.strftime("%A")
    else:
        day = f"{when:%A} {when.day} {when:%B}"
    return f"{day} at {when:%H:%M}"


REPEAT_WORDS = {"daily": "every day", "weekdays": "every weekday", "weekly": "every week"}
_REPEAT_SYNONYMS = {
    "": "none", "none": "none", "no": "none", "once": "none", "never": "none", "false": "none", "null": "none",
    "daily": "daily", "day": "daily", "every day": "daily", "everyday": "daily", "each day": "daily",
    "weekdays": "weekdays", "weekday": "weekdays", "every weekday": "weekdays", "workdays": "weekdays",
    "monday to friday": "weekdays", "weekly": "weekly", "week": "weekly", "every week": "weekly",
}


def normalise_repeat(value) -> str:
    """Map the many ways a model says it ('every weekday', 'day', None) onto REPEATS; unknown stays as-is."""
    text = str(value or "").strip().lower()
    return _REPEAT_SYNONYMS.get(text, text)


def next_occurrence(target: datetime, repeat: str, after: datetime) -> datetime:
    """The first repeat of `target` strictly after `after`."""
    step = timedelta(days=7 if repeat == "weekly" else 1)
    nxt = target
    while nxt <= after or (repeat == "weekdays" and nxt.weekday() >= 5):
        nxt += step
    return nxt


# ─── Reminders / Alarms ───────────────────────────────────────────────────────

class ReminderStore:
    """
    Reminders persisted to a JSON file so they survive restarts.
    IDs are small numbers (the lowest free one: 1, 2, 3...) so they are easy
    to say aloud. Each pending reminder has a waiting worker thread.
    Repeating reminders (daily / weekdays / weekly) reschedule themselves
    under the same number after firing.
    """

    def __init__(self, path, alert=None):
        self.path = Path(path)
        # alert(context, missed_at=None) is called when a reminder fires
        self._alert = alert or _alert_user
        self._lock = threading.RLock()
        self._reminders = {}  # id -> {"time": "YYYY-mm-dd HH:MM", "context": str, "repeat": str}
        self._cancel = {}     # id -> threading.Event
        self.last_fired = None  # (datetime fired, context) — for snooze

    # ── persistence ──
    def load(self):
        """Read saved reminders, schedule future ones and announce any missed while offline."""
        try:
            with self.path.open("r", encoding="utf-8") as f:
                saved = json.load(f)
        except FileNotFoundError:
            return
        except json.JSONDecodeError:
            logger.warning(f"Ignoring corrupted reminders file {self.path}")
            return

        now = datetime.now()
        missed = []
        with self._lock:
            for item in saved if isinstance(saved, list) else []:
                try:
                    r_id = int(item["id"])
                    target = datetime.strptime(item["time"], TIME_FORMAT)
                    context = str(item.get("context", ""))
                    repeat = item.get("repeat", "none")
                    repeat = repeat if repeat in REPEATS else "none"
                except (KeyError, TypeError, ValueError):
                    logger.warning(f"Skipping malformed reminder: {item}")
                    continue
                if target <= now:
                    missed.append((context, item["time"]))
                    if repeat != "none":   # announce the miss once, keep the schedule going
                        self._schedule(r_id, next_occurrence(target, repeat, now), context, repeat)
                else:
                    self._schedule(r_id, target, context, repeat)
            self._save()

        for context, when in missed:
            logger.info(f"Reminder missed while offline: {when} {context}")
            threading.Thread(target=self._alert, args=(context, when), daemon=True).start()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        items = [{"id": r_id, **info} for r_id, info in sorted(self._reminders.items())]
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
        tmp.replace(self.path)

    # ── scheduling ──
    def _next_id(self) -> int:
        r_id = 1
        while r_id in self._reminders:
            r_id += 1
        return r_id

    def _schedule(self, r_id: int, target: datetime, context: str, repeat: str = "none"):
        cancel_event = threading.Event()
        self._reminders[r_id] = {"time": target.strftime(TIME_FORMAT), "context": context, "repeat": repeat}
        self._cancel[r_id] = cancel_event
        threading.Thread(target=self._worker, args=(r_id, target, context, repeat, cancel_event), daemon=True).start()

    def _worker(self, r_id, target, context, repeat, cancel_event):
        while not cancel_event.is_set():
            remaining = (target - datetime.now()).total_seconds()
            if remaining <= 0:
                with self._lock:
                    # Deleted at the last moment? Then don't fire.
                    if cancel_event.is_set() or self._cancel.get(r_id) is not cancel_event:
                        return
                    self._reminders.pop(r_id, None)
                    self._cancel.pop(r_id, None)
                    if repeat != "none":
                        self._schedule(r_id, next_occurrence(target, repeat, datetime.now()), context, repeat)
                    self.last_fired = (datetime.now(), context)
                    self._save()
                self._alert(context)
                return
            # Re-check the wall clock at least every 30s (handles sleep/clock changes)
            cancel_event.wait(timeout=min(remaining, 30))

    # ── public API ──
    def add(self, target: datetime, context: str, repeat: str = "none") -> int:
        with self._lock:
            r_id = self._next_id()
            self._schedule(r_id, target, context, repeat)
            self._save()
            return r_id

    def delete(self, r_id: int) -> bool:
        with self._lock:
            if r_id not in self._reminders:
                return False
            self._cancel.pop(r_id).set()
            self._reminders.pop(r_id)
            self._save()
            return True

    def items(self) -> list:
        with self._lock:
            return sorted(self._reminders.items())

    def stop(self):
        """Stop all worker threads (reminders stay saved)."""
        with self._lock:
            for ev in self._cancel.values():
                ev.set()


_store = None
_store_lock = threading.Lock()


def get_reminder_store() -> ReminderStore:
    """Shared store; created (without loading) on first use."""
    global _store
    with _store_lock:
        if _store is None:
            _store = ReminderStore(settings.REMINDERS_PATH)
        return _store


def start_reminders():
    """Load saved reminders and start their timers. Call once at app startup."""
    get_reminder_store().load()


_TIME_RE = re.compile(r"^\s*(\d{1,2})(?:[:.](\d{2}))?\s*([ap])?\.?\s*m?\.?\s*$", re.IGNORECASE)


def parse_time(text: str) -> tuple[int, int]:
    """'08:30', '8.30', '8:30 pm', '7am', '19:05' -> (hour, minute) in 24-hour time."""
    match = _TIME_RE.match(str(text))
    if not match:
        raise ValueError(f"I can't understand the time {text}")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if meridiem:
        if not 1 <= hour <= 12:
            raise ValueError(f"'{text}' is not a valid 12-hour time")
        hour = hour % 12 + (12 if meridiem == "p" else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"'{text}' is not a valid time")
    return hour, minute


def parse_date(text: str | None, now: datetime) -> date | None:
    """'2026-10-27', 'today', 'tomorrow' or empty (None) -> a date."""
    value = (text or "").strip().lower()
    if not value:
        return None
    if value == "today":
        return now.date()
    if value == "tomorrow":
        return (now + timedelta(days=1)).date()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"I can't understand the date {text}") from None


def create_reminder(context, time=None, date=None, repeat=None, in_minutes=None):
    """
    Set a timed reminder/alarm. Alarms and reminders are the same thing.
    Either `in_minutes` (relative: "in 20 minutes") or `time` (HH:MM 24h, or
    with am/pm) plus optional `date` (YYYY-MM-DD, 'today', 'tomorrow').
    Without a date the next occurrence of that time is used. `repeat` is
    none / daily / weekdays / weekly.
    """
    now = datetime.now()
    repeat = normalise_repeat(repeat)
    # Models sometimes put the repeat in the date field ("date": "daily")
    date_word = normalise_repeat(date) if date else ""
    if date and date_word in REPEATS:
        if date_word != "none" and repeat == "none":
            repeat = date_word
        date = None
    if repeat not in REPEATS:
        return "Error: I can only repeat reminders every day, every weekday or every week."

    if in_minutes is not None and str(in_minutes).strip() not in ("", "0"):
        try:
            minutes = float(in_minutes)
        except (TypeError, ValueError):
            return f"Error: '{in_minutes}' is not a number of minutes."
        if not 0 < minutes <= 60 * 24 * 7:
            return "Error: a reminder in minutes can be from 1 minute up to 7 days away."
        target_dt = now + timedelta(minutes=minutes)
    elif time:
        try:
            hour, minute = parse_time(time)
            day = parse_date(date, now)
        except ValueError as e:
            return f"Error: {e}."
        if day is None:
            target_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target_dt <= now:
                target_dt += timedelta(days=1)
        else:
            target_dt = datetime(day.year, day.month, day.day, hour, minute)
    else:
        return "Error: I need a time for that reminder, like 7:30, or how many minutes from now."

    if target_dt <= now and repeat != "none":
        target_dt = next_occurrence(target_dt, repeat, now)   # repeating: start at the next one
    if target_dt <= now:
        return f"Error: {target_dt:%H:%M} on {target_dt:%A} is already in the past."
    if repeat == "weekdays" and target_dt.weekday() >= 5:
        target_dt = next_occurrence(target_dt, repeat, target_dt)

    r_id = get_reminder_store().add(target_dt, context, repeat)
    repeats = "" if repeat == "none" else f", repeating {REPEAT_WORDS[repeat]}"
    return f"Reminder number {r_id} set for {spoken_when(target_dt, now)}{repeats}, to {context.rstrip('.')}."


def snooze_reminder(minutes=10):
    """Snooze the reminder that just went off by a number of minutes."""
    store = get_reminder_store()
    fired = store.last_fired
    if not fired or datetime.now() - fired[0] > SNOOZE_WINDOW:
        return "Error: no reminder has gone off recently, so there's nothing to snooze."
    try:
        minutes = min(max(float(minutes), 1), 120)
    except (TypeError, ValueError):
        minutes = 10
    target = datetime.now() + timedelta(minutes=minutes)
    r_id = store.add(target, fired[1])
    store.last_fired = None
    return f"Snoozed. I'll remind you again at {target:%H:%M}, to {fired[1].rstrip('.')}."


def list_reminders(search: str = ""):
    """Return a formatted list of active reminders/alarms, optionally filtered by a word."""
    items = get_reminder_store().items()
    if not items:
        return "You currently have no active reminders or alarms."
    if search and search.strip():
        needle = search.strip().lower()
        items = [(r_id, info) for r_id, info in items if needle in info["context"].lower()]
        if not items:
            return f"No active reminders or alarms match '{search.strip()}'."
    now = datetime.now()
    parts = []
    for r_id, info in items:
        when = spoken_when(datetime.strptime(info["time"], TIME_FORMAT), now)
        repeat = info.get("repeat", "none")
        repeats = "" if repeat == "none" else f", repeating {REPEAT_WORDS.get(repeat, repeat)}"
        parts.append(f"number {r_id}, {info['context'].rstrip('.')}, {when}{repeats}")
    count = f"{len(items)} reminder{'s' if len(items) != 1 else ''}"
    return f"You have {count}: " + "; ".join(parts) + "."


def delete_reminder(reminder_id):
    """Cancel and remove an active reminder/alarm by its number."""
    try:
        r_id = int(reminder_id)
    except (TypeError, ValueError):
        return f"Error: {reminder_id} isn't a reminder number."
    if get_reminder_store().delete(r_id):
        return f"Reminder number {r_id} has been canceled and deleted."
    return f"Error: No reminder or alarm with number {r_id}."


def _alert_user(context: str, missed_at: str | None = None):
    """
    Called by a reminder worker when it fires (or at startup for missed ones).
    Lazy-imports the shared modules (never AI_voice_assistant itself: when it
    runs as __main__, importing it would load a second copy of the app).
    """
    try:
        import AnnouncerModule
        import LLMClient
        import MemoryModule

        memory = MemoryModule.get_memory()
        now_str = datetime.now().strftime(TIME_FORMAT)

        if missed_at:
            alert_prompt = (
                f"A reminder for {missed_at} was missed while you were switched off. "
                f"Context: \"{context}\". Tell the user in one short, friendly sentence."
            )
            fallback = f"While I was off, you missed a reminder: {context}."
        else:
            alert_prompt = (
                f"An alarm or reminder just went off. Context: \"{context}\". "
                "Alert the user naturally in one short, friendly sentence."
            )
            fallback = f"Reminder: {context}."

        response = LLMClient.ask_conversational(
            alert_prompt, [], now_str,
            memory.get_user_profile_text(), memory.get_assistant_profile()
        ) or fallback  # still alert if the LLM is unreachable

        # The announcer waits for a quiet moment, so this never talks over a reply
        AnnouncerModule.announce(response, AnnouncerModule.NORMAL)
    except Exception:
        logger.exception("alert_user failed")


# ─── Weather ──────────────────────────────────────────────────────────────────

def get_weather_data(forecast: bool = False):
    """
    Weather from WeatherAPI (forecast=True for the 3-day view). If the service
    can't be reached, fall back to the last good result if it's recent enough.
    """
    result = _fetch_weather(forecast)
    cache_path = Path(settings.WEATHER_CACHE_PATH)
    if not result.startswith("Error"):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"saved_at": datetime.now().strftime(TIME_FORMAT),
                                              "forecast": bool(forecast), "result": result}), encoding="utf-8")
        except OSError:
            logger.exception("Could not write weather cache")
        return result

    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        saved_at = datetime.strptime(cached["saved_at"], TIME_FORMAT)
    except (OSError, ValueError, KeyError, TypeError):
        return result
    age_h = (datetime.now() - saved_at).total_seconds() / 3600
    if age_h > settings.WEATHER_CACHE_MAX_AGE_H:
        return result
    logger.warning(f"Weather fetch failed ({result}); using cache from {cached['saved_at']}")
    return f"I can't reach the weather service, so this is from {saved_at:%H:%M}. {cached['result']}"


def _fetch_weather(forecast: bool = False):
    days = 3 if forecast else 1

    api_key = settings.WEATHER_API_KEY
    if not api_key:
        return "Error: the weather service isn't set up yet (WEATHER_API_KEY is missing)."

    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {"key": api_key, "q": "auto:ip", "days": days}
    try:
        data = requests.get(url, params=params, timeout=10).json()
    except Exception as e:
        return f"Error fetching weather: {e}"

    # WeatherAPI reports bad keys, quota limits, etc. as {"error": {...}}
    if not isinstance(data, dict) or "error" in data:
        message = data.get("error", {}).get("message", "unknown error") if isinstance(data, dict) else "unexpected response"
        logger.error(f"WeatherAPI error: {data}")
        return f"Error fetching weather: {message}"

    try:
        loc     = data["location"]
        current = data["current"]
        f_days  = data["forecast"]["forecastday"]
    except (KeyError, TypeError):
        logger.error(f"Unexpected WeatherAPI response: {data}")
        return "Error fetching weather: unexpected response from the weather service."

    output = {
        "location": f"{loc.get('name')}, {loc.get('country')}",
        "current": {
            "temperature_c":  current.get("temp_c"),
            "feels_like_c":   current.get("feelslike_c"),
            "humidity":       current.get("humidity"),
            "precip_mm":      current.get("precip_mm"),
            "wind_kph":       current.get("wind_kph"),
            "condition":      current["condition"]["text"],
            "last_updated":   current.get("last_updated"),
        },
        "forecast": []
    }

    for day in f_days:
        date_obj = datetime.strptime(day["date"], "%Y-%m-%d")
        output["forecast"].append({
            "date":       day["date"],
            "day_of_week": date_obj.strftime("%A"),
            "sunrise":    day["astro"].get("sunrise"),
            "sunset":     day["astro"].get("sunset"),
            "max_temp_c": day["day"].get("maxtemp_c"),
            "min_temp_c": day["day"].get("mintemp_c"),
            "condition":  day["day"]["condition"]["text"],
            "rain_chance": day["day"].get("daily_chance_of_rain"),
        })

    logger.debug(f"Weather data: {json.dumps(output)}")
    return speak_weather(output, forecast)


def _deg(value) -> str:
    return "unknown" if value is None else f"{round(float(value))} degrees"


def speak_weather(data: dict, forecast: bool = False) -> str:
    """
    Spoken weather summary built from the data (not by the LLM, which was seen
    answering a weather question with "there's no direct response").
    """
    place = data["location"].split(",")[0]
    cur = data["current"]
    parts = [f"Right now in {place} it's {cur['condition'].lower()} and {_deg(cur['temperature_c'])}"]
    feels = cur.get("feels_like_c")
    if feels is not None and cur.get("temperature_c") is not None and abs(feels - cur["temperature_c"]) >= 2:
        parts[0] += f", feeling like {_deg(feels)}"
    if cur.get("wind_kph") is not None and cur["wind_kph"] >= 20:
        parts[0] += f", with wind at {round(cur['wind_kph'])} km/h"
    parts[0] += "."

    days = data.get("forecast") or []
    if days:
        today = days[0]
        rain = f", {today['rain_chance']}% chance of rain" if today.get("rain_chance") else ""
        parts.append(f"Today: {today['condition'].lower()}, high of {_deg(today['max_temp_c'])}, "
                     f"low of {_deg(today['min_temp_c'])}{rain}.")
    if forecast:
        for day in days[1:]:
            rain = f", {day['rain_chance']}% chance of rain" if day.get("rain_chance") else ""
            parts.append(f"{day['day_of_week']}: {day['condition'].lower()}, "
                         f"{_deg(day['min_temp_c'])} to {round(float(day['max_temp_c']))}{rain}.")
    return " ".join(parts)


def end_conversation():
    """Immediately stop the current conversation loop and return to waiting for the wake word."""
    return CONVERSATION_ENDED
