import requests
import json
import os
import uuid
import threading
import logging
from datetime import datetime

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# In-memory store of active reminders: {id: {time, context, thread, cancel_event}}
active_reminders = {}


# ─── Reminders / Alarms ───────────────────────────────────────────────────────

def create_reminder(day, month, year, hour, minute, context):
    """Set a timed reminder/alarm. Alarms and reminders are the same thing."""
    try:
        target_dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
    except (ValueError, TypeError) as e:
        return f"Error: Invalid date/time values ({e})."

    now = datetime.now()
    if target_dt <= now:
        return f"Error: {target_dt.strftime('%Y-%m-%d %H:%M')} is in the past. Please give a future time."

    cancel_event = threading.Event()
    reminder_id = str(uuid.uuid4())[:4]

    def _worker(r_id, ev):
        while not ev.is_set():
            if datetime.now() >= target_dt:
                _alert_user(context)
                active_reminders.pop(r_id, None)
                break
            # Sleep in small increments so we can be interrupted quickly
            diff = (target_dt - datetime.now()).total_seconds()
            ev.wait(timeout=min(diff, 30))

    thread = threading.Thread(target=_worker, args=(reminder_id, cancel_event), daemon=True)
    active_reminders[reminder_id] = {
        "time": target_dt.strftime('%Y-%m-%d %H:%M'),
        "context": context,
        "thread": thread,
        "cancel_event": cancel_event,
    }
    thread.start()
    return f"Reminder/Alarm set for {target_dt.strftime('%Y-%m-%d %H:%M')} with ID: {reminder_id}"


def list_reminders():
    """Return a formatted list of all active reminders/alarms."""
    if not active_reminders:
        return "You currently have no active reminders or alarms."
    lines = ["Active reminders/alarms:"]
    for r_id, info in active_reminders.items():
        lines.append(f"  - ID: {r_id} | Time: {info['time']} | Purpose: {info['context']}")
    return "\n".join(lines)


def delete_reminder(reminder_id):
    """Cancel and remove an active reminder/alarm by its ID."""
    if reminder_id in active_reminders:
        active_reminders[reminder_id]['cancel_event'].set()
        active_reminders.pop(reminder_id, None)
        return f"Reminder/Alarm {reminder_id} has been canceled and deleted."
    return f"Error: No reminder or alarm found with ID '{reminder_id}'."


def _alert_user(context: str):
    """
    Called by the background thread when a reminder fires.
    Lazy-imports AI_voice_assistant to speak the alert via TTS.
    """
    try:
        import AI_voice_assistant
        import LLMClient
        from datetime import datetime as _dt

        memory = AI_voice_assistant.get_memory()
        now_str = _dt.now().strftime("%Y-%m-%d %H:%M")

        user_prof = memory.get_user_profile()
        user_personality = user_prof.get("personality", "") if isinstance(user_prof, dict) else ""

        # Generate a spoken alert using the conversational LLM
        alert_prompt = (
            f"An alarm or reminder just went off. Context: \"{context}\". "
            "Alert the user naturally in one short, friendly sentence."
        )
        response = LLMClient.ask_conversational(
            alert_prompt, [], now_str, user_personality
        )

        import TTSModule
        TTSModule.speak(response)
    except Exception as e:
        logger.error(f"alert_user failed: {e}")


# ─── Weather ──────────────────────────────────────────────────────────────────

def get_weather_data(forecast: bool = False):
    """Fetch weather data from WeatherAPI. Set forecast=True for 3-day view."""
    days = 3 if forecast else 1

    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Error: WEATHER_API_KEY not found in environment."

    url = f"https://api.weatherapi.com/v1/forecast.json?key={api_key}&q=auto:ip&days={days}"
    try:
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        return f"Error fetching weather: {e}"

    loc     = data["location"]
    current = data["current"]
    f_days  = data["forecast"]["forecastday"]

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

    result = json.dumps(output, indent=2)
    logger.debug(f"Weather result: {result}")
    return result


def end_conversation():
    """Immediately stop the current conversation loop and return to waiting for the wake word."""
    return "CONVERSATION_ENDED"