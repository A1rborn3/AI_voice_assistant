import requests
import json
from datetime import datetime
import threading
import time
import logging

logger = logging.getLogger(__name__)


import uuid
active_reminders = {}

def create_reminder(day, month, year, hour, minute, context):
    # Ensure inputs are integers (as per tool definition) or handle strings if passed as such
    try:
        day = int(day)
        month = int(month)
        year = int(year)
        hour = int(hour)
        minute = int(minute)
        target_dt = datetime(year, month, day, hour, minute)
        
    except ValueError:
        return "Error: Invalid date/time format."

    now = datetime.now()
    if target_dt < now:
        return f"Error: The time {target_dt.strftime('%Y-%m-%d %H:%M')} is in the past. Please set a future time."
    
    def reminder_worker(r_id, cancel_event):
        while not cancel_event.is_set():
            now = datetime.now()
            if now >= target_dt:
                alert_LLM(context)
                # Cleanup once finished
                if r_id in active_reminders:
                    del active_reminders[r_id]
                break
            
            # Calculate seconds until target
            diff = (target_dt - now).total_seconds()
            
            # Sleep until the time, but check at least every 60 seconds
            # Use cancel_event.wait so it can be interrupted immediately
            sleep_time = min(diff, 60)
            if sleep_time > 0:
                cancel_event.wait(sleep_time)
            
    cancel_event = threading.Event()
    reminder_id = str(uuid.uuid4())[:4] # Create a short user-friendly ID
    
    thread = threading.Thread(target=reminder_worker, args=(reminder_id, cancel_event), daemon=True)
    
    active_reminders[reminder_id] = {
        "time": target_dt.strftime('%Y-%m-%d %H:%M'),
        "context": context,
        "thread": thread,
        "cancel_event": cancel_event
    }
    
    thread.start()
    return f"Reminder/Alarm set for {target_dt.strftime('%Y-%m-%d %H:%M')} with ID: {reminder_id}"

def list_reminders():
    """Returns a formatted list of all active reminders."""
    if not active_reminders:
        return "You currently have no active reminders or alarms."
    
    output = "Active reminders/alarms:\n"
    for r_id, r_info in active_reminders.items():
        output += f"- ID: {r_id} | Time: {r_info['time']} | Purpose: {r_info['context']}\n"
    return output

def delete_reminder(reminder_id):
    """Cancels and deletes a reminder by its ID."""
    if reminder_id in active_reminders:
        active_reminders[reminder_id]['cancel_event'].set()
        del active_reminders[reminder_id]
        return f"Success: Reminder/Alarm {reminder_id} has been canceled and deleted."
    else:
        return f"Error: Could not find a reminder or alarm with ID '{reminder_id}'."

    
def alert_LLM(context):
    """
    Interact with the main AI_Voice_assistant's LLM and its memory.
    """
    # Lazy import to avoid circular dependency with APIManagerMain
    import AI_voice_assistant
    import MemoryModule

    # Initialize memory
    # Assuming the config path is relative to the execution directory
    memory = AI_voice_assistant.get_memory()
    
    # Use the shared process_interaction function
    response = AI_voice_assistant.process_interaction(context, memory, "reminder")
    
    return response


def get_weather_data(forecast):
    DAYS = 1

    if forecast:
        DAYS = 3
    else:
        DAYS = 1
    
    import os
    from dotenv import load_dotenv
    
    # Load .env relative to this file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    
    Weather_API_KEY = os.getenv("WEATHER_API_KEY")
    if not Weather_API_KEY:
        return "Error: WEATHER_API_KEY not found in environment." 
    LOCATION = "auto:ip"      # city, coords, or auto:ip



    url = f"https://api.weatherapi.com/v1/forecast.json?key={Weather_API_KEY}&q={LOCATION}&days={DAYS}"
    data = requests.get(url).json()

    location = data["location"]
    current = data["current"]
    forecast_days = data["forecast"]["forecastday"]

    output = {
        "current": {
            "name": location.get("name"),
            "country": location.get("country"),
            "temperature_c": current.get("temp_c"),
            "feels_like_c": current.get("feelslike_c"),
            "humidity": current.get("humidity"),
            "precip_mm": current.get("precip_mm"),
            "wind_kph": current.get("wind_kph"),
            "pressure_mb": current.get("pressure_mb"),
            "visibility_km": current.get("vis_km"),
            "last_updated": current.get("last_updated")
        },
        "forecast": []
    }

    for day in forecast_days:
        date_str = day.get("date")  # like "2025-12-04"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_week = date_obj.strftime("%A")  # -> "Thursday"
        day_info = {
            "date": day.get("date"),
            "day_of_week": day_of_week,
            "sunrise": day["astro"].get("sunrise"),
            "sunset": day["astro"].get("sunset"),
            "day_summary": {
                "max_temp_c": day["day"].get("maxtemp_c"),
                "min_temp_c": day["day"].get("mintemp_c"),
                "avg_temp_c": day["day"].get("avgtemp_c"),
                "max_wind_kph": day["day"].get("maxwind_kph"),
                "total_precip_mm": day["day"].get("totalprecip_mm"),
                "avg_humidity": day["day"].get("avghumidity"),
                "condition": day["day"]["condition"]["text"]
            },
            "hourly": [
                {
                    "time": h.get("time"),
                    "temp_c": h.get("temp_c"),
                    "feels_like_c": h.get("feelslike_c"),
                    "humidity": h.get("humidity"),
                    "precip_mm": h.get("precip_mm"),
                    "wind_kph": h.get("wind_kph"),
                    "pressure_mb": h.get("pressure_mb"),
                    "visibility_km": h.get("vis_km"),
                    "condition": h["condition"]["text"]
                }
                for h in day["hour"]
            ]
        }
        output["forecast"].append(day_info)

    logger.debug(json.dumps(output, indent=2))
    return json.dumps(output, indent=2)