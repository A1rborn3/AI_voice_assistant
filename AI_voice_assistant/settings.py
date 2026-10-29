"""
Central configuration.
======================
Loads the repo-root .env exactly once and exposes every tunable as a typed
module constant. Everything has a sensible default, so only API keys are
required in .env (see .env.example for the full list).
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = REPO_ROOT / "config"

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    logger.warning("python-dotenv not installed; reading settings from the process environment only.")


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        logger.warning(f"Invalid number for {name}; using default {default}")
        return default


def _get_path(name: str, default: Path) -> Path:
    """Paths in .env may be absolute or relative to the repo root."""
    value = os.getenv(name)
    if not value or not value.strip():
        return default
    path = Path(value.strip())
    return path if path.is_absolute() else REPO_ROOT / path


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ─── API keys ─────────────────────────────────────────────────────────────────
WEATHER_API_KEY      = os.getenv("WEATHER_API_KEY")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

# ─── LLM (any OpenAI-compatible server: LM Studio, llama.cpp, Ollama...) ──────
LLM_BASE_URL    = _get_str("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL       = _get_str("LLM_MODEL", "qwen/qwen3-4b-2507")   # Qwen3-4B-Instruct-2507
LLM_API_KEY     = _get_str("LLM_API_KEY", "local-placeholder")
# 0.0 = deterministic, 0.7 = natural-sounding speech, 1.0+ = creative but erratic
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.7)
# Recent exchanges sent with each request. Small models call tools less
# reliably as the context grows; raise this for bigger models.
LLM_HISTORY_TURNS = int(_get_float("LLM_HISTORY_TURNS", 4))

# ─── Files ────────────────────────────────────────────────────────────────────
MEMORY_PATH         = CONFIG_DIR / "memory.json"
MEMORY_EXAMPLE_PATH = CONFIG_DIR / "memory.example.json"
REMINDERS_PATH      = CONFIG_DIR / "reminders.json"

# ─── Wake word (Porcupine) ────────────────────────────────────────────────────
# 0.0-1.0: higher catches more wake words but gives more false triggers
WAKE_WORD_SENSITIVITY = min(max(_get_float("WAKE_WORD_SENSITIVITY", 0.5), 0.0), 1.0)
# Built-in keyword used when the custom .ppn can't be loaded on this platform
WAKE_WORD_FALLBACK    = _get_str("WAKE_WORD_FALLBACK", "jarvis")
CUSTOM_WAKE_WORD_PATH = PACKAGE_DIR / "STT_Model" / "PicoVoice" / "Samantha_en_raspberry-pi_v3_0_0.ppn"

# ─── Barge-in (interrupting the assistant with the wake word) ────────────────
# Without echo cancellation the mic hears the speaker, so the assistant can
# trigger its own wake word. Use a headset or OS/hardware AEC if you can.
BARGE_IN_ENABLED = _get_bool("BARGE_IN_ENABLED", True)
# Ignore wake-word hits for this long after playback starts (speaker bleed)
BARGE_IN_GRACE_S = _get_float("BARGE_IN_GRACE_S", 0.4)
# Pause after playback before opening the mic, so the room echo dies down
POST_TTS_DELAY_S = _get_float("POST_TTS_DELAY_S", 0.3)

# ─── Speech-to-text ───────────────────────────────────────────────────────────
# "auto" uses faster-whisper when installed, else Vosk
STT_ENGINE      = _get_str("STT_ENGINE", "auto").lower()
WHISPER_MODEL   = _get_str("WHISPER_MODEL", "base.en")   # tiny.en / base.en / small.en
WHISPER_DEVICE  = _get_str("WHISPER_DEVICE", "cpu")       # cpu / cuda
WHISPER_COMPUTE = _get_str("WHISPER_COMPUTE", "int8")     # int8 on CPU, float16 on GPU
# Words the recogniser should expect; nudges Whisper away from "sit an alarm"
WHISPER_PROMPT  = _get_str(
    "WHISPER_PROMPT",
    "Samantha, set an alarm, set a reminder, delete reminder, weather, forecast, stove.",
)
VOSK_MODEL_PATH = _get_path("VOSK_MODEL_PATH", PACKAGE_DIR / "STT_Model" / "vosk-model-small-en-us-0.15")

# ─── Voice activity detection / endpointing ──────────────────────────────────
VAD_THRESHOLD        = _get_float("VAD_THRESHOLD", 0.5)   # silero speech probability
VAD_END_SILENCE_S    = _get_float("VAD_END_SILENCE_S", 0.8)  # silence that ends a command
VAD_MIN_SPEECH_S     = _get_float("VAD_MIN_SPEECH_S", 0.25)  # shorter blips are ignored
VAD_PRE_ROLL_S       = _get_float("VAD_PRE_ROLL_S", 0.3)  # audio kept from before speech starts
COMMAND_START_TIMEOUT_S = _get_float("COMMAND_START_TIMEOUT_S", 7.0)
COMMAND_MAX_S           = _get_float("COMMAND_MAX_S", 15.0)

# ─── Text-to-speech ───────────────────────────────────────────────────────────
TTS_VOICE_PATH = _get_path("TTS_VOICE_PATH", PACKAGE_DIR / "voices" / "en_GB-cori-medium.onnx")
TTS_VOLUME     = min(max(_get_float("TTS_VOLUME", 1.0), 0.0), 1.0)   # 0.0-1.0
# Non-urgent speech is quieter during these hours ("HH:MM-HH:MM", may wrap midnight; empty = off).
# Hazard alerts always play at full volume.
QUIET_HOURS    = os.getenv("QUIET_HOURS", "22:00-07:00").strip()   # an empty value really disables it
QUIET_VOLUME   = min(max(_get_float("QUIET_VOLUME", 0.4), 0.0), 1.0)

# ─── Vision / hazard monitoring ───────────────────────────────────────────────
VISION_DIR      = REPO_ROOT / "Vision_module"
# Off by default so machines without a camera don't raise "camera offline" alerts
CAMERA_ENABLED  = _get_bool("CAMERA_ENABLED", False)
# Webcam index ("0"), stream/video URL ("rtsp://...") or a folder of images (for testing)
CAMERA_SOURCE   = _get_str("CAMERA_SOURCE", "0")
HAZARD_CHECK_INTERVAL_S  = _get_float("HAZARD_CHECK_INTERVAL_S", 5.0)
VISION_CONFIDENCE        = _get_float("VISION_CONFIDENCE", 0.5)
# Consecutive frames needed before the stove counts as switched on/off (filters flicker)
STOVE_CONFIRM_FRAMES     = int(_get_float("STOVE_CONFIRM_FRAMES", 3))
# Alert when the stove is on and nobody has been seen for this long
STOVE_UNATTENDED_MIN     = _get_float("STOVE_UNATTENDED_MIN", 10)
# Alert when the stove has been on this long, even if someone is around
STOVE_MAX_ON_MIN         = _get_float("STOVE_MAX_ON_MIN", 90)
# While a hazard continues, repeat the alert this often
HAZARD_REPEAT_MIN        = _get_float("HAZARD_REPEAT_MIN", 5)
# Announce "camera offline" after this long without a frame
CAMERA_OFFLINE_ALERT_S   = _get_float("CAMERA_OFFLINE_ALERT_S", 60)
SAVE_SNAPSHOTS           = _get_bool("SAVE_SNAPSHOTS", True)   # only for warnings/alerts
MAX_SNAPSHOTS            = int(_get_float("MAX_SNAPSHOTS", 200))
EVENTS_DB_PATH           = CONFIG_DIR / "events.db"
SNAPSHOT_DIR             = CONFIG_DIR / "snapshots"

# ─── Assistant data files ─────────────────────────────────────────────────────
LISTS_PATH         = CONFIG_DIR / "lists.json"
NOTES_PATH         = CONFIG_DIR / "notes.json"
WEATHER_CACHE_PATH = CONFIG_DIR / "weather_cache.json"
WEATHER_CACHE_MAX_AGE_H = _get_float("WEATHER_CACHE_MAX_AGE_H", 6)
# LM Studio embedding model for searching notes and people facts (falls back to keyword search)
EMBEDDING_MODEL    = _get_str("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

# ─── People memory (RAG) ──────────────────────────────────────────────────────
PEOPLE_DB_PATH        = CONFIG_DIR / "people.db"
MEMORY_FACTS_K        = int(_get_float("MEMORY_FACTS_K", 4))       # relevant facts added to each request
# Below this a fact isn't "relevant". Measured with nomic-embed-text: related facts
# scored 0.64-0.77, unrelated ones up to 0.56 ("what's the weather" vs "Mum likes gardening")
MEMORY_MIN_SIMILARITY = _get_float("MEMORY_MIN_SIMILARITY", 0.6)
MEMORY_DUPLICATE_SIMILARITY = _get_float("MEMORY_DUPLICATE_SIMILARITY", 0.9)  # treat as the same fact
FACT_EXTRACTION_EVERY = int(_get_float("FACT_EXTRACTION_EVERY", 4))  # learn facts every N exchanges
