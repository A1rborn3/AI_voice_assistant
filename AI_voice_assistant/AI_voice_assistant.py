"""
AI Voice Assistant
==================
Voice mode : wake word → VAD-endpointed command → speech-to-text →
             LLMClient.respond() (one LLM call, native tool calling) → TTS
Text mode  : the same pipeline, typed in the console.

Background services started here:
  * AnnouncerModule — one queue for everything said unprompted
  * HazardModule    — camera-based stove monitoring (if CAMERA_ENABLED)
  * reminders       — persisted in config/reminders.json

Usage: python AI_voice_assistant.py [--mode text|voice]
"""

import argparse
import sys
import os
import threading
import time
import logging
import re
from datetime import datetime

# ─── Ensure the module directory is on the path ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings
import LLMClient
import TTSModule
import MemoryModule
import APIManagerMain
import AnnouncerModule
import HazardModule

# Configure logging
try:
    from logging_config import setup_logging
    setup_logging()
except ImportError:
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

QUIT_COMMANDS = {"exit", "exit program", "shut down", "shutdown"}
STOP_KEYWORDS = ("stop", "cancel", "nevermind", "never mind", "goodbye", "good bye", "exit")
# Consecutive microphone failures before the assistant says so out loud
MIC_FAILURE_ALERT = 3
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


# ─── Globals ──────────────────────────────────────────────────────────────────
def get_memory() -> MemoryModule.Memory:
    """Return the shared memory instance (initialised on first call)."""
    return MemoryModule.get_memory()


# ─── Text helpers ─────────────────────────────────────────────────────────────
def clean_llm_response(text: str) -> str:
    """Strip markdown, JSON blocks, and symbol noise from a TTS-bound string."""
    if not text:
        return ""
    # Drop code / markdown blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
    # Drop JSON blobs
    text = re.sub(r'\{[^}]*\}', '', text, flags=re.DOTALL)
    # Drop markdown list bullets and headers
    text = re.sub(r'^\s*[-*#]+\s*', '', text, flags=re.MULTILINE)
    # Drop bold/italic markers
    text = text.replace("**", "").replace("__", "")
    # Drop emoji (the model adds them despite the prompt; Piper can't say them)
    text = _EMOJI_RE.sub("", text)
    # Collapse whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def normalise_command(text: str) -> str:
    """Lowercase and strip punctuation (Whisper adds it: 'Stop.' -> 'stop')."""
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def classify_command(text: str) -> str | None:
    """
    'quit' for a bare quit command, 'stop' for a short standalone stop word
    (1-2 words, so "cancel my alarm" still reaches the LLM), else None.
    """
    words = normalise_command(text)
    if words in QUIT_COMMANDS:
        return "quit"
    if len(words.split()) <= 2 and any(kw in words for kw in STOP_KEYWORDS):
        return "stop"
    return None


# ─── Core Interaction ─────────────────────────────────────────────────────────
def process_interaction(user_text: str, mem: MemoryModule.Memory) -> str | None:
    """
    Send one user utterance through the LLM and speak the reply.
    Returns the spoken text, "EXIT_LOOP" if the user ended the conversation,
    or None if there was nothing to say.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    reply = LLMClient.respond(
        user_text,
        mem.get_history_messages(),
        now_str,
        mem.memory_context(user_text),   # only the remembered facts relevant to this request
        mem.get_assistant_profile(),
        speaker=mem.get_speaker_name(),
    )
    if reply.tools_used:
        logger.info(f"[Main] Tools used: {reply.tools_used}")
    if reply.ended:
        return "EXIT_LOOP"

    spoken_clean = clean_llm_response(reply.text)
    if not spoken_clean:
        return None

    name = mem.get_assistant_profile().get("name", "Samantha")
    print(f"{name}: {spoken_clean}")
    TTSModule.speak(spoken_clean)

    # Save to memory (add_message_pair also schedules learning facts about
    # people in the background every few turns). Errors aren't kept as history.
    if not reply.error:
        mem.add_message_pair(user_text, spoken_clean)
        mem.save()

    return spoken_clean


# ─── Wake-word interruption helper ───────────────────────────────────────────
def wait_for_tts_or_wake_word(stt, spoken_text: str = "") -> bool:
    """Wait for TTS to finish; return True if the user interrupted with the wake word."""
    # Barge-in off, or the reply itself contains the wake word: just wait,
    # otherwise the assistant would interrupt itself through the speaker.
    wake_word = (stt.wake_word or "").lower()
    if not settings.BARGE_IN_ENABLED or (wake_word and wake_word in spoken_text.lower()):
        while TTSModule.is_speaking.is_set():
            time.sleep(0.1)
        return False

    interrupted = threading.Event()

    def _listen():
        if stt.listen_for_wake_word(until_event_cleared=TTSModule.is_speaking):
            interrupted.set()

    t = threading.Thread(target=_listen, daemon=True)
    t.start()
    while TTSModule.is_speaking.is_set() and not interrupted.is_set():
        time.sleep(0.1)
    # Let the listener release the microphone before anyone else opens it
    t.join(timeout=2)
    return interrupted.is_set()


# ─── Modes ────────────────────────────────────────────────────────────────────
def run_text_mode(mem: MemoryModule.Memory, announcer: AnnouncerModule.Announcer):
    print("Type your message. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input:
            announcer.conversation_active.set()
            try:
                if process_interaction(user_input, mem) == "EXIT_LOOP":
                    print("(Conversation ended.)")
            finally:
                announcer.conversation_active.clear()


def run_voice_mode(mem: MemoryModule.Memory, announcer: AnnouncerModule.Announcer):
    import STTModule
    stt = STTModule.STFTModule()
    if not stt.ensure_wake_word_engine():
        print("Wake word engine failed to start; check PICOVOICE_ACCESS_KEY in .env.")
        announcer.say("My wake word engine didn't start, so I can't hear you. Hazard alerts still work.",
                      AnnouncerModule.NORMAL)
        # Keep running so hazard monitoring and reminders continue
        while True:
            time.sleep(60)
    print(f"Wake word: '{stt.wake_word}'")

    mic_failures = 0
    try:
        while True:
            print("Waiting for wake word...")
            if not stt.listen_for_wake_word():
                # Watchdog: a dead microphone shouldn't fail silently
                mic_failures += 1
                if mic_failures == MIC_FAILURE_ALERT:
                    announcer.say("I'm having trouble with the microphone, so I may not hear you.",
                                  AnnouncerModule.NORMAL)
                time.sleep(min(2 ** mic_failures, 60))  # back off if the mic keeps failing
                continue
            if mic_failures >= MIC_FAILURE_ALERT:
                logger.info("Microphone working again")
            mic_failures = 0

            # Non-urgent announcements wait until this conversation is over
            announcer.conversation_active.set()
            try:
                converse(stt, mem)
            finally:
                announcer.conversation_active.clear()
    finally:
        stt.close()


def converse(stt, mem: MemoryModule.Memory):
    """One conversation after the wake word: keeps going while the assistant asks questions."""
    while True:
        text = stt.listen_for_command()
        if not text:
            return
        print(f"You said: {text}")

        command = classify_command(text)
        if command == "quit":
            raise SystemExit(0)
        if command == "stop":
            print("Conversation ended by user.")
            return

        response = process_interaction(text, mem)
        if response == "EXIT_LOOP":
            return

        if wait_for_tts_or_wake_word(stt, response or ""):
            print("Interrupted!")
            continue
        time.sleep(settings.POST_TTS_DELAY_S)

        # If the assistant asked a question, keep the mic open
        if response and response.rstrip().endswith("?"):
            continue
        return  # back to wake word detection


def start_services() -> AnnouncerModule.Announcer:
    """Background services shared by both modes."""
    announcer = AnnouncerModule.get_announcer()
    APIManagerMain.start_reminders()
    monitor = HazardModule.start_hazard_monitoring()
    if monitor:
        print(f"Hazard monitoring: on (camera '{settings.CAMERA_SOURCE}')")
    else:
        print("Hazard monitoring: off (set CAMERA_ENABLED=true in .env)")
    return announcer


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Samantha voice assistant")
    parser.add_argument("--mode", choices=("text", "voice"), help="skip the menu (used by the system service)")
    args = parser.parse_args()

    mem = get_memory()
    TTSModule.warmup()
    announcer = start_services()

    mode = args.mode
    if mode is None:
        print("1. Text Mode")
        print("2. Voice Mode")
        mode = {"1": "text", "2": "voice"}.get(input("Select mode (1/2): ").strip())

    try:
        if mode == "text":
            run_text_mode(mem, announcer)
        elif mode == "voice":
            run_voice_mode(mem, announcer)
        else:
            print("Invalid selection.")
    finally:
        monitor = HazardModule.get_hazard_monitor()
        if monitor:
            monitor.stop()
        announcer.stop()
