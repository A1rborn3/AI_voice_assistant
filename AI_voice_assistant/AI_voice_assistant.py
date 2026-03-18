"""
AI Voice Assistant — Two-Stage Router Architecture
====================================================
Stage 1  : LLMClient.route_intent()   → "TOOL" | "CHAT"
Stage 2a : LLMClient.execute_tool()   → tool result string  (TOOL path)
Stage 2b : LLMClient.confirm_tool_result() → spoken confirmation  (TOOL path)
Stage 2b': LLMClient.ask_conversational()  → spoken reply  (CHAT path)
"""

import sys
import os
import threading
import time
import logging
import re
from datetime import datetime

# ─── Ensure the module directory is on the path ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import LLMClient
import TTSModule
import MemoryModule

# Configure logging
try:
    from logging_config import setup_logging
    setup_logging()
except ImportError:
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# ─── Globals ──────────────────────────────────────────────────────────────────
memory: MemoryModule.Memory | None = None
_memory_lock = threading.Lock()


def get_memory() -> MemoryModule.Memory:
    """Return the shared memory instance (initialised on first call)."""
    global memory
    if memory is None:
        memory_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'memory.json')
        memory = MemoryModule.Memory(memory_path)
    return memory


# ─── Text cleaning for TTS ────────────────────────────────────────────────────
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
    # Collapse whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


# ─── Core Interaction ─────────────────────────────────────────────────────────
def process_interaction(user_text: str, mem: MemoryModule.Memory) -> str | None:
    """
    Run the two-stage router for a single user utterance.
    Returns the spoken reply string (already passed to TTS), or None on error.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build clean profile strings for the conversational prompt
    user_prof = mem.get_user_profile()
    user_personality = (
        user_prof.get("personality", "No user details yet.")
        if isinstance(user_prof, dict) else "No user details yet."
    )
    assistant_prof = mem.get_assistant_profile()
    assistant_context = (
        f"Name: {assistant_prof.get('name', 'Samantha')} | "
        f"Personality: {assistant_prof.get('personality', 'Helpful')} | "
        f"Purpose: {assistant_prof.get('purpose', 'Personal assistant')}"
        if isinstance(assistant_prof, dict) else "Samantha"
    )

    # Conversation history for the CHAT prompt
    history_messages = []
    for u, a in mem.conversation.to_list():
        history_messages.append({"role": "user",      "content": u})
        history_messages.append({"role": "assistant", "content": a})

    # ── Stage 1: Route ──────────────────────────────────────────────────────
    intent = LLMClient.route_intent(user_text)

    # ── Stage 2: Execute ────────────────────────────────────────────────────
    if intent == "TOOL":
        # 2a — Extract parameters and run the tool
        tool_result = LLMClient.execute_tool(user_text, now_str)
        logger.info(f"[Main] Tool result: {tool_result}")
        
        if tool_result == "CONVERSATION_ENDED":
            return "EXIT_LOOP"

        # 2b — Wrap the result in a friendly Samantha reply
        spoken = LLMClient.confirm_tool_result(
            tool_result, user_text, history_messages,
            now_str, user_personality, assistant_context
        )
    else:  # CHAT
        spoken = LLMClient.ask_conversational(
            user_text, history_messages,
            now_str, user_personality, assistant_context
        )

    spoken_clean = clean_llm_response(spoken)
    print(f"Samantha: {spoken_clean}")

    TTSModule.speak(spoken_clean)

    # Save to memory
    if spoken_clean:
        mem.add_message_pair(user_text, spoken_clean)
        mem.save()

        # Trigger background personality update (non-blocking)
        threading.Thread(
            target=mem.update_user_personality,
            daemon=True
        ).start()

    return spoken_clean


# ─── Wake-word interruption helper ───────────────────────────────────────────
def wait_for_tts_or_wake_word(stt) -> bool:
    """Wait for TTS to finish; return True if interrupted by wake word."""
    interrupted = [False]

    def _listen():
        if stt.listen_for_wake_word(until_event_cleared=TTSModule.is_speaking):
            interrupted[0] = True

    t = threading.Thread(target=_listen, daemon=True)
    t.start()
    while TTSModule.is_speaking.is_set() and not interrupted[0]:
        time.sleep(0.1)
    return interrupted[0]


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mem = get_memory()
    TTSModule.warmup()

    print("1. Text Mode")
    print("2. Voice Mode")
    mode = input("Select mode (1/2): ").strip()

    if mode == "1":
        # ── Text Mode ──────────────────────────────────────────────────────
        print("Type your message. Type 'exit' to quit.")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break
            if user_input:
                process_interaction(user_input, mem)

    elif mode == "2":
        # ── Voice Mode ─────────────────────────────────────────────────────
        import STTModule
        stt = STTModule.STFTModule()

        while True:
            print("Waiting for wake word...")
            if stt.listen_for_wake_word():

                while True:  # Continuous conversation loop
                    print("Listening for command...")
                    text = stt.listen_for_command()
                    if not text:
                        break
                    print(f"You said: {text}")
                    if "exit" in text.lower():
                        sys.exit(0)
                        
                    # Manual Stop/Cancel check
                    # Only trigger if it's a short standalone command (1-2 words)
                    # This prevents "cancel my alarm" from breaking the loop.
                    stop_keywords = ["stop", "cancel", "nevermind", "never mind", "goodbye", "good bye", "exit"]
                    words = text.lower().strip().split()
                    if len(words) <= 2 and any(kw in (" ".join(words)) for kw in stop_keywords):
                        print("Conversation ended by user.")
                        break

                    response = process_interaction(text, mem)
                    if response == "EXIT_LOOP":
                        break

                    # If Samantha asked a question (ends with '?'), keep mic open
                    if response and response.rstrip().endswith("?"):
                        if wait_for_tts_or_wake_word(stt):
                            print("Interrupted!")
                            continue
                        time.sleep(0.5)
                        continue  # Skip wake word, listen immediately
                    else:
                        if wait_for_tts_or_wake_word(stt):
                            print("Interrupted!")
                            continue
                        time.sleep(0.5)
                        break  # Return to wake word detection
    else:
        print("Invalid selection.")
