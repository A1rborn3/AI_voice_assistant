![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

# AI Voice Assistant

## Description
"Samantha" is a local-first home voice assistant that **watches the kitchen for safety hazards**:
- A camera checks whether the stove is on and whether anyone is around, and Samantha speaks up if it's been left unattended.
- She also handles everyday tasks: timers, reminders, lists, notes, maths, unit conversions and the weather.

Speech recognition, speech synthesis, the language model and object detection all run on your own machine. The only cloud services are Picovoice (the wake word key check) and WeatherAPI.

## Features

### Kitchen safety (hazard monitoring)
- A **camera service** analyses a frame every few seconds with two YOLOv8 models: a stove on/off detector and a person detector.
- **Stove rules:**
  - A **critical** alert if the stove is on and nobody has been in the kitchen for `STOVE_UNATTENDED_MIN` (default 10 minutes).
  - A **critical** alert if the stove has been on for `STOVE_MAX_ON_MIN` (default 90 minutes) even with people around.
  - Alerts repeat every `HAZARD_REPEAT_MIN` until resolved; Samantha says "The stove has been switched off" when it is.
  - Flicker is filtered out: the stove only counts as on/off after `STOVE_CONFIRM_FRAMES` agreeing frames.
- **Critical alerts jump the queue:** they interrupt whatever Samantha is saying and always play at full volume, even during quiet hours.
- **Event log:** SQLite in `config/events.db`, plus a snapshot JPG for each warning/alert (`config/snapshots/`, capped at `MAX_SNAPSHOTS`).
- **Watchdogs:** Samantha says so out loud if the camera stops delivering frames or the microphone keeps failing.
- **Voice:** "Is the stove on?", "Have a look at the stove", "What happened in the kitchen today?", "Snooze the stove alerts for 15 minutes" (60 minutes at most).

### Assistant tools
| Ask | Tool |
|---|---|
| "Set a pasta timer for 10 minutes" / "how long is left?" / "cancel the timer" | Timers (several at once, named) |
| "Remind me in 20 minutes to check the oven" / "wake me at 6:45 every weekday" / "snooze" | Reminders: saved to disk, repeat daily/weekdays/weekly, spoken-friendly numbers, missed reminders announced at startup |
| "Add milk and eggs to the shopping list" / "what's on it?" / "clear it" | Lists (shopping, to-do, any name); clearing always asks "are you sure?" first |
| "Remember the spare key is in the blue pot" / "where's the spare key?" | Notes, searched by meaning (embeddings) with keyword fallback |
| "What's 17% of 240?" / "how many grams in 3 pounds?" | Exact calculator and unit converter (the LLM never does the maths) |
| "What's the weather like?" / "forecast for the next few days" | WeatherAPI, with a 6-hour cache if the service is down |
| "Turn the volume down to 4" | Volume; non-urgent speech is quieter during `QUIET_HOURS` |
| "What do you know about Sarah?" / "forget that I like jazz" | People memory (see below) |

### People memory (small RAG)
Samantha remembers people without pasting a whole profile into every prompt:
- **Learning:** every `FACT_EXTRACTION_EVERY` exchanges (default 4), the model pulls out lasting facts *the user said*, such as "Sarah is allergic to peanuts" or "Ethan has a cat called Milo". They're stored one per row in `config/people.db`, each with an embedding.
- **Checks before storing:**
  - Requests like alarms and timers are rejected.
  - Facts whose key words never came from the user (only from the assistant) are skipped.
  - Near-duplicates are merged.
  - A fact that changes an older one ("Sarah moved to Wellington") replaces it.
- **Retrieval:** each request gets only the `MEMORY_FACTS_K` (default 4) most relevant facts by meaning, plus a few facts about anyone it names. "Can Sarah have peanut butter?" brings up her allergy; "set a timer" brings up nothing.
- **Without the embedding model,** it falls back to keyword matching, and missing embeddings are filled in later.
- **The old JSON profile is converted** into facts automatically on first start.
- **Who is speaking:** it currently assumes the main user (`user.name` in `memory.json`) and learns about other people by name. Recognising different people's voices would be the next step.

### Voice pipeline
- **Wake word:** Porcupine. **End of speech:** [silero-vad](https://github.com/snakers4/silero-vad). **Speech-to-text:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with Vosk as the fallback. **Speech output:** [Piper](https://github.com/rhasspy/piper), with times, units and symbols rewritten so they read naturally.
- **Barge-in:** say the wake word to interrupt Samantha.
- **Local LLM** through any OpenAI-compatible server (LM Studio, llama.cpp, Ollama). It uses native tool calling and only offers the tools relevant to each request.
- **One announcement queue:** hazard alerts, reminders and timers never talk over each other or over a reply.

### Built for a small local model
The default model is **Qwen3-4B-Instruct-2507** (`qwen/qwen3-4b-2507` in LM Studio, ~2.5 GB at Q4). On a 24-request live benchmark it scored 94% versus 92% for `llama-3.2-3b-instruct`, and unlike llama it never performed an action nobody asked for. The model only **chooses** tools; it never reports what they did. Everything below came out of live testing:
- **Results are spoken in fixed wording:** every tool returns a finished sentence that is said as-is, and errors are spoken in fixed wording too. A small model can't misreport a result ("3.361 kg" for 1361 g) or turn a failure into "done".
- **Invented answers are caught:** a reply that claims an action or talks about the stove, alarms or timers without a tool having run is retried with a forced tool call.
- **Destructive tools are gated:** deleting, cancelling, clearing and ending the conversation need matching words from the user. The model once called `delete_note` for "where is the spare key?".
- **The user's words beat the model's guesses:** "every weekday", "in 20 minutes" and "5 seconds" override what the model filled in.
- **Text-form tool calls are understood:** JSON, `tool(arg=…)` and `tool, arg=…` written in the reply count as calls.
- **Context is kept short:** only the last `LLM_HISTORY_TURNS` exchanges are sent, because small models call tools less reliably as the context grows.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/A1rborn3/AI_voice_assistant.git
   cd AI_voice_assistant
   ```
2. Create a virtual environment with **Python 3.10 or newer** (developed on 3.14) and install the dependencies:
   ```bash
   python -m venv Samantha.venv
   Samantha.venv/Scripts/python -m pip install -r requirements-dev.txt
   ```
   (Use `requirements.txt` alone if you don't need the tests; on Linux, use `Samantha.venv/bin/python`.) The first voice-mode run downloads the Whisper model (~145 MB for `base.en`) into the Hugging Face cache.
3. Start a local LLM server. In LM Studio, load **Qwen3-4B-Instruct-2507** (`qwen/qwen3-4b-2507`; use the *Instruct* build, not *Thinking*) with full GPU offload if you have a GPU. `llama-3.2-3b-instruct` also works if you set `LLM_MODEL`. Also load `text-embedding-nomic-embed-text-v1.5`, which people memory and notes use to search by meaning. Then start the server on port 1234.

### Raspberry Pi (runs as a service)
```bash
bash deploy/install_pi.sh
```
- **First run:** installs system packages and the venv, creates `.env` from the template, then stops so you can fill in the keys.
- **Second run:** installs the `samantha` systemd service, which starts on boot in voice mode and restarts after a crash.
- **Status and logs:** `sudo systemctl status samantha` and `journalctl -u samantha -f`.

## Configuration

1. **Environment:** copy `.env.example` to `.env` in the repo root and fill in the keys. Every other setting has a default and is documented in `.env.example` (all are loaded by `AI_voice_assistant/settings.py`). To turn on hazard monitoring:
   ```env
   CAMERA_ENABLED=true
   CAMERA_SOURCE=0                       # webcam index, rtsp://... URL or video file
   ```
   To try the whole pipeline without a camera, point it at photos: `CAMERA_SOURCE=Vision_module/TestPic`.

2. **Memory and persona:**
   - `config/memory.json` is created from `config/memory.example.json` on the first run.
   - Edit its `assistant` block to change Samantha's name, personality and purpose.
   - Set `user.name` to your name: it's who Samantha assumes she's talking to, and what learned facts about you are filed under.
   - Facts about people are learned automatically (see *People memory*).

3. **Wake word:**
   - The custom "Samantha" `.ppn` file is built for Raspberry Pi; everywhere else the built-in `WAKE_WORD_FALLBACK` keyword (default **"Jarvis"**) is used.
   - Tune false triggers with `WAKE_WORD_SENSITIVITY`.

4. **Echo:** without echo cancellation, the mic hears the speaker.
   - Wake-word hits in the first `BARGE_IN_GRACE_S` of playback are ignored, and barge-in is skipped when the reply itself contains the wake word.
   - A headset or a speakerphone with hardware AEC works best.

5. **Privacy:**
   - All personal data stays in `config/` and is gitignored: memory, conversation, reminders, lists, notes, the event log and snapshots.
   - Snapshots are only saved for warnings and alerts; set `SAVE_SNAPSHOTS=false` to keep none.

## Usage
```bash
python AI_voice_assistant/AI_voice_assistant.py              # menu
python AI_voice_assistant/AI_voice_assistant.py --mode voice # or --mode text
```
- **Voice mode:**
  - Say the wake word, then your request.
  - Samantha keeps listening without the wake word when her reply ends in a question.
  - "Stop" or "goodbye" ends the conversation; "exit" quits the app.
- **Diagnostics:** detailed logs go to `AI_voice_assistant/application.log`; follow them live with `debug_monitor.bat`.

## Testing
```bash
python -m pytest
```
- **Unit tests:** the tests in `tests/` need no microphone, speaker, camera, LLM server or network, and also run in GitHub Actions on every push (`.github/workflows/tests.yml`).
- **What they cover:** hazard rules, the monitor and event log, the announcer, reminders, timers, lists, notes, maths, LLM tool handling (including the real malformed outputs seen from small models), memory, speech endpointing and the YOLO models on the test photos.
- **Real hardware and services:** to check them on the target machine, run:
  ```bash
  python AI_voice_assistant/hardware_check.py
  ```

## Project Structure
```
AI_voice_assistant/
├── AI_voice_assistant/          # Assistant source
│   ├── AI_voice_assistant.py    # Entry point: text/voice loops, starts the services
│   ├── settings.py              # All configuration (.env loaded once here)
│   ├── LLMClient.py             # Prompt, tool registry, tool-call handling and guards
│   ├── HazardModule.py          # Camera monitor, stove rules, event log, hazard tools
│   ├── AnnouncerModule.py       # Priority queue for everything said unprompted
│   ├── AssistantTools.py        # Timers, lists, notes, calculator, units, volume
│   ├── APIManagerMain.py        # Reminders, weather
│   ├── MemoryModule.py          # Persona + conversation history
│   ├── PeopleMemory.py          # Facts about people: learning, retrieval, tools
│   ├── STTModule.py             # Wake word, VAD endpointing, Whisper/Vosk
│   ├── TTSModule.py             # Piper speech, volume, speech normalisation
│   └── hardware_check.py        # Manual hardware/network smoke test
├── Vision_module/
│   ├── detectors.py             # Stove + person detection per frame
│   ├── stove_detector.py        # Stove model wrapper / CLI
│   ├── YoloModel/               # Stove.pt, defaultModel.pt (COCO)
│   └── TestPic/                 # Test photos
├── config/memory.example.json   # Template for memory.json (persona lives here)
├── deploy/                      # Raspberry Pi install script + systemd unit
├── tests/                       # pytest unit tests
└── .env.example
```

## Roadmap
- **More hazards:** smoke/fire, falls (pose model), an open door, water on the floor. Each is a new rule class next to `StoveRule`.
- **Better stove model:**
  - More training data (lighting, angles, stove types), plus a held-out test set.
  - Per-burner tracking.
- Multi-room satellites and speaker recognition.
- Streaming replies (start speaking while the LLM is still generating).
- Phone notifications and smart-home control.
