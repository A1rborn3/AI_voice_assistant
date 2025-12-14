# AI Voice Assistant

## Description
This project is a sophisticated AI Voice Assistant named "Samantha", designed to provide natural and interactive assistance. It integrates advanced Large Language Models (LLM), Text-to-Speech (TTS), and Speech-to-Text (STT) models to understand and respond to user queries in real-time. The assistant features a persistent memory system to adapt to user preferences and maintains context over conversations.

## Features
- Voice Interaction: voice-to-voice communication using Porcupine (Wake Word), Vosk (STT), and Piper (TTS).
- Intelligent Conversation: Powered by Groq's high-speed LLM inference (Llama 3 models) for natural and context-aware responses.
- Persistent Memory: Remembers user details, preferences, and past conversations to personalize interactions.
- Dual Modes: 
  - Text Mode: Type to interact via the console.
  - Voice Mode: Hands-free interaction using wake word detection.
- Tools & Capabilities:
  - Weather Updates: Real-time weather forecasts using WeatherAPI.
  - Smart Reminders: Set and receive voice reminders.

## Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/A1rborn3/AI_voice_assistant.git>
   cd AI_voice_assistant
   ```

2. Install system dependencies (if required for Audio/TTS libraries):
   - Ensure you have Python 3.9 installed.
   - You may need to install drivers for `sounddevice` / `PortAudio` depending on your OS (e.g., `sudo apt-get install python3-pyaudio` on Linux).

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Environment Setup:
   Create a `.env` file in the root directory (based on `.env.example`) and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   WEATHER_API_KEY=your_weather_api_key
   PICOVOICE_ACCESS_KEY=your_picovoice_access_key
   ```

2. Memory & Config:
   - Configuration files are located in the `config/` directory.
   - `memory.json` stores user and assistant profiles.
   - `logging_config.py` manages log settings.

3. Wake word setup:
   - The default wake word is "Jarvis" for the PicoVoice library.
   - You can change the wake word by importing the .ppn model from the PicoVoice website and adding it to STT_Model/PicoVoice/.
   - custom models are tied to the access key and therfore cannot be shared. I suggest using Samantha as the wake word to allign with the name of the assistant.

## Usage Example

Run the main application:
```bash
python AI_voice_assistant/AI_voice_assistant.py
```

**Select a Mode:**
1. **Text Mode**: The assistant acts as a chatbot in the terminal.
   ```
   You: What's the weather in London?
   Assistant: It's currently 15°C and cloudy in London.
   ```

2. **Voice Mode**: The assistant listens for the wake word (default: "Jarvis", "Porcupine" or custom depending on keyword file).
   - Say the Wake Word.
   - Speak your command: "Set a reminder for 5 PM to call John."
   - The assistant will confirm and execute the action.

## Project Structure

```
AI_voice_assistant/
├── AI_voice_assistant/        # Main Source Code
│   ├── AI_voice_assistant.py  # Entry point & main loop
│   ├── APIManagerMain.py      # Tools (Weather, Reminders)
│   ├── LLMClient.py           # Groq API handling
│   ├── MemoryModule.py        # Memory & Profile management
│   ├── STTModule.py           # Speech-to-Text logic
│   ├── TTSModule.py           # Text-to-Speech logic
│   ├── logging_config.py      # Logging setup
│   └── voices/                # TTS Voice models
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── config/                    # Configuration files
└──.env.example            # Environment variables example
```

## Future Improvements
- Add more complex tool integrations (Calendar, Email).
   - Notes to dictate ideas
   - Alarm / Timer
   - Device integration (ie, smart home devices, ESP32 boards, etc)
   - Phone app for usability
- Add live search capabilities with firecrawl.
- Improve wake word sensitivity controls.
- Create a GUI interface for easier configuration.
- Add settings and setup to change name, personal info, voice, volume, speed, Assistant personality, ect..
- Dockerize the application for easier deployment.
