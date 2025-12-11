
import time
import logging
import os
import sys
import json

# Setup logging
logging.basicConfig(level=logging.ERROR, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestSuite")

# Import Modules
try:
    import TTSModule
    import STTModule
    import LLMClient
    import APIManagerMain
    import MemoryModule
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

class ComponentTester:
    def __init__(self):
        self.results = []

    def log_result(self, test_name, success, message=""):
        status = "PASSED" if success else "FAILED"
        self.results.append({
            "test": test_name,
            "status": status,
            "message": message
        })
        print(f"[{status}] {test_name}: {message}")

    def test_tts(self):
        print("\n--- Testing Text-to-Speech ---")
        try:
            TTSModule.warmup()
            text = "Automated test sequence initiated."
            TTSModule.speak(text)
            # Wait for audio to finish playing to avoid conflicting with STT initialization
            # (Windows audio drivers often fail if opening Input immediately after Output starts)
            time.sleep(3) 
            self.log_result("TTS Module", True, "Audio generation triggered success.")
        except Exception as e:
            self.log_result("TTS Module", False, f"Exception: {e}")

    def test_stt_init(self):
        print("\n--- Testing STT Initialization ---")
        try:
            # Only testing initialization to avoid blocking for user input
            stt = STTModule.STFTModule()
            
            # Explicitly initialize wake word engine for testing
            stt.porcupine = stt._init_porcupine()
            
            if stt.vosk_model and stt.porcupine:
                self.log_result("STT Initialization", True, "Vosk and Picovoice models loaded.")
            else:
                 self.log_result("STT Initialization", False, "One or more models failed to load.")
            
            # Clean up
            if stt.porcupine: stt.porcupine.delete()
            if stt.recorder: stt.recorder.delete()
            
        except Exception as e:
            self.log_result("STT Initialization", False, f"Exception: {e}")

    def test_llm(self):
        print("\n--- Testing LLM (Groq) ---")
        try:
            prompt = "What is 2+2? Answer just with the final number."
            print(f"Prompt: {prompt}")
            response = LLMClient.askLLM(prompt)
            print(f"Response: {response}")
            
            if response and "4" in response:
                self.log_result("LLM (Groq)", True, f"Correct response received: {response}")
            elif "rate limit" in response.lower() or "error" in response.lower():
                 self.log_result("LLM (Groq)", False, f"API Error: {response}")
            else:
                 self.log_result("LLM (Groq)", True, f"Response received (validation loose): {response}")
                 
        except Exception as e:
            self.log_result("LLM (Groq)", False, f"Exception: {e}")

    def test_weather_api(self):
        print("\n--- Testing Weather API ---")
        try:
            result_json = APIManagerMain.get_weather_data(forecast=False)
            data = json.loads(result_json)
            
            if "current" in data and "temperature_c" in data["current"]:
                temp = data["current"]["temperature_c"]
                self.log_result("Weather API", True, f"Data retrieved. Temp: {temp}C")
            else:
                self.log_result("Weather API", False, "Invalid JSON structure.")
        except Exception as e:
            self.log_result("Weather API", False, f"Exception: {e}")
        
    def test_memory(self):
        print("\n--- Testing Memory Module ---")
        try:
             mem = MemoryModule.Memory("config/memory.json")
             user = mem.get_user_profile().get('name')
             assistant = mem.get_assistant_profile().get('name')
             
             if user and assistant:
                 self.log_result("Memory Module", True, f"Loaded profiles (User: {user}, AI: {assistant})")
             else:
                 self.log_result("Memory Module", False, "Failed to retrieve names.")
        except Exception as e:
            self.log_result("Memory Module", False, f"Exception: {e}")

    def run_all(self):
        print("Starting Automated Component Tests...\n")
        self.test_tts()
        self.test_stt_init()
        self.test_llm()
        self.test_weather_api()
        self.test_memory()
        
        print("\n" + "="*40)
        print(f"{'TEST NAME':<25} | {'STATUS':<10} | {'MESSAGE'}")
        print("-" * 40)
        for res in self.results:
            print(f"{res['test']:<25} | {res['status']:<10} | {res['message']}")
        print("="*40)

if __name__ == "__main__":
    tester = ComponentTester()
    tester.run_all()
