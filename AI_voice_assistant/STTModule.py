
import os
import pvporcupine
import json
import time
from pvrecorder import PvRecorder
from vosk import Model, KaldiRecognizer
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
if not PICOVOICE_ACCESS_KEY:
    logger.warning("PICOVOICE_ACCESS_KEY not found in .env") 

class STFTModule:
    """Combines Wake Word detection and Speech-to-Text."""
    
    def __init__(self, stt_model_path="STT_Model/vosk-model-small-en-us-0.15"):
        # Resolve path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.stt_model_path = os.path.join(base_dir, stt_model_path)
        
        self._init_vosk()
        self.porcupine = None
        self.recorder = None

    def _init_vosk(self):
        if not os.path.exists(self.stt_model_path):
            raise FileNotFoundError(f"Vosk model not found at {self.stt_model_path}")
        logger.info(f"Loading Vosk model from {self.stt_model_path}...")
        self.vosk_model = Model(self.stt_model_path)
        logger.info("Vosk model loaded.")

    def _init_porcupine(self):
        """Initializes the Porcupine wake word engine."""
        if PICOVOICE_ACCESS_KEY == "YOUR_ACCESS_KEY_HERE":
            logger.warning("Picovoice AccessKey not set in STTModule.py")
            print("Please set PICOVOICE_ACCESS_KEY in STTModule.py")
            return None

        # Check for custom model "Samantha"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        samantha_path = os.path.join(base_dir, "STT_Model", "PicoVoice", "Samantha_en_raspberry-pi_v3_0_0.ppn")
        
        # Platform detection
        import platform
        system_os = platform.system()
        
        porcupine = None
        
        if system_os == "Linux" and os.path.exists(samantha_path):
             logger.info(f"Loading custom wake word from {samantha_path}")
             print(f"DEBUG: Attempting to load custom wake word 'Samantha' from {samantha_path}")
             try:
                 porcupine = pvporcupine.create(
                    access_key=PICOVOICE_ACCESS_KEY,
                    keyword_paths=[samantha_path]
                )
             except Exception as e:
                 logger.error(f"Failed to load custom RPi model (expected on non-RPi): {e}")

        
        # Fallback if custom model failed (e.g. wrong OS) or wasn't tried
        if porcupine is None:
            if system_os != "Linux":
                logger.info(f"Detected {system_os}. Custom RPi model skipped. using default 'jarvis' for testing.")
                print(f"DEBUG: Detected {system_os}. Skipping RPi model. Defaulting to wake word: 'Jarvis'")
            else:
                logger.warning(f"Custom model not found at {samantha_path}. Using default 'jarvis'.")
                print(f"DEBUG: Custom model not found. Defaulting to wake word: 'Jarvis'")
            
            try:
                porcupine = pvporcupine.create(
                    access_key=PICOVOICE_ACCESS_KEY,
                    keywords=['jarvis']
                )
            except Exception as e:
                logger.error(f"Failed to load default 'jarvis' model: {e}")
                return None
        
        return porcupine

    def listen_for_wake_word(self, until_event_cleared=None):
        """Blocks until wake word is detected or until_event_cleared is no longer set."""
        try:
            if self.porcupine is None:
                self.porcupine = self._init_porcupine()
                
            if self.porcupine is None:
                print("Error: Porcupine could not be initialized.")
                return False
            
            # Use Index -1 to select the default OS microphone, instead of hardcoding to 0
            self.recorder = PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)

            self.recorder.start()
            
            logger.info("Listening for wake word on default device...")
            
            # Lazy import to prevent circular dependency
            import TTSModule
            
            while True:
                # If an event was provided and it's no longer set, exit without detecting
                if until_event_cleared is not None and not until_event_cleared.is_set():
                    logger.info("Wake word listening cancelled (event cleared).")
                    return False
                    
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                if result >= 0:
                    logger.info("Wake word detected!")
                    
                    # Interruption logic: If the AI is currently talking, stop it immediately!
                    if TTSModule.is_speaking.is_set():
                        logger.info("Interrupting TTS playback...")
                        TTSModule.cut_off()
                        
                    return True
                    
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return False
        finally:
            if self.recorder is not None:
                self.recorder.delete()
                self.recorder = None
            if self.porcupine is not None:
                self.porcupine.delete()
                self.porcupine = None

    def listen_for_command(self):
        """Listens for a command using Vosk and returns text."""
        # Note: We need a new recorder or stream for Vosk since Porcupine uses fixed frame length
        # Vosk likes chunks but we can feed it from pvrecorder too if we want, or sounddevice.
        # Let's use PvRecorder again for consistency if possible, or stick to sounddevice.
        # Vosk is flexible.
        
        try:
            # Vosk requires 16k samplerate normally
            rec = KaldiRecognizer(self.vosk_model, 16000)
            
            # Use PvRecorder for capturing audio. Default device (-1).
            recorder = PvRecorder(device_index=-1, frame_length=512)
            recorder.start()
            
            logger.info("Listening for command...")
            print("Listening...")
            
            full_text = ""
            # Small delay to let the audio system stabilize/clear buffers
            time.sleep(0.5)
            
            start_time = time.perf_counter()
            INITIAL_TIMEOUT = 7.0 
            TOTAL_TIMEOUT = 10.0
            has_started_speaking = False
            
            while True:
                elapsed = time.perf_counter() - start_time
                if int(elapsed * 2) % 10 == 0: # Simple way to print roughly every 1s
                     logger.debug(f"STT: Elapsed {elapsed:.1f}s, speaking={has_started_speaking}")

                # Check for initial silence timeout
                if not has_started_speaking and elapsed > INITIAL_TIMEOUT:
                    logger.info(f"STT: No speech detected in initial {INITIAL_TIMEOUT}s window. Elapsed: {elapsed:.2f}s")
                    print(f"No command heard, timed out after {elapsed:.1f}s.")
                    break
                
                # Check for total command timeout
                if elapsed > TOTAL_TIMEOUT:
                    logger.info(f"STT: Max {TOTAL_TIMEOUT}s limit reached. Disregarding.")
                    full_text = "" 
                    print(f"Command too long (noise?), timed out after {elapsed:.1f}s.")
                    break

                pcm = recorder.read()
                # PvRecorder gives list of ints, need bytes for Vosk
                import struct
                pcm_bytes = struct.pack("h" * len(pcm), *pcm)
                
                if rec.AcceptWaveform(pcm_bytes):
                    res = json.loads(rec.Result())
                    text = res.get('text', '')
                    if text:
                        has_started_speaking = True
                        full_text += " " + text
                        print(f"Recognized: {text}")
                        # For a single-shot command after wake word, we break here.
                        break 
                else:
                    # Check partial result to see if speech has started
                    partial = json.loads(rec.PartialResult())
                    if partial.get('partial', '').strip():
                        has_started_speaking = True
                        
            recorder.stop()
            recorder.delete()
            return full_text.strip()

        except Exception as e:
            logger.error(f"Error in STT: {e}")
            return ""

