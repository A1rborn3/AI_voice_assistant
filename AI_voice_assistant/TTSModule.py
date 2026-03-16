from piper.voice import PiperVoice
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import os
import logging
import threading

# Setup logger
logger = logging.getLogger(__name__)

# Global flag for audio sync
is_speaking = threading.Event()

def get_default_voice_path():
    base = Path(__file__).parent / "voices"
    return base / "en_GB-cori-medium.onnx"
    #return base / "en_US-amy-medium.onnx"

def play_audio(file_path):
    try:
        is_speaking.set()
        data, samplerate = sf.read(file_path)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
    finally:
        is_speaking.clear()

def cut_off():
    try:
        sd.stop()
        is_speaking.clear()
    except Exception as e:
        logger.error(f"Error stopping audio: {e}")

class TextToSpeech:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = get_default_voice_path()

        if not os.path.exists(model_path):
            logger.error(f"Voice model not found: {model_path}")
            # Raising error is correct, but let's ensure we can handle it if needed
            raise FileNotFoundError(f"Voice model not found: {model_path}")
            
        self.voice = PiperVoice.load(str(model_path))

    def speak(self, text, output="tts.wav", delete_after=True):
        """Generate speech and stream it to output device immediately."""
        try:
            import numpy as np
            
            # Create a stream
            # Piper usually outputs 16-bit mono audio. Sample rate varies but often 22050.
            # We need to peek at the first chunk to get the rate, then start the stream.
            stream = None
            rate = 22050 

            for chunk in self.voice.synthesize(text):
                # Check for cutoff request mid-synthesis
                if not is_speaking.is_set() and stream is not None:
                    # 'cut_off' sets is_speaking to False.
                    # We should safely abort generating more audio.
                    break
                    
                if chunk.audio_int16_array is not None:
                    # Initialize stream on first chunk with data
                    if stream is None:
                        rate = chunk.sample_rate
                        # We need to run the stream in the MAIN execution context if possible, 
                        # but this function is now threaded. 
                        # sounddevice plays nice with threads usually.
                        stream = sd.OutputStream(samplerate=rate, channels=1, dtype='int16')
                        stream.start()
                        is_speaking.set()
                    
                    stream.write(chunk.audio_int16_array)

            if stream:
                stream.stop()
                stream.close()
            else:
                logger.warning("No audio generated.")

            logger.debug("Finished streaming audio")
            
        except Exception as e:
            logger.error(f"Error in TTS speak: {e}")
        finally:
            is_speaking.clear()

    def save_audio(self, text, output):
        """Generate audio but do NOT play."""
        try:
            audio_data = []
            sample_rate = 22050
            import numpy as np

            for chunk in self.voice.synthesize(text):
                if chunk.audio_int16_array is not None:
                    audio_data.append(chunk.audio_int16_array)
                    sample_rate = chunk.sample_rate
            
            if audio_data:
                full_data = np.concatenate(audio_data)
                sf.write(output, full_data, sample_rate)
        except Exception as e:
            logger.error(f"Error saving audio: {e}")

# Global instance for easy access
_tts_instance = None
import threading
import uuid

def speak(text):
    """Global function to easily use TTS in a non-blocking thread."""
    global _tts_instance
    try:
        if _tts_instance is None:
            _tts_instance = TextToSpeech()
        
        # Use unique filename to avoid conflicts if called rapidly
        filename = f"tts_{uuid.uuid4().hex}.wav"
        
        # Run in a daemon thread so it doesn't block exit
        t = threading.Thread(target=_tts_instance.speak, args=(text, filename, True), daemon=True)
        t.start()
        
    except Exception as e:
        logger.error(f"Failed to initialize or speak: {e}")

def warmup():
    """Initialize the TTS model immediately."""
    global _tts_instance
    try:
        if _tts_instance is None:
            _tts_instance = TextToSpeech()
        logger.info("TTS Warmup complete.")
    except Exception as e:
        logger.error(f"TTS Warmup failed: {e}")
