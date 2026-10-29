from piper.voice import PiperVoice
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import queue
import re
import time
import logging
import threading

import settings

# Setup logger
logger = logging.getLogger(__name__)

# Set while speech is being synthesised/played; cleared when done or cut off
is_speaking = threading.Event()

# time.monotonic() when audio for the current utterance started playing
# (None until the first chunk reaches the speaker). Used for barge-in grace.
speech_started_at = None

# Current master volume 0.0-1.0 (changed at runtime by the set_volume tool)
volume = settings.TTS_VOLUME


def get_default_voice_path():
    return settings.TTS_VOICE_PATH


def set_volume(level: float) -> float:
    """Set the master volume (0.0-1.0) and return the value applied."""
    global volume
    volume = min(max(float(level), 0.0), 1.0)
    return volume


def effective_volume(override: float | None = None) -> float:
    """Explicit override wins; otherwise the master volume, capped during quiet hours."""
    if override is not None:
        return min(max(float(override), 0.0), 1.0)
    import AnnouncerModule   # lazy: avoids a circular import
    if AnnouncerModule.in_quiet_hours():
        return min(volume, settings.QUIET_VOLUME)
    return volume


# ─── Text normalisation for speech ────────────────────────────────────────────
_UNIT_WORDS = [
    (re.compile(r"\s*°\s*C\b"), " degrees Celsius"),
    (re.compile(r"\s*°\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"\s*°"), " degrees"),
    (re.compile(r"(?<=\d)\s*%"), " percent"),
    (re.compile(r"(?<=\d)\s*(?:km/h|kph)\b", re.IGNORECASE), " kilometres per hour"),
    (re.compile(r"(?<=\d)\s*mm\b"), " millimetres"),
    (re.compile(r"(?<=\d)\s*km\b"), " kilometres"),
    (re.compile(r"\s&\s"), " and "),
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
]
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _speak_time(match: re.Match) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if minute == 0:
        return f"{hour} o'clock" if hour else "midnight"
    if minute < 10:
        return f"{hour} oh {minute}"
    return f"{hour} {minute}"


def normalise_for_speech(text: str) -> str:
    """Rewrite symbols and clock times the voice would read badly ('08:30' -> '8 30')."""
    text = _TIME_RE.sub(_speak_time, text)
    for pattern, words in _UNIT_WORDS:
        text = pattern.sub(words, text)
    return re.sub(r"\s{2,}", " ", text).strip()


def cut_off():
    """Stop playback now and cancel anything queued before this moment."""
    global _generation
    try:
        with _state_lock:
            _generation += 1
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
            raise FileNotFoundError(f"Voice model not found: {model_path}")

        self.voice = PiperVoice.load(str(model_path))

    def speak(self, text, gain: float = 1.0, generation: int | None = None) -> bool:
        """
        Synthesise speech and stream it to the output device as it is generated.
        Returns True if it played to the end, False if it was cut off.
        """
        global speech_started_at
        stream = None
        completed = False
        try:
            for chunk in self.voice.synthesize(normalise_for_speech(text)):
                # cut_off() bumps the generation; stop generating more audio
                if generation is not None and generation != _generation:
                    break

                if chunk.audio_int16_array is not None:
                    # Open the stream on the first chunk, once the sample rate is known
                    if stream is None:
                        stream = sd.OutputStream(samplerate=chunk.sample_rate, channels=1, dtype='int16')
                        stream.start()
                        speech_started_at = time.monotonic()
                    audio = chunk.audio_int16_array
                    if gain < 1.0:
                        audio = (audio.astype(np.float32) * gain).astype(np.int16)
                    stream.write(audio)

            else:
                completed = generation is None or generation == _generation
            if stream is None:
                logger.warning("No audio generated.")
            logger.debug("Finished streaming audio")

        except Exception as e:
            logger.error(f"Error in TTS speak: {e}")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as e:
                    logger.error(f"Error closing audio stream: {e}")
            speech_started_at = None
        return completed

    def save_audio(self, text, output):
        """Generate audio to a WAV file without playing it."""
        try:
            audio_data = []
            sample_rate = 22050

            for chunk in self.voice.synthesize(text):
                if chunk.audio_int16_array is not None:
                    audio_data.append(chunk.audio_int16_array)
                    sample_rate = chunk.sample_rate

            if audio_data:
                sf.write(output, np.concatenate(audio_data), sample_rate)
        except Exception as e:
            logger.error(f"Error saving audio: {e}")


# Global instance for easy access
_tts_instance = None

# ─── Playback queue ───────────────────────────────────────────────────────────
# Everything spoken goes through ONE worker in first-in-first-out order, so a
# reply and an announcement can never play over each other: the second one
# waits. cut_off() (barge-in) bumps the generation, which stops the current
# utterance and cancels whatever was queued before it.
_play_queue = queue.Queue()
_state_lock = threading.Lock()
_pending = 0          # utterances queued or playing
_generation = 0
_worker = None


def _play_worker():
    global _pending
    while True:
        text, gain, generation, on_done = _play_queue.get()
        completed = False
        try:
            if generation == _generation:
                completed = _tts_instance.speak(text, gain, generation)
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")
        finally:
            with _state_lock:
                _pending -= 1
                if _pending == 0:
                    is_speaking.clear()
            if on_done is not None:
                try:
                    on_done(completed)
                except Exception:
                    logger.exception("on_done callback failed")


def speak(text, volume: float | None = None, on_done=None):
    """
    Queue `text` to be spoken (returns immediately; is_speaking stays set until
    the queue is empty). `volume` overrides the master volume and quiet hours
    (used for hazard alerts). `on_done(completed: bool)` is called afterwards;
    completed is False if it was cut off or cancelled.
    """
    global _tts_instance, _pending, _worker
    try:
        if _tts_instance is None:
            _tts_instance = TextToSpeech()
        with _state_lock:
            if _worker is None or not _worker.is_alive():
                _worker = threading.Thread(target=_play_worker, name="tts-playback", daemon=True)
                _worker.start()
            _pending += 1
            # Set here so callers checking is_speaking right after speak() see it
            is_speaking.set()
            _play_queue.put((text, effective_volume(volume), _generation, on_done))
    except Exception as e:
        logger.error(f"Failed to initialize or speak: {e}")
        if on_done is not None:
            on_done(False)


def warmup():
    """Initialize the TTS model immediately."""
    global _tts_instance
    try:
        if _tts_instance is None:
            _tts_instance = TextToSpeech()
        logger.info("TTS Warmup complete.")
    except Exception as e:
        logger.error(f"TTS Warmup failed: {e}")
