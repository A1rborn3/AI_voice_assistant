"""
Speech input: wake word (Porcupine) + command capture (VAD) + transcription.
===========================================================================
* Porcupine is created once and reused for every wake-word listen.
* Commands are end-pointed with silero-vad (energy fallback) instead of fixed
  timeouts: recording stops once the speaker has been quiet for a moment.
* The captured audio is transcribed by faster-whisper (default when
  installed) or Vosk.
"""

import json
import logging
import platform
import time
from collections import deque

import numpy as np
import pvporcupine
from pvrecorder import PvRecorder

import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_LENGTH = 512                       # Porcupine and silero both use 512 samples @ 16 kHz
FRAME_S = FRAME_LENGTH / SAMPLE_RATE     # 32 ms

if not settings.PICOVOICE_ACCESS_KEY:
    logger.warning("PICOVOICE_ACCESS_KEY not found in .env")


# ─── Voice activity detection ─────────────────────────────────────────────────
class SileroVAD:
    """Neural VAD; returns the probability that a 512-sample frame contains speech."""

    def __init__(self):
        import torch
        from silero_vad import load_silero_vad
        self._torch = torch
        # ONNX backend: lighter than TorchScript, whose loader is deprecated on Python 3.14+
        self.model = load_silero_vad(onnx=True)

    def reset(self):
        self.model.reset_states()

    def prob(self, frame: np.ndarray) -> float:
        audio = self._torch.from_numpy(frame.astype(np.float32) / 32768.0)
        return float(self.model(audio, SAMPLE_RATE).item())


class EnergyVAD:
    """Fallback when silero-vad isn't installed: loudness threshold on RMS."""

    def __init__(self, rms_threshold: float = 500.0):
        self.rms_threshold = rms_threshold

    def reset(self):
        pass

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) if len(frame) else 0.0
        return 1.0 if rms >= self.rms_threshold else 0.0


def create_vad():
    try:
        return SileroVAD()
    except Exception as e:
        logger.warning(f"silero-vad unavailable ({e}); falling back to energy-based VAD.")
        return EnergyVAD()


class Endpointer:
    """
    Decides when a spoken command starts and ends from per-frame speech
    probabilities. Pure logic (no audio devices), so it's unit-testable.

    feed() returns "waiting" (no speech yet), "speaking", "done" (command
    finished) or "timeout" (nobody spoke).
    """

    def __init__(self, frame_s=FRAME_S, threshold=None, end_silence_s=None, min_speech_s=None,
                 pre_roll_s=None, start_timeout_s=None, max_s=None):
        self.frame_s = frame_s
        self.threshold = settings.VAD_THRESHOLD if threshold is None else threshold
        # Hysteresis: once talking, only clearly-silent frames count as silence
        self.silence_threshold = max(self.threshold - 0.15, 0.05)
        self.end_silence_s = settings.VAD_END_SILENCE_S if end_silence_s is None else end_silence_s
        self.min_speech_s = settings.VAD_MIN_SPEECH_S if min_speech_s is None else min_speech_s
        self.start_timeout_s = settings.COMMAND_START_TIMEOUT_S if start_timeout_s is None else start_timeout_s
        self.max_s = settings.COMMAND_MAX_S if max_s is None else max_s
        pre_roll_s = settings.VAD_PRE_ROLL_S if pre_roll_s is None else pre_roll_s

        # Pre-roll must also hold the frames that confirmed speech started
        pre_frames = int(round((pre_roll_s + self.min_speech_s) / frame_s)) + 1
        self._pre_roll = deque(maxlen=pre_frames)
        self._frames = []
        self._elapsed = 0.0
        self._speech_run = 0.0
        self._silence = 0.0
        self.started = False

    def feed(self, frame: np.ndarray, prob: float) -> str:
        self._elapsed += self.frame_s

        if not self.started:
            self._pre_roll.append(frame)
            self._speech_run = self._speech_run + self.frame_s if prob >= self.threshold else 0.0
            if self._speech_run >= self.min_speech_s:
                self.started = True
                self._frames = list(self._pre_roll)
                return "speaking"
            if self._elapsed >= self.start_timeout_s:
                return "timeout"
            return "waiting"

        self._frames.append(frame)
        self._silence = 0.0 if prob >= self.silence_threshold else self._silence + self.frame_s
        if self._silence >= self.end_silence_s or self._elapsed >= self.max_s:
            return "done"
        return "speaking"

    def audio(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._frames).astype(np.int16)


# ─── Transcription ────────────────────────────────────────────────────────────
class WhisperTranscriber:
    name = "whisper"

    def __init__(self):
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper '{settings.WHISPER_MODEL}' on {settings.WHISPER_DEVICE}...")
        # First use downloads the model (~75-500 MB depending on size) into the Hugging Face cache
        self.model = WhisperModel(settings.WHISPER_MODEL, device=settings.WHISPER_DEVICE,
                                  compute_type=settings.WHISPER_COMPUTE)
        logger.info("faster-whisper loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _ = self.model.transcribe(
            audio.astype(np.float32) / 32768.0,
            language="en",
            beam_size=5,
            initial_prompt=settings.WHISPER_PROMPT,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class VoskTranscriber:
    name = "vosk"

    def __init__(self, model_path=None):
        from vosk import Model
        path = model_path or settings.VOSK_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Vosk model not found at {path}")
        logger.info(f"Loading Vosk model from {path}...")
        self.model = Model(str(path))
        logger.info("Vosk model loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.AcceptWaveform(audio.astype(np.int16).tobytes())
        return json.loads(rec.FinalResult()).get("text", "").strip()


def create_transcriber(engine: str | None = None):
    engine = (engine or settings.STT_ENGINE).lower()
    if engine in ("auto", "whisper"):
        try:
            return WhisperTranscriber()
        except Exception as e:
            if engine == "whisper":
                raise
            logger.warning(f"faster-whisper unavailable ({e}); using Vosk.")
    return VoskTranscriber()


# ─── Wake word + command capture ──────────────────────────────────────────────
class STFTModule:
    """Combines wake word detection and speech-to-text."""

    def __init__(self, engine: str | None = None):
        self.transcriber = create_transcriber(engine)
        self.vad = create_vad()
        self.porcupine = None
        self.wake_word = None
        self.recorder = None
        logger.info(f"STT ready: engine={self.transcriber.name}, vad={type(self.vad).__name__}")

    def _init_porcupine(self):
        """Create the Porcupine engine. Returns (engine, wake_word) or (None, None)."""
        key = settings.PICOVOICE_ACCESS_KEY
        if not key or key == "YOUR_ACCESS_KEY_HERE":
            logger.error("PICOVOICE_ACCESS_KEY is not set in the .env file at the repo root")
            return None, None

        sensitivity = settings.WAKE_WORD_SENSITIVITY
        custom_path = settings.CUSTOM_WAKE_WORD_PATH

        # The custom "Samantha" model is built for Raspberry Pi only
        if platform.system() == "Linux" and custom_path.exists():
            try:
                logger.info(f"Loading custom wake word from {custom_path}")
                engine = pvporcupine.create(access_key=key, keyword_paths=[str(custom_path)],
                                            sensitivities=[sensitivity])
                return engine, "samantha"
            except Exception as e:
                logger.error(f"Failed to load custom wake word (expected off a Raspberry Pi): {e}")

        fallback = settings.WAKE_WORD_FALLBACK
        if fallback not in pvporcupine.KEYWORDS:
            logger.warning(f"'{fallback}' is not a built-in Porcupine keyword; using 'jarvis'")
            fallback = "jarvis"
        try:
            engine = pvporcupine.create(access_key=key, keywords=[fallback], sensitivities=[sensitivity])
            logger.info(f"Using built-in wake word '{fallback}' (sensitivity {sensitivity})")
            return engine, fallback
        except Exception as e:
            logger.error(f"Failed to load wake word '{fallback}': {e}")
            return None, None

    def ensure_wake_word_engine(self) -> bool:
        """Create Porcupine on first use; it is then kept for the life of the app."""
        if self.porcupine is None:
            self.porcupine, self.wake_word = self._init_porcupine()
        return self.porcupine is not None

    def listen_for_wake_word(self, until_event_cleared=None):
        """
        Block until the wake word is heard. If `until_event_cleared` is given
        (barge-in while the assistant talks), stop when that event clears, and
        ignore detections in the first BARGE_IN_GRACE_S of playback, which are
        usually the speaker bleeding into the mic.
        """
        import TTSModule  # lazy: avoids loading audio output in STT-only tests

        if not self.ensure_wake_word_engine():
            print("Error: wake word engine could not be initialized (check PICOVOICE_ACCESS_KEY).")
            return False

        try:
            # Index -1 selects the default OS microphone
            self.recorder = PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)
            self.recorder.start()
            logger.info("Listening for wake word on default device...")

            while True:
                if until_event_cleared is not None and not until_event_cleared.is_set():
                    logger.debug("Wake word listening cancelled (event cleared).")
                    return False

                pcm = self.recorder.read()
                if self.porcupine.process(pcm) < 0:
                    continue

                if until_event_cleared is not None:
                    started = TTSModule.speech_started_at
                    if started is not None and time.monotonic() - started < settings.BARGE_IN_GRACE_S:
                        logger.info("Ignoring wake word during barge-in grace period (likely speaker echo).")
                        continue

                logger.info("Wake word detected!")
                # Interruption: if the assistant is talking, stop it immediately
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

    def listen_for_command(self) -> str:
        """Record one spoken command (VAD end-pointed) and return its transcription."""
        recorder = None
        endpointer = Endpointer()
        self.vad.reset()
        try:
            recorder = PvRecorder(device_index=-1, frame_length=FRAME_LENGTH)
            recorder.start()
            logger.info("Listening for command...")
            print("Listening...")

            while True:
                frame = np.asarray(recorder.read(), dtype=np.int16)
                status = endpointer.feed(frame, self.vad.prob(frame))
                if status == "timeout":
                    logger.info("STT: no speech detected.")
                    print("No command heard.")
                    return ""
                if status == "done":
                    break
        except Exception as e:
            logger.error(f"Error capturing command audio: {e}")
            if not endpointer.started:
                return ""
        finally:
            if recorder is not None:
                try:
                    recorder.stop()
                except Exception:
                    pass
                recorder.delete()

        audio = endpointer.audio()
        start = time.perf_counter()
        try:
            text = self.transcriber.transcribe(audio)
        except Exception as e:
            logger.error(f"Error transcribing command: {e}")
            return ""
        logger.info(f"STT ({self.transcriber.name}, {len(audio) / SAMPLE_RATE:.1f}s audio, "
                    f"{time.perf_counter() - start:.2f}s): {text!r}")
        return text

    def close(self):
        if self.porcupine is not None:
            self.porcupine.delete()
            self.porcupine = None
