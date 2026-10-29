"""
Proactive announcements.
========================
Everything the assistant says without being asked — hazard alerts, reminders,
timers, warnings — goes through one priority queue, so announcements never
talk over each other or over a reply.

* CRITICAL (hazards) jumps the queue, waits at most a few seconds for the
  current speech, then interrupts it, and always plays at full volume.
* NORMAL/LOW wait until the assistant is quiet and the user isn't mid-
  conversation, and are quieter during QUIET_HOURS.

Nothing is ever dropped: every announcement is queued and played in turn
(same priority = first come, first served), and one that gets cut off by a
barge-in is queued again.
"""

import itertools
import logging
import queue
import threading
import time
from datetime import datetime

import settings

logger = logging.getLogger(__name__)

CRITICAL, NORMAL, LOW = 0, 1, 2        # lower number = more urgent
PRIORITY_NAMES = {CRITICAL: "critical", NORMAL: "normal", LOW: "low"}

CRITICAL_MAX_WAIT_S = 3       # then interrupt whatever is playing
NORMAL_MAX_WAIT_S = 90        # stop waiting for a quiet moment and play anyway (it still never
                              # overlaps: TTS plays everything one after another)
MAX_REPLAYS = 1               # times an interrupted announcement is queued again


def parse_quiet_hours(spec: str):
    """'22:00-07:00' -> ((22, 0), (7, 0)); empty/invalid -> None."""
    try:
        start, end = (part.strip() for part in spec.split("-"))
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
        return (sh, sm), (eh, em)
    except (ValueError, AttributeError):
        return None


def in_quiet_hours(now: datetime | None = None, spec: str | None = None) -> bool:
    hours = parse_quiet_hours(settings.QUIET_HOURS if spec is None else spec)
    if not hours:
        return False
    now = now or datetime.now()
    current = (now.hour, now.minute)
    start, end = hours
    if start <= end:
        return start <= current < end
    return current >= start or current < end   # wraps midnight


class Announcer:
    def __init__(self, speak=None, is_speaking=None, cut_off=None):
        # Injected for tests; default to the real TTS module
        if speak is None or is_speaking is None or cut_off is None:
            import TTSModule
            speak = speak or TTSModule.speak
            is_speaking = is_speaking or TTSModule.is_speaking
            cut_off = cut_off or TTSModule.cut_off
        self._speak = speak
        self._is_speaking = is_speaking
        self._cut_off = cut_off

        # Set by the main loop while the user is talking to the assistant
        self.conversation_active = threading.Event()

        self._queue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._stop = threading.Event()
        self._thread = None
        self.history = []                 # (time, priority, text) of what was spoken

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="announcer", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._queue.put((-1, next(self._seq), None, 0))   # wake the worker

    def say(self, text: str, priority: int = NORMAL) -> bool:
        """Queue an announcement (never dropped; identical texts are each played)."""
        text = (text or "").strip()
        if not text:
            return False
        self._queue.put((priority, next(self._seq), text, 0))
        logger.info(f"Queued {PRIORITY_NAMES.get(priority, priority)} announcement: {text}")
        return True

    def pending(self) -> int:
        return self._queue.qsize()

    def _wait_for_turn(self, priority: int):
        if priority == CRITICAL:
            deadline = time.monotonic() + CRITICAL_MAX_WAIT_S
            while self._is_speaking.is_set() and time.monotonic() < deadline:
                time.sleep(0.1)
            if self._is_speaking.is_set():
                logger.info("Interrupting current speech for a critical announcement")
                self._cut_off()
                time.sleep(0.2)
            return
        deadline = time.monotonic() + NORMAL_MAX_WAIT_S
        while (self._is_speaking.is_set() or self.conversation_active.is_set()) and time.monotonic() < deadline:
            if self._stop.is_set():
                return
            time.sleep(0.2)

    def _run(self):
        while not self._stop.is_set():
            priority, seq, text, replays = self._queue.get()
            if text is None:
                continue
            try:
                self._wait_for_turn(priority)
                print(f"\n[{PRIORITY_NAMES.get(priority, priority)}] {text}")
                self.history.append((datetime.now(), priority, text))
                finished = threading.Event()
                outcome = {"completed": True}

                def on_done(completed, outcome=outcome, finished=finished):
                    outcome["completed"] = completed
                    finished.set()

                # Hazards ignore volume and quiet hours; TTS applies them to the rest
                volume = 1.0 if priority == CRITICAL else None
                self._speak(text, volume=volume, on_done=on_done)
                # Wait for this one to finish before taking the next
                while not finished.wait(0.2):
                    if self._stop.is_set():
                        return
                if not outcome["completed"] and replays < MAX_REPLAYS:
                    # Cut off (barge-in): play it again later rather than lose it.
                    # Keeps its original place among announcements of the same priority.
                    logger.info(f"Announcement was interrupted; queued again: {text}")
                    self._queue.put((priority, seq, text, replays + 1))
                    self._wait_for_quiet()
            except Exception:
                logger.exception("Announcement failed")

    def _wait_for_quiet(self):
        """After an interruption, let the user finish before replaying."""
        time.sleep(0.5)
        deadline = time.monotonic() + NORMAL_MAX_WAIT_S
        while (self._is_speaking.is_set() or self.conversation_active.is_set()) and time.monotonic() < deadline:
            if self._stop.is_set():
                return
            time.sleep(0.2)


_announcer = None
_announcer_lock = threading.Lock()


def get_announcer() -> Announcer:
    """Shared announcer, started on first use."""
    global _announcer
    with _announcer_lock:
        if _announcer is None:
            _announcer = Announcer()
            _announcer.start()
        return _announcer


def set_announcer(announcer: Announcer | None):
    """Swap the shared announcer (tests)."""
    global _announcer
    with _announcer_lock:
        _announcer = announcer


def announce(text: str, priority: int = NORMAL) -> bool:
    return get_announcer().say(text, priority)
