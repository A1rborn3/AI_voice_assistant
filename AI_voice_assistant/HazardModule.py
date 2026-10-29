"""
Hazard monitoring.
==================
A background thread grabs a camera frame every HAZARD_CHECK_INTERVAL_S,
runs the vision models (Vision_module/detectors.py) and feeds the result to
hazard rules. Rules turn observations into HazardEvents, which are logged to
SQLite (with a snapshot for warnings) and spoken by the announcer.

Current rule: the stove (StoveRule)
  * the stove counts as on/off only after STOVE_CONFIRM_FRAMES agreeing frames
  * CRITICAL alert if it's on and nobody has been seen for STOVE_UNATTENDED_MIN
  * CRITICAL alert if it's been on for STOVE_MAX_ON_MIN, even with people around
  * alerts repeat every HAZARD_REPEAT_MIN until resolved or snoozed
The monitor also warns when the camera stops delivering frames.
"""

import logging
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import settings
import AnnouncerModule

logger = logging.getLogger(__name__)

STOVE_ON, STOVE_OFF = "STOVE-ON", "STOVE-OFF"

# Severity → how it's announced (info is only logged)
SEVERITY_PRIORITY = {
    "critical": AnnouncerModule.CRITICAL,
    "warning": AnnouncerModule.NORMAL,
    "notice": AnnouncerModule.LOW,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SNOOZE_MIN = 60   # safety cap: alerts can't be silenced for longer than this


def _minutes(seconds: float) -> str:
    m = max(int(round(seconds / 60)), 0)
    return "less than a minute" if m == 0 else f"{m} minute{'s' if m != 1 else ''}"


@dataclass
class HazardEvent:
    kind: str          # stove_on, stove_off, stove_unattended, stove_on_long, stove_resolved, camera_offline, camera_restored
    severity: str      # info | notice | warning | critical
    message: str
    time: float
    snapshot: str | None = None


# ─── Rules ────────────────────────────────────────────────────────────────────
class StoveRule:
    """Pure logic: feed it (time, FrameAnalysis) and it returns events."""

    def __init__(self, start_time: float, confirm_frames: int | None = None, unattended_s: float | None = None,
                 max_on_s: float | None = None, repeat_s: float | None = None):
        self.confirm_frames = max(int(settings.STOVE_CONFIRM_FRAMES if confirm_frames is None else confirm_frames), 1)
        self.unattended_s = settings.STOVE_UNATTENDED_MIN * 60 if unattended_s is None else unattended_s
        self.max_on_s = settings.STOVE_MAX_ON_MIN * 60 if max_on_s is None else max_on_s
        self.repeat_s = settings.HAZARD_REPEAT_MIN * 60 if repeat_s is None else repeat_s

        self.stove_on = False
        self.on_since = None
        self.burners_on = 0
        self.last_person_seen = start_time   # assume someone was around at startup
        self.last_alert_at = None
        self.alerting = False
        self.snoozed_until = 0.0
        self._on_streak = self._off_streak = 0
        self._streak_started = None

    def snooze(self, until: float):
        self.snoozed_until = until

    def update(self, now: float, analysis) -> list:
        events = []
        if analysis.person:
            self.last_person_seen = now

        # Debounce: count agreeing frames; frames without a visible stove change nothing
        if analysis.stove == STOVE_ON:
            if self._on_streak == 0:
                self._streak_started = now
            self._on_streak += 1
            self._off_streak = 0
            self.burners_on = analysis.burners_on
        elif analysis.stove == STOVE_OFF:
            if self._off_streak == 0:
                self._streak_started = now
            self._off_streak += 1
            self._on_streak = 0

        if not self.stove_on and self._on_streak >= self.confirm_frames:
            self.stove_on, self.on_since = True, self._streak_started
            events.append(HazardEvent("stove_on", "info", "The stove was turned on.", now))
        elif self.stove_on and self._off_streak >= self.confirm_frames:
            on_for = self._streak_started - self.on_since
            if self.alerting:
                events.append(HazardEvent("stove_resolved", "notice",
                                          "The stove has been switched off.", now))
            events.append(HazardEvent("stove_off", "info",
                                      f"The stove was turned off after {_minutes(on_for)}.", now))
            self.stove_on, self.on_since, self.burners_on = False, None, 0
            self.alerting, self.last_alert_at = False, None

        if self.stove_on:
            alert = self._check_alert(now)
            if alert:
                events.append(alert)
        return events

    def _check_alert(self, now: float):
        on_for = now - self.on_since
        away = now - self.last_person_seen
        if on_for >= self.max_on_s:
            kind = "stove_on_long"
            message = f"Safety check: the stove has been on for {_minutes(on_for)}. Please make sure it's meant to be."
        elif on_for >= self.unattended_s and away >= self.unattended_s:
            kind = "stove_unattended"
            message = (f"Warning: the stove has been on for {_minutes(on_for)} and nobody has been "
                       f"in the kitchen for {_minutes(away)}.")
        else:
            return None
        if now < self.snoozed_until:
            return None
        if self.last_alert_at is not None and now - self.last_alert_at < self.repeat_s:
            return None
        self.last_alert_at, self.alerting = now, True
        return HazardEvent(kind, "critical", message, now)

    def status(self, now: float) -> dict:
        return {
            "stove_on": self.stove_on,
            "on_for_s": (now - self.on_since) if self.stove_on else 0.0,
            "burners_on": self.burners_on if self.stove_on else 0,
            "person_last_seen_s": now - self.last_person_seen,
            "alerting": self.alerting,
            "snoozed_for_s": max(self.snoozed_until - now, 0.0),
        }


# ─── Event log ────────────────────────────────────────────────────────────────
class EventLog:
    """SQLite log of hazard events (thread-safe)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL, kind TEXT NOT NULL, severity TEXT NOT NULL,
            message TEXT NOT NULL, snapshot TEXT)""")
        self._db.commit()

    def add(self, event: HazardEvent):
        with self._lock:
            self._db.execute("INSERT INTO events (ts, kind, severity, message, snapshot) VALUES (?, ?, ?, ?, ?)",
                             (event.time, event.kind, event.severity, event.message, event.snapshot))
            self._db.commit()

    def recent(self, since: float, min_severity: str = "info") -> list:
        order = ["info", "notice", "warning", "critical"]
        allowed = order[order.index(min_severity):]
        with self._lock:
            rows = self._db.execute(
                f"SELECT ts, kind, severity, message, snapshot FROM events WHERE ts >= ? "
                f"AND severity IN ({','.join('?' * len(allowed))}) ORDER BY ts", (since, *allowed)).fetchall()
        return [HazardEvent(kind, sev, msg, ts, snap) for ts, kind, sev, msg, snap in rows]

    def close(self):
        with self._lock:
            self._db.close()


# ─── Frame sources ────────────────────────────────────────────────────────────
class CameraSource:
    """Webcam index, video file or stream URL via OpenCV. Reconnects after failures."""

    def __init__(self, spec):
        self.spec = int(spec) if str(spec).isdigit() else str(spec)
        self._cap = None

    def read(self):
        import cv2
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.spec)
            if not self._cap.isOpened():
                self._cap = None
                return None
        ok, frame = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = None
            return None
        return frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageFolderSource:
    """Cycles through still images — for testing the pipeline without a camera."""

    def __init__(self, path):
        path = Path(path)
        self.images = [path] if path.is_file() else sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self._index = 0

    def read(self):
        import cv2
        if not self.images:
            return None
        frame = cv2.imread(str(self.images[self._index % len(self.images)]))
        self._index += 1
        return frame

    def close(self):
        pass


def make_source(spec: str):
    """'0' → webcam, 'rtsp://…' or a video path → OpenCV, a folder/image → stills."""
    path = Path(spec)
    if not path.is_absolute():
        path = settings.REPO_ROOT / path
    if path.is_dir() or (path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        return ImageFolderSource(path)
    return CameraSource(str(path) if path.is_file() else spec)


def make_detector():
    if str(settings.VISION_DIR) not in sys.path:
        sys.path.insert(0, str(settings.VISION_DIR))
    import detectors
    return detectors.VisionDetector(min_confidence=settings.VISION_CONFIDENCE)


# ─── Monitor ──────────────────────────────────────────────────────────────────
class HazardMonitor:
    def __init__(self, source, detector, event_log: EventLog | None = None, announce=None,
                 interval_s: float | None = None, clock=time.time, rule: StoveRule | None = None,
                 snapshot_dir: Path | None = None, save_snapshots: bool | None = None):
        self.source = source
        self.detector = detector
        self.event_log = event_log
        self.announce = announce or AnnouncerModule.announce
        self.interval_s = settings.HAZARD_CHECK_INTERVAL_S if interval_s is None else interval_s
        self.clock = clock
        self.started_at = clock()
        self.rule = rule or StoveRule(self.started_at)
        self.snapshot_dir = Path(snapshot_dir or settings.SNAPSHOT_DIR)
        self.save_snapshots = settings.SAVE_SNAPSHOTS if save_snapshots is None else save_snapshots

        self.last_analysis = None
        self.last_frame_at = None
        self.camera_offline = False
        self._step_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # ── lifecycle ──
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hazard-monitor", daemon=True)
        self._thread.start()
        logger.info(f"Hazard monitor started (every {self.interval_s}s)")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.source.close()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                logger.exception("Hazard monitor step failed")
            self._stop.wait(self.interval_s)

    # ── one observation ──
    def step(self):
        """Grab a frame, analyse it, apply the rules. Returns the new events."""
        with self._step_lock:
            now = self.clock()
            frame = self.source.read()
            if frame is None:
                return self._handle_missing_frame(now)

            events = []
            if self.camera_offline:
                self.camera_offline = False
                events.append(HazardEvent("camera_restored", "notice", "The kitchen camera is working again.", now))

            analysis = self.detector.analyse(frame)
            self.last_analysis, self.last_frame_at = analysis, now
            logger.debug(f"[Hazard] {analysis.describe()} {analysis.labels}")
            events += self.rule.update(now, analysis)
            for event in events:
                self._record(event, frame)
            return events

    def _handle_missing_frame(self, now):
        last_seen = self.last_frame_at or self.started_at
        if self.camera_offline or now - last_seen < settings.CAMERA_OFFLINE_ALERT_S:
            return []
        self.camera_offline = True
        event = HazardEvent("camera_offline", "warning",
                            "Heads up: I can't see the kitchen camera, so I can't watch the stove.", now)
        self._record(event, None)
        return [event]

    def _record(self, event: HazardEvent, frame):
        if frame is not None and self.save_snapshots and event.severity in ("warning", "critical"):
            event.snapshot = self._save_snapshot(event, frame)
        if self.event_log is not None:
            self.event_log.add(event)
        logger.info(f"[Hazard] {event.severity}: {event.message}")
        priority = SEVERITY_PRIORITY.get(event.severity)
        if priority is not None:
            self.announce(event.message, priority)

    def _save_snapshot(self, event, frame):
        try:
            import cv2
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.snapshot_dir / f"{datetime.fromtimestamp(event.time):%Y%m%d-%H%M%S}_{event.kind}.jpg"
            cv2.imwrite(str(path), frame)
            snapshots = sorted(self.snapshot_dir.glob("*.jpg"))
            for old in snapshots[:-settings.MAX_SNAPSHOTS]:
                old.unlink(missing_ok=True)
            return str(path)
        except Exception:
            logger.exception("Could not save snapshot")
            return None

    # ── queries for voice tools ──
    def snooze(self, minutes: float) -> float:
        minutes = min(max(float(minutes), 0.0), MAX_SNOOZE_MIN)
        self.rule.snooze(self.clock() + minutes * 60)
        return minutes

    def status_text(self) -> str:
        now = self.clock()
        if self.last_frame_at is None:
            return "The hazard monitor is starting up and hasn't seen a camera frame yet."
        s = self.rule.status(now)
        parts = []
        if s["stove_on"]:
            burners = f" with {s['burners_on']} burner{'s' if s['burners_on'] != 1 else ''} lit" if s["burners_on"] else ""
            parts.append(f"The stove is on{burners}, and has been for {_minutes(s['on_for_s'])}.")
        else:
            parts.append("The stove is off.")
        if self.last_analysis is not None and self.last_analysis.person:
            parts.append("Someone is in the kitchen right now.")
        else:
            parts.append(f"Nobody has been seen in the kitchen for {_minutes(s['person_last_seen_s'])}.")
        if self.camera_offline:
            parts.append(f"Warning: the camera hasn't delivered a frame for {_minutes(now - self.last_frame_at)}, so this may be out of date.")
        if s["snoozed_for_s"] > 0:
            parts.append(f"Stove alerts are snoozed for another {_minutes(s['snoozed_for_s'])}.")
        return " ".join(parts)


# ─── Shared instance + voice tools ────────────────────────────────────────────
_monitor = None
_event_log = None


def get_hazard_monitor() -> HazardMonitor | None:
    return _monitor


def set_hazard_monitor(monitor: HazardMonitor | None):
    global _monitor
    _monitor = monitor


def start_hazard_monitoring() -> HazardMonitor | None:
    """Start the camera monitor if CAMERA_ENABLED. Returns it, or None."""
    global _monitor, _event_log
    if not settings.CAMERA_ENABLED:
        logger.info("Hazard monitoring disabled (CAMERA_ENABLED=false)")
        return None
    try:
        _event_log = EventLog(settings.EVENTS_DB_PATH)
        _monitor = HazardMonitor(make_source(settings.CAMERA_SOURCE), make_detector(), _event_log)
        _monitor.start()
        return _monitor
    except Exception:
        logger.exception("Could not start hazard monitoring")
        AnnouncerModule.announce("Hazard monitoring failed to start, so I'm not watching the stove.",
                                 AnnouncerModule.NORMAL)
        _monitor = None
        return None


def _not_running() -> str:
    return "Hazard monitoring is not running, so I can't see the kitchen. A camera needs to be set up first."


def hazard_status():
    """Current stove/kitchen status from the latest observations."""
    monitor = get_hazard_monitor()
    return monitor.status_text() if monitor else _not_running()


def check_stove():
    """Take a fresh look at the stove right now."""
    monitor = get_hazard_monitor()
    if not monitor:
        return _not_running()
    monitor.step()
    if monitor.last_analysis is None or monitor.camera_offline:
        return "I couldn't get a picture from the camera just now."
    return f"I just looked. {monitor.status_text()}"


def snooze_hazard_alerts(minutes=15):
    """Pause repeated stove alerts (max 60 minutes; 0 cancels the snooze)."""
    monitor = get_hazard_monitor()
    if not monitor:
        return _not_running()
    applied = monitor.snooze(minutes)
    if applied == 0:
        return "Stove alerts are active again."
    capped = " That's the maximum." if float(minutes) > MAX_SNOOZE_MIN else ""
    return f"Stove alerts snoozed for {_minutes(applied * 60)}.{capped} I'll still keep watching."


def recent_events(hours=12):
    """Summarise hazard events from the last few hours."""
    monitor = get_hazard_monitor()
    log = (monitor.event_log if monitor else None) or _event_log
    if log is None:
        return _not_running()
    hours = min(max(float(hours), 0.1), 24 * 7)
    now = monitor.clock() if monitor else time.time()
    events = log.recent(now - hours * 3600)
    span = f"the last {hours:g} hour{'s' if hours != 1 else ''}"
    if not events:
        return f"Nothing happened in the kitchen in {span}."
    # Spoken directly (never paraphrased by the LLM, which once summarised a
    # critical alert as "nothing unusual"). Alerts first, then the latest events.
    alerts = [e for e in events if e.severity in ("critical", "warning")]
    parts = []
    if alerts:
        parts.append(f"There {'was' if len(alerts) == 1 else 'were'} {len(alerts)} safety "
                     f"alert{'s' if len(alerts) != 1 else ''} in {span}.")
    else:
        parts.append(f"No safety alerts in {span}.")
    for e in events[-EVENTS_SPOKEN:]:
        when = datetime.fromtimestamp(e.time)
        day = "" if when.date() == datetime.fromtimestamp(now).date() else f"{when:%A} "
        parts.append(f"At {day}{when:%H:%M}, {EVENT_PHRASES.get(e.kind, e.message.rstrip('.').lower())}.")
    return " ".join(parts)


EVENTS_SPOKEN = 6
EVENT_PHRASES = {
    "stove_on": "the stove was turned on",
    "stove_off": "the stove was turned off",
    "stove_unattended": "I warned that the stove was on with nobody in the kitchen",
    "stove_on_long": "I warned that the stove had been on a long time",
    "stove_resolved": "the stove alert was resolved",
    "camera_offline": "the camera stopped working",
    "camera_restored": "the camera came back",
}
