import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import AnnouncerModule
import HazardModule as H

ON, OFF = H.STOVE_ON, H.STOVE_OFF


def frame(stove=None, person=False, burners=1):
    return SimpleNamespace(stove=stove, person=person, burners_on=burners if stove == ON else 0,
                           labels=[], describe=lambda: f"{stove} {person}")


def rule(t0=0.0, **kw):
    params = dict(confirm_frames=3, unattended_s=600, max_on_s=3600, repeat_s=300)
    params.update(kw)
    return H.StoveRule(t0, **params)


def feed(r, start, end, step, analysis):
    events = []
    t = start
    while t < end:
        events += r.update(t, analysis)
        t += step
    return events


# ─── StoveRule ────────────────────────────────────────────────────────────────
def test_single_flicker_does_not_turn_stove_on():
    r = rule()
    assert r.update(0, frame(ON)) == []
    assert r.update(5, frame(ON)) == []
    r.update(10, frame(OFF))
    assert not r.stove_on


def test_confirmed_on_uses_first_frame_time():
    r = rule()
    events = feed(r, 0, 15, 5, frame(ON))
    assert [e.kind for e in events] == ["stove_on"]
    assert r.stove_on and r.on_since == 0


def test_no_alert_while_someone_is_in_the_kitchen():
    r = rule()
    events = feed(r, 0, 1800, 5, frame(ON, person=True))
    assert [e.kind for e in events] == ["stove_on"]


def test_unattended_alert_then_repeat_then_resolved():
    r = rule()
    events = feed(r, 0, 601, 5, frame(ON))          # nobody seen since t=0
    alerts = [e for e in events if e.severity == "critical"]
    assert len(alerts) == 1 and alerts[0].kind == "stove_unattended"
    assert "10 minutes" in alerts[0].message

    events = feed(r, 605, 905, 5, frame(ON))        # repeat after 5 min
    assert [e.kind for e in events if e.severity == "critical"] == ["stove_unattended"]

    events = feed(r, 905, 925, 5, frame(OFF))
    kinds = [e.kind for e in events]
    assert kinds == ["stove_resolved", "stove_off"]
    assert not r.stove_on and not r.alerting


def test_person_returning_resets_absence():
    r = rule()
    feed(r, 0, 500, 5, frame(ON))
    r.update(500, frame(ON, person=True))
    events = feed(r, 505, 1000, 5, frame(ON))       # only ~500s since person seen
    assert not [e for e in events if e.severity == "critical"]


def test_max_on_alert_even_with_people_present():
    r = rule(max_on_s=1200)
    events = feed(r, 0, 1201, 5, frame(ON, person=True))
    assert [e.kind for e in events if e.severity == "critical"] == ["stove_on_long"]


def test_no_stove_visible_keeps_state():
    r = rule()
    feed(r, 0, 15, 5, frame(ON))
    feed(r, 15, 100, 5, frame(None))
    assert r.stove_on


def test_snooze_suppresses_alerts_until_it_expires():
    r = rule()
    r.snooze(1200)
    events = feed(r, 0, 1195, 5, frame(ON))
    assert not [e for e in events if e.severity == "critical"]
    events = feed(r, 1195, 1210, 5, frame(ON))
    assert [e.kind for e in events if e.severity == "critical"] == ["stove_unattended"]


def test_resolved_only_announced_after_an_alert():
    r = rule()
    feed(r, 0, 100, 5, frame(ON))
    events = feed(r, 100, 120, 5, frame(OFF))
    assert [e.kind for e in events] == ["stove_off"]


# ─── EventLog ─────────────────────────────────────────────────────────────────
def test_event_log_filters_by_time_and_severity(tmp_path):
    log = H.EventLog(tmp_path / "e.db")
    log.add(H.HazardEvent("stove_on", "info", "on", 100))
    log.add(H.HazardEvent("stove_unattended", "critical", "alert", 200, "snap.jpg"))
    assert [e.kind for e in log.recent(0)] == ["stove_on", "stove_unattended"]
    assert [e.kind for e in log.recent(150)] == ["stove_unattended"]
    assert [e.kind for e in log.recent(0, "warning")] == ["stove_unattended"]
    assert log.recent(0)[1].snapshot == "snap.jpg"
    log.close()


# ─── HazardMonitor ────────────────────────────────────────────────────────────
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class ListSource:
    def __init__(self):
        self.frames = []

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def close(self):
        pass


class ScriptedDetector:
    def __init__(self):
        self.next = frame(None)

    def analyse(self, _frame):
        return self.next


@pytest.fixture
def monitor(tmp_path):
    clock, source, detector, said = Clock(), ListSource(), ScriptedDetector(), []
    mon = H.HazardMonitor(source, detector, H.EventLog(tmp_path / "e.db"),
                          announce=lambda text, prio: said.append((prio, text)), clock=clock,
                          rule=rule(clock()), snapshot_dir=tmp_path / "snaps")
    mon.test = SimpleNamespace(clock=clock, source=source, detector=detector, said=said)
    H.set_hazard_monitor(mon)
    yield mon
    H.set_hazard_monitor(None)


def run(mon, seconds, analysis, step=5):
    mon.test.detector.next = analysis
    end = mon.test.clock.t + seconds
    while mon.test.clock.t < end:
        mon.test.source.frames.append(np.zeros((8, 8, 3), dtype=np.uint8))
        mon.step()
        mon.test.clock.t += step


def test_monitor_announces_critical_and_saves_snapshot(monitor, tmp_path):
    run(monitor, 605, frame(ON))
    crit = [t for p, t in monitor.test.said if p == AnnouncerModule.CRITICAL]
    assert len(crit) == 1 and "nobody" in crit[0]
    assert len(list((tmp_path / "snaps").glob("*.jpg"))) == 1
    logged = monitor.event_log.recent(0)
    assert [e.kind for e in logged] == ["stove_on", "stove_unattended"]
    assert logged[1].snapshot


def test_camera_offline_and_restored(monitor):
    monitor.test.clock.t += 120          # no frames for two minutes
    events = monitor.step()
    assert [e.kind for e in events] == ["camera_offline"]
    assert monitor.step() == []          # announced once only
    run(monitor, 5, frame(OFF))
    kinds = [e.kind for e in monitor.event_log.recent(0)]
    assert kinds == ["camera_offline", "camera_restored"]
    assert [p for p, _ in monitor.test.said] == [AnnouncerModule.NORMAL, AnnouncerModule.LOW]


def test_voice_tools(monitor):
    assert "starting up" in H.hazard_status()
    run(monitor, 30, frame(ON, burners=2))
    status = H.hazard_status()
    assert "stove is on with 2 burners lit" in status and "Nobody" in status

    monitor.test.source.frames.append(np.zeros((8, 8, 3), dtype=np.uint8))
    monitor.test.detector.next = frame(ON, person=True)
    assert H.check_stove().startswith("I just looked.")
    assert "Someone is in the kitchen" in H.hazard_status()

    assert "15 minutes" in H.snooze_hazard_alerts(15)
    assert "maximum" in H.snooze_hazard_alerts(500)
    assert "snoozed for another" in H.hazard_status()
    assert "active again" in H.snooze_hazard_alerts(0)

    report = H.recent_events(24)
    assert report.startswith("No safety alerts in the last 24 hours.")
    assert "the stove was turned on" in report


def test_tools_without_monitor():
    H.set_hazard_monitor(None)
    for result in (H.hazard_status(), H.check_stove(), H.snooze_hazard_alerts()):
        assert "not running" in result


def test_make_source(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert isinstance(H.make_source(str(tmp_path)), H.ImageFolderSource)
    assert isinstance(H.make_source("0"), H.CameraSource)
    assert isinstance(H.make_source("rtsp://cam.local/stream"), H.CameraSource)


# ─── Real models end to end ───────────────────────────────────────────────────
def test_real_detector_on_test_photos():
    detector = H.make_detector()
    pics = H.settings.VISION_DIR / "TestPic"
    import cv2
    on = detector.analyse(cv2.imread(str(pics / "stoveOn1.JPG")))
    off = detector.analyse(cv2.imread(str(pics / "stoveOff.JPG")))
    assert on.stove == ON and on.burners_on >= 1
    assert off.stove == OFF
    blank = detector.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
    assert blank.stove is None and not blank.person


# ─── Announcer ────────────────────────────────────────────────────────────────
class FakeTTS:
    """Plays one thing at a time, like the real queue; cut_off cancels the current item."""

    def __init__(self, duration=0.05):
        self.is_speaking = threading.Event()
        self.spoken = []
        self.cut = 0
        self.duration = duration
        self._current = None

    def speak(self, text, volume=None, on_done=None):
        self.spoken.append((text, volume))
        self.is_speaking.set()
        done = threading.Event()
        self._current = (done, on_done)

        def finish():
            if not done.is_set():
                done.set()
                self.is_speaking.clear()
                if on_done:
                    on_done(True)
        threading.Timer(self.duration, finish).start()

    def cut_off(self):
        self.cut += 1
        self.is_speaking.clear()
        if self._current and not self._current[0].is_set():
            done, on_done = self._current
            done.set()
            if on_done:
                on_done(False)


def _wait(cond, timeout=3):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.02)
    return cond()


def test_announcer_priority_dedupe_and_full_volume_for_critical(monkeypatch):
    monkeypatch.setattr(AnnouncerModule.settings, "QUIET_HOURS", "")
    tts = FakeTTS()
    a = AnnouncerModule.Announcer(tts.speak, tts.is_speaking, tts.cut_off)
    a.conversation_active.set()                 # user is talking: normal items must wait
    assert a.say("Your timer is done!", AnnouncerModule.NORMAL)
    a.say("Stove alert!", AnnouncerModule.CRITICAL)
    a.start()
    try:
        assert _wait(lambda: len(tts.spoken) == 1)
        assert tts.spoken[0] == ("Stove alert!", 1.0)     # critical goes first, at full volume
        time.sleep(0.3)
        assert len(tts.spoken) == 1                       # normal still waiting
        a.conversation_active.clear()
        assert _wait(lambda: len(tts.spoken) == 2)
        assert tts.spoken[1] == ("Your timer is done!", None)
    finally:
        a.stop()


def test_critical_interrupts_long_speech(monkeypatch):
    monkeypatch.setattr(AnnouncerModule, "CRITICAL_MAX_WAIT_S", 0.2)
    tts = FakeTTS(duration=0.05)
    tts.is_speaking.set()                        # something long is already playing
    a = AnnouncerModule.Announcer(tts.speak, tts.is_speaking, tts.cut_off)
    a.start()
    try:
        a.say("Stove alert!", AnnouncerModule.CRITICAL)
        assert _wait(lambda: tts.spoken)
        assert tts.cut == 1
    finally:
        a.stop()


@pytest.mark.parametrize("now,spec,expected", [
    ((23, 0), "22:00-07:00", True), ((6, 59), "22:00-07:00", True), ((7, 0), "22:00-07:00", False),
    ((12, 0), "22:00-07:00", False), ((13, 0), "12:00-14:00", True), ((13, 0), "", False),
])
def test_quiet_hours(now, spec, expected):
    from datetime import datetime
    assert AnnouncerModule.in_quiet_hours(datetime(2026, 1, 1, *now), spec) is expected


def test_recent_events_reports_alerts_first(monitor):
    run(monitor, 605, frame(ON))
    report = H.recent_events(12)
    assert report.startswith("There was 1 safety alert in the last 12 hours.")
    assert "I warned that the stove was on with nobody in the kitchen" in report


def test_identical_announcements_are_all_played_in_order():
    tts = FakeTTS(duration=0.05)
    a = AnnouncerModule.Announcer(tts.speak, tts.is_speaking, tts.cut_off)
    for text in ["Your timer is done!", "Your timer is done!", "Reminder: bins", "Your timer is done!"]:
        a.say(text, AnnouncerModule.NORMAL)
    a.start()
    try:
        assert _wait(lambda: len(tts.spoken) == 4)
        assert [t for t, _ in tts.spoken] == ["Your timer is done!", "Your timer is done!",
                                               "Reminder: bins", "Your timer is done!"]
    finally:
        a.stop()


def test_announcement_waits_for_the_previous_one_to_finish():
    tts = FakeTTS(duration=0.4)
    a = AnnouncerModule.Announcer(tts.speak, tts.is_speaking, tts.cut_off)
    a.say("first")
    a.say("second")
    a.start()
    try:
        assert _wait(lambda: len(tts.spoken) == 1)
        time.sleep(0.2)
        assert len(tts.spoken) == 1          # still playing "first"
        assert _wait(lambda: len(tts.spoken) == 2)
    finally:
        a.stop()


def test_interrupted_announcement_is_replayed_once():
    tts = FakeTTS(duration=0.5)
    a = AnnouncerModule.Announcer(tts.speak, tts.is_speaking, tts.cut_off)
    a.say("Reminder: call mum")
    a.start()
    try:
        assert _wait(lambda: len(tts.spoken) == 1)
        tts.cut_off()                         # user barged in with the wake word
        assert _wait(lambda: len(tts.spoken) == 2, timeout=5)
        assert tts.spoken[1][0] == "Reminder: call mum"
        assert _wait(lambda: not tts.is_speaking.is_set())
        tts.cut_off()                         # nothing playing: no further replay
        time.sleep(1)
        assert len(tts.spoken) == 2
    finally:
        a.stop()
