import json
import time

import pytest

import AssistantTools as T
import LLMClient
import TTSModule


# ─── Timers ───────────────────────────────────────────────────────────────────
@pytest.fixture
def timers(monkeypatch):
    done = []
    store = T.TimerStore(on_done=done.append)
    monkeypatch.setattr(T, "_timers", store)
    store.done = done
    return store


def test_timer_fires_and_is_removed(timers):
    assert T.set_timer(seconds=0.3, label="egg").startswith("'egg' timer number 1 started")
    deadline = time.time() + 3
    while not timers.done and time.time() < deadline:
        time.sleep(0.05)
    assert timers.done == ["egg"]
    assert T.timer_status() == "No timers are running."


def test_timer_status_and_cancel(timers):
    assert T.set_timer(minutes=10, label="pasta") == "'pasta' timer number 1 started for 10 minutes."
    assert "number 2" in T.set_timer(minutes=1, seconds=30)
    status = T.timer_status()
    assert "number 1 (pasta) has 10 minutes left" in status or "9 minutes" in status
    assert "more than one timer" in T.cancel_timer()
    assert T.cancel_timer(label="pasta") == "Cancelled timer number 1 (pasta)."
    assert T.cancel_timer() == "Cancelled timer number 2."
    assert T.cancel_timer() == "There are no timers to cancel."
    assert not timers.done


@pytest.mark.parametrize("kwargs", [{}, {"minutes": 0}, {"minutes": "abc"}, {"minutes": 2000}])
def test_timer_rejects_bad_lengths(timers, kwargs):
    assert T.set_timer(**kwargs).startswith("Error")


def test_default_timer_announces(isolated_data, monkeypatch):
    monkeypatch.setattr(T, "_timers", None)
    T.set_timer(seconds=0.2, label="tea")
    deadline = time.time() + 3
    while not isolated_data.said and time.time() < deadline:
        time.sleep(0.05)
    assert isolated_data.said == [(1, "Your tea timer is done!")]


# ─── Lists ────────────────────────────────────────────────────────────────────
def test_lists_add_read_remove(tmp_path):
    assert T.add_to_list("milk, eggs and bread") == "Added milk, eggs, bread to the shopping list."
    assert "Already on the list: Milk" in T.add_to_list("Milk, butter")
    assert T.read_list("Shopping list") == "The shopping list has 4 items: milk, eggs, bread, butter."
    assert T.remove_from_list("egg") == "Removed eggs from the shopping list."
    assert "isn't on" in T.remove_from_list("caviar")
    assert "to-do" in T.add_to_list("call plumber", list_name="todo")
    assert "Lists with items: shopping, to-do" in T.read_list("garden")


def test_clear_list_always_asks_first(monkeypatch):
    monkeypatch.setattr(T, "_pending_clear", {})
    T.add_to_list("milk, eggs")
    # Even if the model passes confirm=true straight away, the first call only asks
    assert T.clear_list(confirm=True) == "Your shopping list has 2 items. Are you sure you want to clear it?"
    assert "2 items" in T.read_list()
    assert T.clear_list(confirm=True) == "Done, I cleared the shopping list. 2 items removed."
    assert "empty" in T.read_list()


def test_clear_list_confirmation_expires(monkeypatch):
    monkeypatch.setattr(T, "_pending_clear", {})
    T.add_to_list("milk")
    T.clear_list()
    monkeypatch.setattr(T, "CONFIRM_WINDOW_S", 0)
    T._pending_clear["shopping"] = 0           # window has passed
    assert T.clear_list(confirm=True).endswith("Are you sure you want to clear it?")
    assert "1 item" in T.read_list()


# ─── Notes ────────────────────────────────────────────────────────────────────
def test_notes_keyword_search_without_embeddings(monkeypatch):
    monkeypatch.setattr(LLMClient, "client", None)
    T.remember_note("The spare key is in the blue pot by the door")
    T.remember_note("Wifi password is on the fridge")
    result = T.recall_note("where is the spare key")
    assert result == "You told me: The spare key is in the blue pot by the door."
    assert "don't have any notes" in T.recall_note("passport")
    assert "Deleted note 1" in T.delete_note(1)
    assert T.delete_note(1).startswith("Error")


def test_notes_semantic_search_with_embeddings(tmp_path):
    vectors = {"key": [1.0, 0.0], "wifi": [0.0, 1.0]}

    def fake_embed(text, kind):
        return vectors["key"] if ("key" in text or "door" in text) else vectors["wifi"]

    store = T.NotesStore(tmp_path / "n.json", embed=fake_embed)
    store.add("The spare key is in the blue pot")
    store.add("Router login is on the fridge")
    # No shared keywords, but the embedding matches
    found = store.search("how do I get in the front door")
    assert [n["text"] for n in found] == ["The spare key is in the blue pot"]
    assert json.loads((tmp_path / "n.json").read_text())[0]["embedding"] == [1.0, 0.0]


# ─── Calculator & units ───────────────────────────────────────────────────────
@pytest.mark.parametrize("expr,expected", [
    ("2 + 2", "4"), ("(12.5 * 4) + 3", "53"), ("10 / 4", "2.5"), ("2^10", "1024"),
    ("sqrt(81)", "9"), ("15% * 80", "12"), ("7 x 6", "42"), ("round(pi, 2)", "3.14"), ("17% of 240", "40.8"),
])
def test_calculate(expr, expected):
    assert T.calculate(expr) == f"{expr} is {expected}."


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "1/0", "9**9999", "a + 1", "x" * 300])
def test_calculate_rejects_unsafe_or_invalid(expr):
    assert T.calculate(expr).startswith("Error")


@pytest.mark.parametrize("value,f,t,expected", [
    (1, "cup", "ml", "250"), (2, "pounds", "kg", "0.9072"), (350, "fahrenheit", "celsius", "176.7"),
    (100, "km/h", "mph", "62.14"), (5, "feet", "metres", "1.524"), (0, "celsius", "kelvin", "273.1"),
    (90, "minutes", "hours", "1.5"),
])
def test_convert_units(value, f, t, expected):
    assert T.convert_units(value, f, t) == f"{value:g} {f} is {expected} {t}."


def test_convert_units_errors():
    assert "can't convert" in T.convert_units(1, "kg", "metres")
    assert "don't know" in T.convert_units(1, "smoots", "m")
    assert "can't convert" in T.convert_units(1, "celsius", "kg")


# ─── Volume & speech ──────────────────────────────────────────────────────────
def test_set_volume(monkeypatch):
    monkeypatch.setattr(TTSModule, "volume", 1.0)
    assert T.set_volume(4).startswith("Volume set to 4 out of 10")
    assert TTSModule.volume == pytest.approx(0.4)
    assert T.set_volume(25).startswith("Volume set to 10")
    assert T.set_volume("loud").startswith("Error")


def test_quiet_hours_cap_volume_but_override_wins(monkeypatch):
    import AnnouncerModule
    monkeypatch.setattr(TTSModule, "volume", 0.9)
    monkeypatch.setattr(AnnouncerModule, "in_quiet_hours", lambda *a, **k: True)
    assert TTSModule.effective_volume() == TTSModule.settings.QUIET_VOLUME
    assert TTSModule.effective_volume(1.0) == 1.0
    monkeypatch.setattr(AnnouncerModule, "in_quiet_hours", lambda *a, **k: False)
    assert TTSModule.effective_volume() == 0.9


@pytest.mark.parametrize("text,expected", [
    ("Alarm set for 08:30.", "Alarm set for 8 30."),
    ("Meet at 19:00", "Meet at 19 o'clock"),
    ("At 7:05 it is 15°C with 81% humidity", "At 7 oh 5 it is 15 degrees Celsius with 81 percent humidity"),
    ("Wind 14 km/h & 2mm rain", "Wind 14 kilometres per hour and 2 millimetres rain"),
    ("0:00", "midnight"),
])
def test_normalise_for_speech(text, expected):
    assert TTSModule.normalise_for_speech(text) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Shopping list", "shopping"), ("groceries", "shopping"), ("todo", "to-do"), ("garden jobs", "garden jobs"),
    ("function requires confirm. please say yes to clear the whole", "shopping"),
    ('{"list_name": "shopping', "shopping"), ("", "shopping"),
])
def test_list_name_sanitised(raw, expected):
    assert T._list_name(raw) == expected


def test_unnamed_timers_are_announced_by_number(isolated_data, monkeypatch):
    monkeypatch.setattr(T, "_timers", None)
    T.set_timer(seconds=0.2)
    T.set_timer(seconds=0.3)
    deadline = time.time() + 3
    while len(isolated_data.said) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert [t for _, t in isolated_data.said] == ["Timer number 1 is done!", "Timer number 2 is done!"]


# ─── TTS playback queue (real TTSModule, fake voice) ─────────────────────────
class FakeVoice:
    def __init__(self):
        self.log = []
        self.active = 0
        self.max_active = 0

    def speak(self, text, gain=1.0, generation=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.log.append(("start", text))
        for _ in range(10):
            if generation is not None and generation != TTSModule._generation:
                self.active -= 1
                self.log.append(("cut", text))
                return False
            time.sleep(0.02)
        self.active -= 1
        self.log.append(("end", text))
        return True


@pytest.fixture
def fake_voice(monkeypatch):
    voice = FakeVoice()
    monkeypatch.setattr(TTSModule, "_tts_instance", voice)
    monkeypatch.setattr(TTSModule.sd, "stop", lambda: None)
    yield voice
    deadline = time.time() + 3
    while TTSModule.is_speaking.is_set() and time.time() < deadline:
        time.sleep(0.02)


def test_tts_plays_one_at_a_time_in_order(fake_voice):
    results = []
    for text in ["reply", "timer done", "reminder"]:
        TTSModule.speak(text, on_done=lambda ok, t=text: results.append((t, ok)))
    assert TTSModule.is_speaking.is_set()
    deadline = time.time() + 5
    while len(results) < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert fake_voice.max_active == 1                       # never overlapping
    assert [t for kind, t in fake_voice.log if kind == "start"] == ["reply", "timer done", "reminder"]
    assert results == [("reply", True), ("timer done", True), ("reminder", True)]
    assert not TTSModule.is_speaking.is_set()


def test_tts_cut_off_cancels_queued_but_not_later_speech(fake_voice):
    results = []
    TTSModule.speak("long reply", on_done=lambda ok: results.append(("long reply", ok)))
    TTSModule.speak("queued", on_done=lambda ok: results.append(("queued", ok)))
    time.sleep(0.05)
    TTSModule.cut_off()                                     # barge-in
    TTSModule.speak("after", on_done=lambda ok: results.append(("after", ok)))
    deadline = time.time() + 5
    while len(results) < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert results == [("long reply", False), ("queued", False), ("after", True)]
