import json
import time
from datetime import datetime, timedelta

import pytest

import APIManagerMain


def _future(minutes=60):
    t = datetime.now() + timedelta(minutes=minutes)
    return t.replace(second=0, microsecond=0)


def _create(when, context):
    return APIManagerMain.create_reminder(context, time=when.strftime("%H:%M"), date=when.strftime("%Y-%m-%d"))


@pytest.mark.parametrize("text,expected", [
    ("08:30", (8, 30)), ("8.30", (8, 30)), ("19:05", (19, 5)), ("7am", (7, 0)),
    ("8:30 pm", (20, 30)), ("12 am", (0, 0)), ("12:15 p.m.", (12, 15)),
])
def test_parse_time(text, expected):
    assert APIManagerMain.parse_time(text) == expected


@pytest.mark.parametrize("text", ["25:00", "8:75", "13pm", "half eight", ""])
def test_parse_time_rejects_nonsense(text):
    with pytest.raises(ValueError):
        APIManagerMain.parse_time(text)


def test_parse_date():
    now = datetime(2026, 10, 26, 21, 0)
    assert APIManagerMain.parse_date("tomorrow", now) == datetime(2026, 10, 27).date()
    assert APIManagerMain.parse_date("Today", now) == now.date()
    assert APIManagerMain.parse_date("2026-12-25", now) == datetime(2026, 12, 25).date()
    assert APIManagerMain.parse_date("", now) is None
    with pytest.raises(ValueError):
        APIManagerMain.parse_date("next friday", now)


def test_time_without_date_means_next_occurrence(reminder_store):
    soon = datetime.now() - timedelta(minutes=5)  # already passed today
    result = APIManagerMain.create_reminder("coffee", time=soon.strftime("%H:%M"))
    expected_day = (soon + timedelta(days=1)).strftime("%Y-%m-%d")
    assert "tomorrow at" in result or "today at" in result
    assert reminder_store.items()[0][1]["time"].startswith(expected_day)


def test_bad_time_is_reported(reminder_store):
    assert APIManagerMain.create_reminder("x", time="half eight").startswith("Error: I can't understand the time")


def test_ids_are_small_sequential_numbers(reminder_store):
    assert "number 1" in _create(_future(10), "a")
    assert "number 2" in _create(_future(20), "b")
    assert "number 3" in _create(_future(30), "c")
    assert [r_id for r_id, _ in reminder_store.items()] == [1, 2, 3]


def test_lowest_free_number_is_reused(reminder_store):
    for i in range(3):
        _create(_future(10 + i), f"r{i}")
    assert "canceled" in APIManagerMain.delete_reminder(2)
    assert "number 2" in _create(_future(40), "new")


def test_delete_accepts_spoken_number_strings_and_rejects_unknown(reminder_store):
    _create(_future(), "a")
    assert "canceled" in APIManagerMain.delete_reminder("1")
    assert APIManagerMain.delete_reminder(1).startswith("Error")
    assert APIManagerMain.delete_reminder("one").startswith("Error")


def test_past_time_rejected(reminder_store):
    past = datetime.now() - timedelta(hours=1)
    assert "in the past" in _create(past, "late")
    assert reminder_store.items() == []


def test_list_reminders(reminder_store):
    assert "no active reminders" in APIManagerMain.list_reminders()
    _create(_future(), "take the bins out")
    listing = APIManagerMain.list_reminders()
    assert listing.startswith("You have 1 reminder: number 1, take the bins out, ")
    _create(_future(30), "gym session")
    assert "gym" in APIManagerMain.list_reminders(search="GYM")
    assert "bins" not in APIManagerMain.list_reminders(search="gym")
    assert "match" in APIManagerMain.list_reminders(search="dentist")


def test_reminders_survive_restart(reminder_store, tmp_path):
    when = _future(120)
    _create(when, "dentist")
    saved = json.loads((tmp_path / "reminders.json").read_text())
    assert saved == [{"id": 1, "time": when.strftime("%Y-%m-%d %H:%M"), "context": "dentist", "repeat": "none"}]

    # A new store (as after a restart) reloads and reschedules it
    reloaded = APIManagerMain.ReminderStore(tmp_path / "reminders.json", alert=lambda *a, **k: None)
    reloaded.load()
    try:
        assert reloaded.items() == [(1, {"time": when.strftime("%Y-%m-%d %H:%M"), "context": "dentist", "repeat": "none"})]
    finally:
        reloaded.stop()


def test_missed_reminders_are_announced_on_load(tmp_path):
    path = tmp_path / "reminders.json"
    path.write_text(json.dumps([
        {"id": 1, "time": "2000-01-01 09:00", "context": "old"},
        {"id": 2, "time": _future().strftime("%Y-%m-%d %H:%M"), "context": "upcoming"},
        {"bad": "entry"},
    ]))
    alerts = []
    store = APIManagerMain.ReminderStore(path, alert=lambda context, missed_at=None: alerts.append((context, missed_at)))
    store.load()
    try:
        time.sleep(0.2)
        assert alerts == [("old", "2000-01-01 09:00")]
        assert [r_id for r_id, _ in store.items()] == [2]
        # The missed one is removed from the file too
        assert [r["id"] for r in json.loads(path.read_text())] == [2]
    finally:
        store.stop()


def test_due_reminder_fires_once_and_is_removed(reminder_store):
    target = datetime.now() + timedelta(seconds=1)
    reminder_store.add(target, "kettle")
    deadline = time.time() + 5
    while not reminder_store.alerts and time.time() < deadline:
        time.sleep(0.05)
    assert reminder_store.alerts == [("kettle", None)]
    assert reminder_store.items() == []


def test_deleted_reminder_never_fires(reminder_store):
    r_id = reminder_store.add(datetime.now() + timedelta(seconds=1), "cancelled")
    reminder_store.delete(r_id)
    time.sleep(1.5)
    assert reminder_store.alerts == []


# ─── Repeats, relative times, snooze ──────────────────────────────────────────
@pytest.mark.parametrize("target,repeat,after,expected", [
    ("2026-10-26 07:00", "daily", "2026-10-26 08:00", "2026-10-27 07:00"),
    ("2026-10-23 07:00", "weekdays", "2026-10-23 08:00", "2026-10-26 07:00"),   # Friday -> Monday
    ("2026-10-26 07:00", "weekly", "2026-11-01 00:00", "2026-11-02 07:00"),
    ("2026-10-01 07:00", "daily", "2026-10-26 12:00", "2026-10-27 07:00"),      # long offline gap
])
def test_next_occurrence(target, repeat, after, expected):
    fmt = "%Y-%m-%d %H:%M"
    result = APIManagerMain.next_occurrence(datetime.strptime(target, fmt), repeat, datetime.strptime(after, fmt))
    assert result.strftime(fmt) == expected


def test_relative_reminder(reminder_store):
    before = datetime.now()
    result = APIManagerMain.create_reminder("check the oven", in_minutes=20)
    assert result.startswith("Reminder number 1 set for")
    stored = datetime.strptime(reminder_store.items()[0][1]["time"], "%Y-%m-%d %H:%M")
    assert timedelta(minutes=18) < stored - before <= timedelta(minutes=21)


def test_reminder_needs_time_or_minutes(reminder_store):
    assert APIManagerMain.create_reminder("x").startswith("Error: I need a time")
    assert APIManagerMain.create_reminder("x", in_minutes="soon").startswith("Error")
    assert APIManagerMain.create_reminder("x", time="08:00", repeat="hourly").startswith("Error: I can only repeat")


def test_repeating_reminder_reschedules_after_firing(reminder_store):
    reminder_store.add(datetime.now() + timedelta(seconds=1), "pills", "daily")
    deadline = time.time() + 5
    while not reminder_store.alerts and time.time() < deadline:
        time.sleep(0.05)
    assert reminder_store.alerts == [("pills", None)]
    items = reminder_store.items()
    assert len(items) == 1 and items[0][0] == 1 and items[0][1]["repeat"] == "daily"
    nxt = datetime.strptime(items[0][1]["time"], "%Y-%m-%d %H:%M")
    assert timedelta(hours=23) < nxt - datetime.now() < timedelta(hours=25)
    assert "repeating every day" in APIManagerMain.list_reminders()


def test_missed_repeating_reminder_keeps_schedule(tmp_path):
    path = tmp_path / "reminders.json"
    path.write_text(json.dumps([{"id": 1, "time": "2000-01-01 09:00", "context": "pills", "repeat": "daily"}]))
    alerts = []
    store = APIManagerMain.ReminderStore(path, alert=lambda c, missed_at=None: alerts.append((c, missed_at)))
    store.load()
    try:
        time.sleep(0.2)
        assert alerts == [("pills", "2000-01-01 09:00")]
        nxt = datetime.strptime(store.items()[0][1]["time"], "%Y-%m-%d %H:%M")
        assert nxt > datetime.now() and nxt.strftime("%H:%M") == "09:00"
    finally:
        store.stop()


def test_snooze_reminder(reminder_store):
    assert APIManagerMain.snooze_reminder().startswith("Error")
    reminder_store.add(datetime.now() + timedelta(seconds=1), "stretch")
    deadline = time.time() + 5
    while not reminder_store.alerts and time.time() < deadline:
        time.sleep(0.05)
    result = APIManagerMain.snooze_reminder(15)
    assert result.startswith("Snoozed. I'll remind you again at") and "stretch" in result
    assert [info["context"] for _, info in reminder_store.items()] == ["stretch"]
    assert APIManagerMain.snooze_reminder().startswith("Error")   # only once


def test_repeat_word_in_date_field(reminder_store):
    result = APIManagerMain.create_reminder("gym", time="06:45", date="daily", repeat="weekdays")
    assert "repeating every weekday" in result
    result = APIManagerMain.create_reminder("pills", time="08:00", date="every day")
    assert "repeating every day" in result


@pytest.mark.parametrize("word,expected", [
    ("every weekday", "weekdays"), ("Weekday", "weekdays"), ("everyday", "daily"), (None, "none"),
    ("once", "none"), ("every week", "weekly"), ("fortnightly", "fortnightly"),
])
def test_normalise_repeat(word, expected):
    assert APIManagerMain.normalise_repeat(word) == expected


def test_repeating_reminder_with_past_first_time_rolls_forward(reminder_store):
    past = datetime.now() - timedelta(hours=1)
    result = APIManagerMain.create_reminder("gym", time=past.strftime("%H:%M"), date="today", repeat="daily")
    assert result.startswith("Reminder number 1 set for tomorrow at")
