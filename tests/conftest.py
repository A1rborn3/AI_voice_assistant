"""Shared test setup: no microphone, speaker, LLM server or network is used."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "AI_voice_assistant"))
sys.path.insert(0, str(REPO_ROOT / "Vision_module"))


class FakeLLM:
    """
    Stands in for the OpenAI client. Give it a list of responses; each is
    either a string (plain reply) or a list of (name, args) tool calls.
    Every request is recorded in `.requests`.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeLLM got more requests than expected")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            message = SimpleNamespace(content=item, tool_calls=None)
        else:
            message = SimpleNamespace(content="", tool_calls=[
                SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(
                    name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
                for i, (name, args) in enumerate(item)
            ])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class SilentAnnouncer:
    """Records announcements instead of speaking them."""

    def __init__(self):
        self.said = []
        import threading
        self.conversation_active = threading.Event()

    def say(self, text, priority=1):
        self.said.append((priority, text))
        return True

    def stop(self):
        pass


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Every test gets its own data files and a silent announcer."""
    import settings
    import AnnouncerModule
    data = tmp_path / "data"
    for name, filename in [("REMINDERS_PATH", "reminders.json"), ("LISTS_PATH", "lists.json"),
                           ("NOTES_PATH", "notes.json"), ("WEATHER_CACHE_PATH", "weather_cache.json"),
                           ("EVENTS_DB_PATH", "events.db"), ("SNAPSHOT_DIR", "snapshots"),
                           ("PEOPLE_DB_PATH", "people.db")]:
        monkeypatch.setattr(settings, name, data / filename)
    import PeopleMemory
    PeopleMemory.set_people_memory(None)          # recreated at the temp path on first use
    announcer = SilentAnnouncer()
    AnnouncerModule.set_announcer(announcer)
    yield announcer
    AnnouncerModule.set_announcer(None)
    store = PeopleMemory._store
    PeopleMemory.set_people_memory(None)
    if store is not None:
        store.close()


@pytest.fixture
def fake_llm(monkeypatch):
    """Usage: llm = fake_llm(["reply"]) — patches LLMClient.client."""
    import LLMClient

    def _install(responses):
        llm = FakeLLM(responses)
        monkeypatch.setattr(LLMClient, "client", llm)
        return llm
    return _install


@pytest.fixture
def reminder_store(tmp_path, monkeypatch):
    """A fresh ReminderStore in a temp file, installed as the shared store; alerts are recorded."""
    import APIManagerMain
    alerts = []
    store = APIManagerMain.ReminderStore(tmp_path / "reminders.json",
                                         alert=lambda context, missed_at=None: alerts.append((context, missed_at)))
    store.alerts = alerts
    monkeypatch.setattr(APIManagerMain, "_store", store)
    yield store
    store.stop()
