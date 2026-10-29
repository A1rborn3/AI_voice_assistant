import json
import time

import pytest

import LLMClient
import MemoryModule
import PeopleMemory


@pytest.fixture
def example_file(tmp_path):
    path = tmp_path / "memory.example.json"
    path.write_text(json.dumps({
        "user": {"name": ""},
        "assistant": {"name": "Nova", "personality": "Dry wit", "purpose": "Safety", "requirements": "Brief"},
        "meta": {"counter": 0, "profile_migrated": True},
    }))
    return path


@pytest.fixture
def people(tmp_path):
    store = PeopleMemory.PeopleMemory(tmp_path / "people.db", embed=lambda text, kind: None)
    yield store
    store.close()


def test_fresh_install_is_seeded_from_example(tmp_path, example_file):
    mem = MemoryModule.Memory(tmp_path / "memory.json", example_path=example_file)
    assert mem.get_assistant_profile()["name"] == "Nova"
    assert (tmp_path / "memory.json").exists()


def test_missing_example_falls_back_to_defaults(tmp_path):
    mem = MemoryModule.Memory(tmp_path / "memory.json", example_path=tmp_path / "nope.json")
    assert mem.get_assistant_profile()["name"] == "Samantha"
    assert "personality" not in mem.get_user_profile()
    assert mem.get_speaker_name() == "The user"


def test_corrupted_file_is_backed_up(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json")
    mem = MemoryModule.Memory(path)
    assert (tmp_path / "memory.json.corrupt").exists()
    assert mem.get_assistant_profile()["name"]


def test_history_messages_and_persistence(tmp_path, people):
    mem = MemoryModule.Memory(tmp_path / "memory.json", people=people)
    mem.add_message_pair("hi", "hello")
    mem.save()
    assert mem.get_history_messages() == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    again = MemoryModule.Memory(tmp_path / "memory.json", people=people)
    assert again.get_history_messages() == mem.get_history_messages()


# ─── Old JSON profile -> people facts ─────────────────────────────────────────
def test_old_profile_is_imported_once_then_removed(tmp_path, people):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({
        "user": {"name": "Ethan", "personality": json.dumps({
            "identity": {"location": "Hamilton", "field_of_study": "computer science", "skills": ["Python"]},
            "interests": ["weather", "alarms"], "technical_preferences": ["Alarming devices"]})},
        "assistant": {"name": "Samantha"},
    }))
    mem = MemoryModule.Memory(path, people=people)
    facts = mem.people.facts_about("Ethan")
    assert "Ethan lives in Hamilton" in facts
    assert "Ethan studies computer science" in facts
    assert "Ethan is skilled in Python" in facts
    assert "Ethan is interested in weather" in facts and "Ethan is interested in alarms" in facts
    assert "Ethan prefers alarming devices" in facts
    saved = json.loads(path.read_text())
    assert "personality" not in saved["user"] and saved["meta"]["profile_migrated"] is True

    # Not imported a second time
    count = people.count()
    MemoryModule.Memory(path, people=people).people
    assert people.count() == count


def test_free_text_profile_is_kept_as_a_fact(tmp_path, people):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"user": {"name": "Ethan", "personality": "likes hiking"}}))
    mem = MemoryModule.Memory(path, people=people)
    assert mem.people.facts_about("Ethan") == ["Ethan: likes hiking"]


# ─── Learning facts from the conversation ─────────────────────────────────────
def test_facts_learned_every_n_exchanges(tmp_path, people, fake_llm, monkeypatch):
    monkeypatch.setattr(MemoryModule.settings, "FACT_EXTRACTION_EVERY", 3)
    llm = fake_llm(['{"facts": [{"person": "Sarah", "fact": "Sarah is allergic to peanuts", "replaces": ""}]}'])
    mem = MemoryModule.Memory(tmp_path / "memory.json", people=people)
    mem.add_message_pair("my sister Sarah is allergic to peanuts", "Noted, I'll keep that in mind.")
    mem.add_message_pair("set a timer for 5 minutes", "Timer number 1 started for 5 minutes.")
    assert not llm.requests
    mem.add_message_pair("tell me a joke", "Why did the clock go to therapy?")
    deadline = time.time() + 3
    while not people.facts_about("Sarah") and time.time() < deadline:
        time.sleep(0.05)
    assert people.facts_about("Sarah") == ["Sarah is allergic to peanuts"]
    # One call, for the only real statement; commands and assistant replies aren't sent
    assert len(llm.requests) == 1
    prompt = llm.requests[0]["messages"][0]["content"]
    assert 'Said: "my sister Sarah is allergic to peanuts"' in prompt
    assert "keep that in mind" not in prompt and "Timer number 1" not in prompt
    assert mem.counter == 0


def test_learning_ignores_non_json_output(tmp_path, people, monkeypatch):
    mem = MemoryModule.Memory(tmp_path / "memory.json", people=people)
    monkeypatch.setattr(LLMClient, "ask_raw", lambda *a, **k: "Sorry, I can't do that.")
    assert mem.learn_facts([("I live in Hamilton", "Nice!")]) == []
    assert people.count() == 0


def test_memory_context_only_returns_relevant_facts(tmp_path, people):
    mem = MemoryModule.Memory(tmp_path / "memory.json", people=people)
    mem.data["user"]["name"] = "Ethan"
    people.add_fact("Sarah", "Sarah is allergic to peanuts")
    people.add_fact("Ethan", "Ethan has a cat called Milo")
    context = mem.memory_context("can Sarah have peanut butter cookies?")
    assert "Sarah is allergic to peanuts" in context
    assert "Milo" not in context
