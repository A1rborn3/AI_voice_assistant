import json
import time

import pytest

import LLMClient
import MemoryModule


@pytest.fixture
def example_file(tmp_path):
    path = tmp_path / "memory.example.json"
    path.write_text(json.dumps({
        "user": {"name": "", "personality": MemoryModule.EMPTY_PROFILE},
        "assistant": {"name": "Nova", "personality": "Dry wit", "purpose": "Safety", "requirements": "Brief"},
        "meta": {"counter": 0},
    }))
    return path


def test_fresh_install_is_seeded_from_example(tmp_path, example_file):
    mem = MemoryModule.Memory(tmp_path / "memory.json", example_path=example_file)
    assert mem.get_assistant_profile()["name"] == "Nova"
    assert (tmp_path / "memory.json").exists()


def test_missing_example_falls_back_to_defaults(tmp_path):
    mem = MemoryModule.Memory(tmp_path / "memory.json", example_path=tmp_path / "nope.json")
    assert mem.get_assistant_profile()["name"] == "Samantha"
    assert isinstance(mem.get_user_profile()["personality"], dict)


def test_old_string_profile_is_migrated_to_dict(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({
        "user": {"name": "Ethan", "personality": json.dumps({"interests": ["weather"]})},
        "assistant": {"name": "Samantha"},
    }))
    mem = MemoryModule.Memory(path)
    assert mem.get_user_profile()["personality"] == {"interests": ["weather"]}
    # Migration is written back, and missing assistant fields are filled in
    saved = json.loads(path.read_text())
    assert saved["user"]["personality"] == {"interests": ["weather"]}
    assert "purpose" in saved["assistant"]


def test_free_text_profile_is_kept(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"user": {"personality": "likes hiking"}}))
    mem = MemoryModule.Memory(path)
    assert "likes hiking" in mem.get_user_profile()["personality"]["identity"]["other_details"]


def test_corrupted_file_is_backed_up(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json")
    mem = MemoryModule.Memory(path)
    assert (tmp_path / "memory.json.corrupt").exists()
    assert mem.get_assistant_profile()["name"]


def test_history_messages_and_persistence(tmp_path):
    mem = MemoryModule.Memory(tmp_path / "memory.json")
    mem.add_message_pair("hi", "hello")
    mem.save()
    assert mem.get_history_messages() == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    again = MemoryModule.Memory(tmp_path / "memory.json")
    assert again.get_history_messages() == mem.get_history_messages()


def test_personality_update_runs_once_per_ten_turns(tmp_path, fake_llm):
    llm = fake_llm(['{"identity": {"name": "Ethan"}, "interests": ["weather"]}'])
    mem = MemoryModule.Memory(tmp_path / "memory.json")
    for i in range(10):
        mem.add_message_pair(f"u{i}", f"a{i}")
    deadline = time.time() + 3
    while mem.get_user_profile()["personality"].get("identity", {}).get("name") != "Ethan" and time.time() < deadline:
        time.sleep(0.05)
    assert len(llm.requests) == 1
    assert mem.get_user_profile()["personality"]["interests"] == ["weather"]


def test_personality_update_ignores_non_json(tmp_path, monkeypatch):
    mem = MemoryModule.Memory(tmp_path / "memory.json")
    before = json.dumps(mem.get_user_profile()["personality"])
    monkeypatch.setattr(LLMClient, "ask_raw", lambda *a, **k: "")
    mem.update_user_personality()
    monkeypatch.setattr(LLMClient, "ask_raw", lambda *a, **k: "Sorry, I can't do that.")
    mem.update_user_personality()
    assert json.dumps(mem.get_user_profile()["personality"]) == before
