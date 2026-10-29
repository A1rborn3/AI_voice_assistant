from unittest import mock

import numpy as np
import pytest

import APIManagerMain
import AI_voice_assistant as app
import settings


# ─── Command handling ────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("exit", "quit"),
    ("Exit.", "quit"),                 # Whisper adds punctuation
    ("Shut down!", "quit"),
    ("Stop.", "stop"),
    ("never mind", "stop"),
    ("Goodbye", "stop"),
    ("where's the fire exit", None),   # must not quit the app
    ("cancel my alarm for tomorrow", None),
    ("what's the weather", None),
])
def test_classify_command(text, expected):
    assert app.classify_command(text) == expected


def test_clean_llm_response_strips_markdown_and_json():
    text = "**Sure!**\n- item one\n```py\ncode\n```\nIt is {\"t\": 1} sunny."
    assert app.clean_llm_response(text) == "Sure! item one It is  sunny."


def test_process_interaction_skips_memory_on_llm_error(tmp_path, fake_llm):
    import MemoryModule
    import TTSModule
    mem = MemoryModule.Memory(tmp_path / "memory.json")
    fake_llm([ConnectionError("down")])
    with mock.patch.object(TTSModule, "speak") as speak:
        out = app.process_interaction("hello", mem)
    speak.assert_called_once()
    assert out
    assert mem.get_history_messages() == []


def test_process_interaction_saves_normal_reply(tmp_path, fake_llm):
    import MemoryModule
    import TTSModule
    mem = MemoryModule.Memory(tmp_path / "memory.json")
    fake_llm(["Hi there!"])
    with mock.patch.object(TTSModule, "speak"):
        assert app.process_interaction("hello", mem) == "Hi there!"
    assert mem.get_history_messages()[-1] == {"role": "assistant", "content": "Hi there!"}


def test_barge_in_skipped_when_reply_contains_wake_word(monkeypatch):
    import TTSModule
    stt = mock.Mock(wake_word="samantha")
    monkeypatch.setattr(settings, "BARGE_IN_ENABLED", True)
    TTSModule.is_speaking.clear()
    assert app.wait_for_tts_or_wake_word(stt, "Hi, I'm Samantha!") is False
    stt.listen_for_wake_word.assert_not_called()


# ─── Weather ─────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


@pytest.fixture
def weather_key(monkeypatch):
    monkeypatch.setattr(settings, "WEATHER_API_KEY", "test-key")


def test_weather_api_error_is_reported(weather_key):
    with mock.patch("requests.get", return_value=_Resp({"error": {"message": "API key is invalid."}})):
        assert APIManagerMain.get_weather_data(False) == "Error fetching weather: API key is invalid."


def test_weather_malformed_response(weather_key):
    with mock.patch("requests.get", return_value=_Resp({"weird": 1})):
        assert APIManagerMain.get_weather_data(False).startswith("Error fetching weather")


def test_weather_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "WEATHER_API_KEY", None)
    assert "WEATHER_API_KEY" in APIManagerMain.get_weather_data(False)


def test_weather_success_shape(weather_key):
    data = {
        "location": {"name": "Hamilton", "country": "New Zealand"},
        "current": {"temp_c": 12, "condition": {"text": "Sunny"}},
        "forecast": {"forecastday": [{"date": "2026-10-26", "astro": {}, "day": {"condition": {"text": "Rain"}}}]},
    }
    with mock.patch("requests.get", return_value=_Resp(data)) as get:
        out = APIManagerMain.get_weather_data(True)
    assert out.startswith("Right now in Hamilton it's sunny and 12 degrees.")
    assert "Today: rain" in out
    assert get.call_args.kwargs["params"]["days"] == 3


# ─── Vision ──────────────────────────────────────────────────────────────────
def test_check_stove_on_test_images_and_blank_frame():
    import stove_detector as sd
    pics = sd.SCRIPT_DIR / "TestPic"
    assert sd.check_stove(pics / "stoveOff.JPG") == "STOVE-OFF"
    assert sd.check_stove(pics / "stoveOn1.JPG") == "STOVE-ON"
    assert sd.check_stove(np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_weather_cache_used_when_service_down(weather_key):
    good = {
        "location": {"name": "Hamilton", "country": "New Zealand"},
        "current": {"temp_c": 12, "condition": {"text": "Sunny"}},
        "forecast": {"forecastday": []},
    }
    with mock.patch("requests.get", return_value=_Resp(good)):
        fresh = APIManagerMain.get_weather_data(False)
    with mock.patch("requests.get", side_effect=ConnectionError("offline")):
        cached = APIManagerMain.get_weather_data(False)
    assert cached.startswith("I can't reach the weather service, so this is from")
    assert fresh in cached


def test_weather_cache_too_old_is_ignored(weather_key, monkeypatch):
    import json as _json
    settings.WEATHER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.WEATHER_CACHE_PATH.write_text(_json.dumps(
        {"saved_at": "2000-01-01 00:00", "forecast": False, "result": "{}"}))
    with mock.patch("requests.get", side_effect=ConnectionError("offline")):
        assert APIManagerMain.get_weather_data(False).startswith("Error fetching weather")


def test_clean_llm_response_strips_emoji():
    assert app.clean_llm_response("You've got this! 😊 Good luck ❤️") == "You've got this!  Good luck"
