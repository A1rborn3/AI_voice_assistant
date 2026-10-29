import hashlib
import math
import re

import pytest

import LLMClient
import PeopleMemory as P

_STOP = {"is", "the", "a", "an", "to", "of", "in", "and", "has", "have", "can", "what", "does", "do", "for", "her", "his"}


def fake_embed(text, kind):
    """Deterministic bag-of-words embedding: shared words -> higher cosine."""
    vec = [0.0] * 64
    for word in re.findall(r"[a-z]+", text.lower()):
        word = word.rstrip("s")
        if word in _STOP:
            continue
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(P.settings, "MEMORY_MIN_SIMILARITY", 0.3)
    s = P.PeopleMemory(tmp_path / "people.db", embed=fake_embed)
    P.set_people_memory(s)
    yield s
    P.set_people_memory(None)
    s.close()


def test_add_and_list(store):
    assert store.add_fact("sarah", "Sarah is allergic to peanuts") == "added"
    assert store.add_fact("Ethan", "Ethan has a cat called Milo") == "added"
    assert store.people() == ["Ethan", "Sarah"]
    assert store.facts_about("SARAH") == ["Sarah is allergic to peanuts"]


def test_near_duplicates_are_merged(store):
    store.add_fact("Sarah", "Sarah is allergic to peanuts")
    assert store.add_fact("Sarah", "sarah is allergic to peanuts.") == "updated"
    assert store.count() == 1


def test_replaces_old_fact(store):
    store.add_fact("Sarah", "Sarah lives in Auckland")
    store.add_fact("Sarah", "Sarah lives in Wellington", replaces="Sarah lives in Auckland")
    assert store.facts_about("Sarah") == ["Sarah lives in Wellington"]


@pytest.mark.parametrize("person,fact", [
    ("", "Something true"), ("Sarah", "hi"), ("Sarah", "x" * 200),
    ("Ethan", "Ethan wants to set an alarm for 7"), ("Ethan", "Ethan asked for a timer"),
])
def test_rejects_bad_facts(store, person, fact):
    assert store.add_fact(person, fact) == "rejected"
    assert store.count() == 0


def test_search_finds_relevant_and_skips_unrelated(store):
    store.add_fact("Sarah", "Sarah is allergic to peanuts")
    store.add_fact("Ethan", "Ethan has a cat called Milo")
    store.add_fact("Ethan", "Ethan studies computer science")
    found = [r["fact"] for r in store.search("is that cake safe with peanuts allergy")]
    assert found[0] == "Sarah is allergic to peanuts"
    assert store.search("set a timer for ten minutes") == []


def test_context_includes_named_people(store):
    store.add_fact("Sarah", "Sarah's birthday is in March")
    store.add_fact("Ethan", "Ethan has a cat called Milo")
    context = store.context_for("I'm seeing Sarah tonight")
    assert "Sarah's birthday is in March" in context
    assert "Milo" not in context


def test_keyword_fallback_without_embeddings(tmp_path):
    s = P.PeopleMemory(tmp_path / "p.db", embed=lambda text, kind: None)
    s.add_fact("Sarah", "Sarah is allergic to peanuts")
    assert [r["fact"] for r in s.search("peanuts")] == ["Sarah is allergic to peanuts"]
    assert s.search("weather tomorrow") == []
    s.close()


def test_backfill_embeddings(tmp_path):
    available = {"on": False}
    s = P.PeopleMemory(tmp_path / "p.db", embed=lambda t, k: fake_embed(t, k) if available["on"] else None)
    s.add_fact("Sarah", "Sarah is allergic to peanuts")
    assert s.backfill_embeddings() == 0          # still unavailable
    available["on"] = True
    assert s.backfill_embeddings() == 1
    assert s._rows()[0]["embedding"]
    s.close()


def test_forget(store):
    store.add_fact("Ethan", "Ethan likes jazz")
    store.add_fact("Ethan", "Ethan has a cat called Milo")
    assert store.forget("Ethan", "I like jazz") == ["Ethan likes jazz"]
    assert store.forget("Ethan", "skiing holidays") == []
    assert store.forget("Ethan", "everything") == ["Ethan has a cat called Milo"]
    assert store.count() == 0


def test_import_profile(store):
    added = store.import_profile("Ethan", {
        "identity": {"location": "Hamilton", "age": "", "skills": ["Python"], "other_details": ["Has two sisters"]},
        "interests": ["weather"], "hobbies": [], "assistant_expectations": ["Provide accurate forecasts.", "maintain context"]})
    assert added == 5
    assert set(store.facts_about("Ethan")) == {
        "Ethan lives in Hamilton", "Ethan is skilled in Python", "Ethan is interested in weather",
        "Ethan wants the assistant to provide accurate forecasts", "Ethan: Has two sisters"}


# ─── Extraction from conversation ─────────────────────────────────────────────
def test_extraction_stores_grounded_facts_only(store, fake_llm):
    fake_llm(['```json\n{"facts": ['
              '{"person": "Ethan", "fact": "Ethan has a dog called Rex", "replaces": ""},'
              '{"person": "Ethan", "fact": "Ethan enjoys skydiving", "replaces": ""},'
              '{"person": "Ethan", "fact": "Ethan wants an alarm at 7", "replaces": ""}]}\n```'])
    stored = P.extract_and_store(store, [
        ("my dog Rex keeps barking", "Rex sounds lively! Maybe he wants to go skydiving."),
        ("set an alarm for 7", "Reminder number 1 set for tomorrow at 07:00.")], "Ethan")
    # skydiving only came from the assistant; the alarm is a request, not a fact
    assert stored == ["Ethan has a dog called Rex"]
    assert store.facts_about("Ethan") == ["Ethan has a dog called Rex"]


def test_extraction_with_replacement(store, fake_llm):
    store.add_fact("Sarah", "Sarah lives in Auckland")
    fake_llm(['{"facts": [{"person": "Sarah", "fact": "Sarah lives in Wellington", '
              '"replaces": "Sarah lives in Auckland"}]}'])
    P.extract_and_store(store, [("Sarah moved to Wellington last week", "Oh nice!")], "Ethan")
    assert store.facts_about("Sarah") == ["Sarah lives in Wellington"]


def test_extraction_handles_garbage(store, fake_llm):
    fake_llm(["I couldn't find any facts."])
    assert P.extract_and_store(store, [("hello", "hi")], "Ethan") == []


# ─── Voice tools ──────────────────────────────────────────────────────────────
@pytest.fixture
def speaker(monkeypatch):
    monkeypatch.setattr(P, "_speaker", lambda: "Ethan")


def test_recall_person(store, speaker):
    assert "don't know anything about Sarah" in P.recall_person("Sarah")
    store.add_fact("Sarah", "Sarah is allergic to peanuts")
    store.add_fact("Ethan", "Ethan has a cat called Milo")
    assert P.recall_person("sarah") == "Here's what I know about sarah: Sarah is allergic to peanuts."
    assert "Milo" in P.recall_person("me")
    assert "I do know a little about Ethan, Sarah" in P.recall_person("Bob")


def test_forget_fact_tool(store, speaker):
    store.add_fact("Ethan", "Ethan likes jazz")
    assert P.forget_fact("that I like jazz") == "Okay, I've forgotten that Ethan likes jazz."
    assert P.forget_fact("jazz").startswith("Error")


def test_recall_note_falls_back_to_people_facts(store):
    import AssistantTools
    store.add_fact("Sarah", "Sarah works at the hospital")
    assert AssistantTools.recall_note("where does Sarah work") == "You told me: Sarah works at the hospital."


def test_people_tools_offered_and_gated():
    assert "recall_person" in LLMClient.select_tools("what do you know about Sarah")
    assert "forget_fact" in LLMClient.select_tools("forget that I like jazz")
    assert not LLMClient.WORD_GATES["forget_fact"].search("what do you know about me")


def test_prompt_shows_speaker_and_relevant_facts():
    prompt = LLMClient.build_system_prompt("2026-10-29 09:00", "- Sarah is allergic to peanuts",
                                           {"name": "Samantha"}, with_tools=True, speaker="Ethan")
    assert "You are talking to Ethan." in prompt
    assert "- Sarah is allergic to peanuts" in prompt


@pytest.mark.parametrize("text,is_statement", [
    ("my sister Sarah is allergic to peanuts", True),
    ("I've got a cat called Milo", True),
    ("Sarah moved to Wellington last month", True),
    ("set a timer for 20 minutes", False),
    ("can you remind me to call mum", False),
    ("what's the weather like", False),
    ("tell me a joke", False),
    ("thanks", False),
])
def test_only_statements_are_sent_for_extraction(text, is_statement):
    assert P._is_statement(text) is is_statement


def test_extraction_is_one_call_per_statement_with_known_facts(store, fake_llm):
    store.add_fact("Sarah", "Sarah lives in Auckland")
    llm = fake_llm(['{"facts": [{"person": "Ethan", "fact": "Ethan has a cat called Milo", "replaces": ""}]}',
                    '{"facts": [{"person": "Sarah", "fact": "Sarah lives in Wellington", "replaces": "Sarah lives in Auckland"}]}'])
    stored = P.extract_and_store(store, [
        ("I've got a cat called Milo", "Cute!"),
        ("set a timer for 5 minutes", "Timer number 1 started for 5 minutes."),
        ("Sarah moved to Wellington last month", "Nice city."),
    ], "Ethan")
    assert stored == ["Ethan has a cat called Milo", "Sarah lives in Wellington"]
    assert len(llm.requests) == 2
    # The known Sarah fact is in the second prompt so the model can say what it replaces
    assert "- Sarah lives in Auckland" in llm.requests[1]["messages"][0]["content"]
    assert store.facts_about("Sarah") == ["Sarah lives in Wellington"]


def test_relationship_fact_naming_the_speaker_is_grounded(store, fake_llm):
    fake_llm(["""{"facts": [{"person": "Sarah", "fact": "Sarah is Ethan's sister", "replaces": ""}]}"""])
    assert P.extract_and_store(store, [("my sister Sarah is coming over", "Lovely!")], "Ethan") == ["Sarah is Ethan's sister"]
