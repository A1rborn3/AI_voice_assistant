import pytest

import APIManagerMain
import LLMClient

NOW = "2026-10-26 10:00"


# ─── Tool definitions & argument handling ────────────────────────────────────
def test_tool_specs_are_openai_function_format():
    specs = LLMClient.tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == set(LLMClient.TOOLS)
    weather = next(s for s in specs if s["function"]["name"] == "get_weather_data")
    assert weather["type"] == "function"
    assert weather["function"]["parameters"]["properties"]["forecast"]["type"] == "boolean"
    assert weather["function"]["parameters"]["required"] == ["forecast"]


@pytest.mark.parametrize("value,expected", [("false", False), ("True", True), (0, False), (True, True), ("no", False)])
def test_boolean_args_are_coerced(value, expected):
    assert LLMClient._validate_tool_args("get_weather_data", {"forecast": value}) == {"forecast": expected}


@pytest.mark.parametrize("value", ["2", 2, 2.0])
def test_integer_args_are_coerced(value):
    assert LLMClient._validate_tool_args("delete_reminder", {"reminder_id": value}) == {"reminder_id": 2}


def test_unknown_args_dropped_and_strings_coerced():
    args = LLMClient._validate_tool_args("create_reminder", {"time": 830, "context": "bins", "minute": 30})
    assert args == {"time": "830", "context": "bins"}


def test_missing_required_args_raise():
    with pytest.raises(ValueError, match="missing required"):
        LLMClient._validate_tool_args("create_reminder", {"time": "08:00"})


def test_execute_tool_call_reports_errors_as_text():
    assert LLMClient.execute_tool_call("nope", {}).startswith("Error: unknown tool")
    assert "not valid JSON" in LLMClient.execute_tool_call("get_weather_data", "{bad")
    assert "missing required" in LLMClient.execute_tool_call("delete_reminder", {})


def test_extract_json_object_handles_fences_and_trailing_text():
    raw = 'Sure! ```json\n{"a": {"b": 1}}\n``` and {junk'
    assert LLMClient.extract_json_object(raw) == {"a": {"b": 1}}
    assert LLMClient.extract_json_object("no json here") is None


# ─── Prompt ──────────────────────────────────────────────────────────────────
def test_system_prompt_uses_persona_from_memory():
    profile = {"name": "Nova", "personality": "Dry wit", "purpose": "Home safety", "requirements": "Be brief"}
    prompt = LLMClient.build_system_prompt(NOW, '{"identity": {"name": "Ethan"}}', profile, with_tools=True)
    assert "You are Nova" in prompt
    assert "Dry wit" in prompt and "Home safety" in prompt and "Be brief" in prompt
    assert "Samantha" not in prompt
    assert "Monday" in prompt  # 2026-10-26 is a Monday
    assert "Ethan" in prompt
    assert "Tool rules" in prompt and "kitchen safety tools" in prompt
    assert "Tool rules" not in LLMClient.build_system_prompt(NOW, "", profile, with_tools=False)


# ─── respond(): one call for chat, tool loop for tools ───────────────────────
def test_chat_reply_takes_a_single_call(fake_llm):
    llm = fake_llm(["I'm doing great, thanks!"])
    reply = LLMClient.respond("how are you", [], NOW)
    assert reply.text == "I'm doing great, thanks!"
    assert not reply.error and not reply.ended and reply.tools_used == []
    assert len(llm.requests) == 1
    assert llm.requests[0]["tools"]  # tools offered on the first call


def test_native_tool_call_runs_tool_then_phrases_result(fake_llm, monkeypatch):
    monkeypatch.setattr(LLMClient, "PHRASED_BY_LLM", {"get_weather_data"})
    seen = {}

    def fake_weather(forecast):
        seen["forecast"] = forecast
        return '{"temp_c": 12}'

    monkeypatch.setitem(LLMClient.TOOLS["get_weather_data"], "callable", fake_weather)
    llm = fake_llm([[("get_weather_data", {"forecast": "false"})], "It's 12 degrees."])
    reply = LLMClient.respond("what's the weather", [], NOW)
    assert reply.text == "It's 12 degrees."
    assert reply.tools_used == ["get_weather_data"]
    assert seen["forecast"] is False
    assert len(llm.requests) == 2
    followup = llm.requests[1]["messages"]
    assert followup[-3]["role"] == "assistant" and followup[-3]["tool_calls"][0]["function"]["name"] == "get_weather_data"
    assert followup[-2] == {"role": "tool", "tool_call_id": "call_0", "content": '{"temp_c": 12}'}
    assert followup[-1]["role"] == "system" and followup[-1]["content"] == f"Tool used: get_weather_data. {LLMClient.TOOL_RESULT_NUDGE}"


def test_text_form_tool_call_is_accepted(fake_llm, reminder_store):
    llm = fake_llm(['{"name": "list_reminders", "parameters": {}}'])
    reply = LLMClient.respond("list my alarms", [], NOW)
    assert reply.tools_used == ["list_reminders"]
    assert reply.text == "You currently have no active reminders or alarms."
    assert len(llm.requests) == 1   # spoken directly, no second call


def test_tool_result_is_phrased_without_tools(fake_llm, monkeypatch):
    monkeypatch.setattr(LLMClient, "PHRASED_BY_LLM", {"get_weather_data"})
    monkeypatch.setitem(LLMClient.TOOLS["get_weather_data"], "callable", lambda forecast: "{}")
    llm = fake_llm([[("get_weather_data", {"forecast": False})], "Sunny."])
    LLMClient.respond("weather?", [], NOW)
    assert "tools" in llm.requests[0]
    assert "tools" not in llm.requests[1]


def test_empty_reply_asks_user_to_repeat(fake_llm):
    fake_llm([""])
    reply = LLMClient.respond("mumble", [], NOW)
    assert reply.text == LLMClient.EMPTY_REPLY and reply.error


def test_chained_tool_calls(fake_llm, reminder_store, monkeypatch):
    monkeypatch.setattr(LLMClient, "MAX_TOOL_ROUNDS", 3)
    monkeypatch.setattr(LLMClient, "PHRASED_BY_LLM", set(LLMClient.TOOLS))
    fake_llm([
        [("create_reminder", {"time": "07:00", "date": "2099-01-01", "context": "gym"})],
        [("list_reminders", {})],
        "Done, your gym alarm is number one.",
    ])
    reply = LLMClient.respond("set a gym alarm and tell me my alarms", [], NOW)
    assert reply.tools_used == ["create_reminder", "list_reminders"]
    assert [r_id for r_id, _ in reminder_store.items()] == [1]


def test_bare_tool_name_reply_is_treated_as_call(fake_llm):
    fake_llm(["end_conversation"])
    assert LLMClient.respond("goodbye", [], NOW).ended


def test_prompt_states_tomorrows_date():
    assert "Tomorrow is Tuesday 2026-10-27." in LLMClient.build_system_prompt(NOW, with_tools=True)


def test_end_conversation_ignored_without_goodbye_words(fake_llm, reminder_store):
    llm = fake_llm([[("end_conversation", {})], [("clear_list", {"confirm": True})]])
    reply = LLMClient.respond("yes clear the shopping list", [], NOW)
    assert not reply.ended
    assert "end_conversation" not in {t["function"]["name"] for t in llm.requests[1]["tools"]}


def test_end_conversation_ends_without_reply(fake_llm):
    llm = fake_llm([[("end_conversation", {})]])
    reply = LLMClient.respond("that's all, thanks", [], NOW)
    assert reply.ended and reply.text == ""
    assert len(llm.requests) == 1


def test_llm_unreachable_returns_error_reply(fake_llm):
    fake_llm([ConnectionError("refused")])
    reply = LLMClient.respond("hello", [], NOW)
    assert reply.error
    assert reply.text == LLMClient.FALLBACK_REPLY


def test_tool_rounds_are_capped(fake_llm, reminder_store, monkeypatch):
    monkeypatch.setattr(LLMClient, "MAX_TOOL_ROUNDS", 3)
    monkeypatch.setattr(LLMClient, "PHRASED_BY_LLM", set(LLMClient.TOOLS))
    # The model keeps calling tools; the last round has no tools so it must answer
    responses = [[("list_reminders", {})]] * LLMClient.MAX_TOOL_ROUNDS + ["Final answer."]
    llm = fake_llm(responses)
    reply = LLMClient.respond("loop", [], NOW)
    assert reply.text == "Final answer."
    assert "tools" not in llm.requests[-1]


def test_conversation_ended_constant_matches_tool():
    assert APIManagerMain.end_conversation() == APIManagerMain.CONVERSATION_ENDED


# ─── Tool selection ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected_groups", [
    ("how are you", {"core"}),
    ("is the stove still on?", {"core", "hazards"}),
    ("set a pasta timer for 10 minutes", {"core", "timers"}),
    ("add milk to the shopping list", {"core", "lists"}),
    ("remember the spare key is in the blue pot", {"core", "notes"}),
    ("where's my passport", {"core", "notes"}),
    ("what's 12 x 7", {"core", "maths"}),
    ("how many grams in a pound", {"core", "maths"}),
    ("will it rain tomorrow", {"core", "weather"}),
    ("wake me up at 7", {"core", "reminders"}),
    ("turn it down a bit", {"core", "volume"}),
])
def test_select_tools_by_keywords(text, expected_groups):
    groups = {LLMClient.TOOLS[n]["group"] for n in LLMClient.select_tools(text)}
    assert groups == expected_groups


def test_select_tools_uses_last_reply_for_follow_ups():
    history = [{"role": "user", "content": "clear my shopping list"},
               {"role": "assistant", "content": "Your shopping list has 4 items. Are you sure?"}]
    assert "clear_list" in LLMClient.select_tools("yes", history)


def test_only_selected_tools_are_sent(fake_llm):
    llm = fake_llm(["Hi!"])
    LLMClient.respond("how are you", [], NOW)
    sent = {t["function"]["name"] for t in llm.requests[0]["tools"]}
    assert sent == {"end_conversation"}


def test_empty_schemas_get_placeholder_arg():
    spec = next(s for s in LLMClient.tool_specs(["hazard_status"]))
    assert LLMClient.PLACEHOLDER_ARG in spec["function"]["parameters"]["properties"]
    # and the placeholder is silently dropped when the model fills it in
    assert LLMClient._validate_tool_args("hazard_status", {"reason": "user asked"}) == {}


def test_every_tool_callable_accepts_its_schema():
    import inspect
    for name, info in LLMClient.TOOLS.items():
        params = inspect.signature(info["callable"]).parameters
        for arg in info["schema"]:
            assert arg in params, f"{name} does not accept '{arg}'"
        for arg in info["required"]:
            assert arg in info["schema"], f"{name} requires undeclared '{arg}'"


# ─── Real llama-3.2-3b output shapes seen in live testing ────────────────────
def test_text_form_with_function_key(fake_llm):
    fake_llm(['{"function": "add_to_list", "parameters": {"items": "milk", "list_name": "shopping"}}'])
    reply = LLMClient.respond("add milk to the shopping list", [], NOW)
    assert reply.tools_used == ["add_to_list"]
    assert reply.text == "Added milk to the shopping list."


def test_empty_strings_for_optional_numbers_are_ignored():
    args = LLMClient._validate_tool_args("create_reminder", {
        "context": "gym", "time": "06:45", "date": "", "repeat": "none", "in_minutes": ""})
    assert args == {"context": "gym", "time": "06:45", "date": "", "repeat": "none"}
    assert LLMClient._validate_tool_args("set_volume", {"level": "4"}) == {"level": 4.0}


def test_bare_arguments_infer_the_tool(fake_llm):
    llm = fake_llm(['{"items": "milk, eggs", "list_name": "shopping"}', "Added."])
    reply = LLMClient.respond("add milk and eggs to the shopping list", [], NOW)
    assert reply.tools_used == ["add_to_list"]


def test_ambiguous_bare_arguments_are_not_guessed():
    # list_name alone fits read_list and clear_list
    assert LLMClient._infer_tool_from_args({"list_name": "shopping"}, LLMClient.select_tools("shopping list")) is None


def test_json_reply_is_retried_without_history(fake_llm):
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    llm = fake_llm(['{"weird": 1}', "Here's a joke."])
    reply = LLMClient.respond("tell me a joke", history, NOW)
    assert reply.text == "Here's a joke."
    assert len(llm.requests[1]["messages"]) == 2 and llm.requests[1]["temperature"] == 0.0


def test_json_reply_twice_gives_up_politely(fake_llm):
    fake_llm(['{"weird": 1}', '{"still": 2}'])
    reply = LLMClient.respond("tell me a joke", [], NOW)
    assert reply.text == LLMClient.EMPTY_REPLY and reply.error


def test_history_is_trimmed(fake_llm, monkeypatch):
    monkeypatch.setattr(LLMClient.settings, "LLM_HISTORY_TURNS", 2)
    history = [{"role": r, "content": f"{r}{i}"} for i in range(10) for r in ("user", "assistant")]
    llm = fake_llm(["ok"])
    LLMClient.respond("hello", history, NOW)
    assert len(llm.requests[0]["messages"]) == 1 + 4 + 1


def test_direct_results_skip_the_second_call(fake_llm):
    llm = fake_llm([[("calculate", {"expression": "17% of 240"})]])
    reply = LLMClient.respond("what's 17 percent of 240", [], NOW)
    assert reply.text == "17% of 240 is 40.8."
    assert len(llm.requests) == 1


def test_errors_are_spoken_in_fixed_wording(fake_llm):
    llm = fake_llm([[("set_timer", {})]])
    reply = LLMClient.respond("set a timer", [], NOW)
    assert reply.text == "Sorry, the timer needs a length, like 10 minutes."
    assert len(llm.requests) == 1     # the model never gets to rephrase a failure


@pytest.mark.parametrize("error,spoken", [
    ("Error: bad arguments for create_reminder: missing required argument(s): context.",
     "Sorry, I didn't catch all the details for that. Could you say it again?"),
    ("Error: No reminder or alarm with number 3.", "Sorry, no reminder or alarm with number 3."),
    ("Error: set_timer failed (boom).", "Sorry, something went wrong while doing that."),
    ("Error: I can't convert kg to metres.", "Sorry, I can't convert kg to metres."),
])
def test_spoken_error(error, spoken):
    assert LLMClient.spoken_error(error) == spoken


def test_unoffered_code_call_is_not_read_aloud(fake_llm):
    llm = fake_llm(['convert_units(value=1, from_unit="db", to_unit="%")', "Here's a fun fact."])
    reply = LLMClient.respond("tell me something interesting", [], NOW)
    assert reply.text == "Here's a fun fact."


@pytest.mark.parametrize("text,expected", [
    ('read_list list_name="shopping"', ("read_list", {"list_name": "shopping"})),
    ('recall_note(query="spare key")', ("recall_note", {"query": "spare key"})),
    ('read_list("shopping")', ("read_list", {"list_name": "shopping"})),
    ('clear_list("shopping", confirm=true)', ("clear_list", {"list_name": "shopping", "confirm": True})),
    ('convert_units(value=3, from_unit="pound", to_unit="gram")',
     ("convert_units", {"value": 3, "from_unit": "pound", "to_unit": "gram"})),
    ('set_volume(level=4)', ("set_volume", {"level": 4})),
    ('create_reminder(context="gym", time="06:45", repeat=None)',
     ("create_reminder", {"context": "gym", "time": "06:45", "repeat": None})),
    ("`timer_status()`", ("timer_status", {})),
    # comma/space/quoted forms seen live from llama-3.2-3b
    ('read_list, list_name="shopping"', ("read_list", {"list_name": "shopping"})),
    ("'clear_list', list_name='shopping', confirm=False", ("clear_list", {"list_name": "shopping", "confirm": False})),
    ('timer_status, reason = "pasta"', ("timer_status", {"reason": "pasta"})),
    ("read_list shopping", ("read_list", {"list_name": "shopping"})),
    ("convert_units, value=3, from_unit='pounds', to_unit='grams'",
     ("convert_units", {"value": 3, "from_unit": "pounds", "to_unit": "grams"})),
    ('"recall_note", query="the spare key location"', ("recall_note", {"query": "the spare key location"})),
    ("set_volume, level=4", ("set_volume", {"level": 4})),
    ("calculate the tip for me please", None),
    ('Read_list(list_name="shopping")', ("read_list", {"list_name": "shopping"})),
    ("I'll read_list for you now", None),
    ("Sounds good (see you later)", None),
])
def test_parse_code_style_calls(text, expected):
    assert LLMClient._parse_code_call(text) == expected


def test_code_style_call_executes(fake_llm):
    fake_llm(['calculate(expression="2+2")'])
    assert LLMClient.respond("what's 2 plus 2", [], NOW).text == "2+2 is 4."


def test_number_args_accept_text_with_units():
    assert LLMClient._validate_tool_args("set_timer", {"minutes": "10 minutes"}) == {"minutes": 10.0}
    with pytest.raises(ValueError):
        LLMClient._validate_tool_args("set_timer", {"minutes": "ten"})


def test_claimed_action_without_tool_forces_a_tool_call(fake_llm, reminder_store):
    llm = fake_llm(["The shopping list has been cleared.", [("read_list", {"list_name": "shopping"})]])
    reply = LLMClient.respond("clear my shopping list", [], NOW)
    assert reply.tools_used == ["read_list"]
    assert llm.requests[1]["tool_choice"] == "required"
    assert llm.requests[0]["tool_choice"] == "auto"


def test_plain_chat_is_not_forced(fake_llm):
    llm = fake_llm(["Cooking is great fun!"])
    reply = LLMClient.respond("I love cooking in the kitchen", [], NOW)
    assert reply.text == "Cooking is great fun!" and len(llm.requests) == 1


@pytest.mark.parametrize("text,claims", [
    ("Running reminders: The shopping list has been cleared.", True),
    ("Milk, eggs and bread are on your shopping list.", True),
    ("The pasta timer has 9 minutes left.", True),
    ("Volume adjusted to 4.", True),
    ("Shopping list updated: milk, eggs, bread.", True),
    ("'shopping' list: milk, eggs", True),
    ("Why did the clock go to therapy? It was wound up.", False),
    ("I'm doing well, thanks for asking!", False),
    ("Clearing the shopping list... Done.", True),
])
def test_claims_action(text, claims):
    assert LLMClient._claims_action(text) is claims


@pytest.mark.parametrize("user_text,model_repeat,expected", [
    ("set an alarm for 6:45 every weekday", "weekly", "weekdays"),
    ("remind me to take my pills every day at 8", "none", "daily"),
    ("bins reminder weekly on monday at 7", "daily", "weekly"),
    ("set an alarm for 7", "daily", "none"),      # the user never asked for a repeat
])
def test_arg_hints_from_user_words(user_text, model_repeat, expected):
    args = LLMClient._apply_arg_hints("create_reminder", {"context": "x", "repeat": model_repeat}, user_text)
    assert args["repeat"] == expected


def test_text_calls_limited_to_offered_tools(fake_llm):
    fake_llm(['convert_units, value=3, from_unit="lb", to_unit="g"', 'convert_units, value=3'])
    reply = LLMClient.respond("tell me something interesting", [], NOW)
    assert reply.tools_used == [] and reply.text == LLMClient.EMPTY_REPLY


def test_missing_timer_length_filled_from_words():
    args = LLMClient._apply_arg_hints("set_timer", {"label": "pasta"}, "set a pasta timer for 10 minutes")
    assert args == {"label": "pasta", "minutes": "10"}
    args = LLMClient._apply_arg_hints("set_timer", {"minutes": 5}, "set a timer for 10 minutes")
    assert args["minutes"] == 5   # the model's value is kept when present


@pytest.mark.parametrize("user_text,args,expected", [
    ("set a tea timer for 5 seconds", {"minutes": 5, "seconds": 5, "label": "tea"}, {"seconds": 5, "label": "tea"}),
    ("timer for 2 minutes", {"minutes": 2, "seconds": 2}, {"minutes": 2}),
    ("one hour timer", {"minutes": 60}, {"minutes": 60}),
])
def test_timer_units_follow_the_users_words(user_text, args, expected):
    assert LLMClient._apply_arg_hints("set_timer", args, user_text) == expected


def test_unprompted_hazard_talk_forces_a_tool(fake_llm, monitor_off):
    llm = fake_llm(["No alarms were triggered in the kitchen.", [("recent_events", {})]])
    reply = LLMClient.respond("what happened in the kitchen recently?", [], NOW)
    assert reply.tools_used == ["recent_events"]
    assert llm.requests[1]["tool_choice"] == "required"


@pytest.fixture
def monitor_off():
    import HazardModule
    HazardModule.set_hazard_monitor(None)
    yield


@pytest.mark.parametrize("user_text,args,expected", [
    ("remind me in 20 minutes to check the oven",
     {"context": "oven", "in_minutes": 20, "repeat": "daily"}, {"context": "oven", "in_minutes": 20, "repeat": "none"}),
    ("set an alarm for 6:45 every weekday to go to the gym",
     {"context": "gym", "time": "06:45", "in_minutes": 645, "repeat": "weekly"},
     {"context": "gym", "time": "06:45", "repeat": "weekdays"}),
    ("remind me in 2 hours to call mum", {"context": "call mum"},
     {"context": "call mum", "in_minutes": "120.0", "repeat": "none"}),
])
def test_reminder_args_follow_the_users_words(user_text, args, expected):
    assert LLMClient._apply_arg_hints("create_reminder", args, user_text) == expected


def test_unreadable_confirm_means_no():
    assert LLMClient._validate_tool_args("clear_list", {"confirm": "pending"}) == {"confirm": False}


@pytest.mark.parametrize("tool,user_text,allowed", [
    ("delete_note", "where is the spare key", False),
    ("delete_note", "forget the note about the key", True),
    ("delete_reminder", "what reminders do I have", False),
    ("delete_reminder", "cancel alarm number two", True),
    ("cancel_timer", "how long is left on the timer", False),
    ("clear_list", "what's on my list", False),
    ("clear_list", "yes", True),
    ("end_conversation", "yes clear it", False),
    ("end_conversation", "that's all thanks", True),
])
def test_word_gates(tool, user_text, allowed):
    assert bool(LLMClient.WORD_GATES[tool].search(user_text)) is allowed


def test_destructive_call_without_matching_words_is_blocked(fake_llm):
    import AssistantTools
    AssistantTools.remember_note("The spare key is in the blue pot")
    llm = fake_llm([[("delete_note", {"note_id": 1})], [("recall_note", {"query": "spare key"})]])
    reply = LLMClient.respond("where is the spare key", [], NOW)
    assert reply.text == "You told me: The spare key is in the blue pot."
    assert "delete_note" not in {t["function"]["name"] for t in llm.requests[1]["tools"]}


@pytest.mark.parametrize("text,offered_text,claims", [
    ("12 times 34 is 408.", "what's 12 times 34", True),            # did the maths itself
    ("I'll remember that the key is in the pot.", "remember the key", True),
    ("Are you sure you want to clear the shopping list?", "clear my list", True),
    ("I don't have current weather data.", "what's the weather", True),
    ("Cooking is great fun!", "I love cooking", False),
])
def test_claims_action_qwen_phrasings(text, offered_text, claims):
    assert LLMClient._claims_action(text, LLMClient.select_tools(offered_text)) is claims


def test_bare_capitalised_tool_name():
    assert LLMClient._text_tool_calls("Timer_status", None) == [("call_text_0", "timer_status", {})]


def test_tool_not_forced_by_groups_from_the_previous_reply(fake_llm):
    history = [{"role": "user", "content": "my sister is allergic to peanuts"},
               {"role": "assistant", "content": "I'll check the kitchen safety settings."}]
    llm = fake_llm(["Oh no, I hope Milo feels better soon. Have you seen a vet?"])
    reply = LLMClient.respond("my cat Milo has been sick lately", history, NOW)
    assert reply.tools_used == [] and len(llm.requests) == 1


def test_unbacked_promise_is_regenerated(fake_llm):
    llm = fake_llm(["I'll make sure to remind you to keep peanuts away from Sarah.",
                    "Thanks for telling me, I'll keep that in mind for when Sarah visits."])
    reply = LLMClient.respond("my sister Sarah is allergic to peanuts", [], NOW)
    assert reply.text.startswith("Thanks for telling me")
    # A fresh retry with the instruction in the system prompt (not appended after the reply)
    retry = llm.requests[1]["messages"]
    assert retry[0]["content"].endswith(LLMClient.NO_PROMISES_NUDGE)
    assert retry[-1] == {"role": "user", "content": "my sister Sarah is allergic to peanuts"}
    assert "tools" not in llm.requests[1]


def test_long_sentences_dont_pick_tools_from_the_last_reply():
    history = [{"role": "assistant", "content": "I'll check the kitchen safety settings."}]
    assert {LLMClient.TOOLS[n]["group"] for n in LLMClient.select_tools("my cat Milo has been sick lately", history)} == {"core"}
    assert "clear_list" in LLMClient.select_tools("yes clear it", [{"role": "assistant", "content": "Clear the shopping list?"}])


def test_promise_removed_if_regeneration_still_promises(fake_llm):
    fake_llm(["I'll check the kitchen safety for Milo.",
              "Oh no, poor Milo. I'll keep an eye on things. I hope he's better soon."])
    reply = LLMClient.respond("my cat Milo has been sick lately", [], NOW)
    assert reply.text == "Oh no, poor Milo. I hope he's better soon."


def test_topic_rules_only_for_offered_groups():
    chat = LLMClient.build_system_prompt(NOW, with_tools=True, tool_groups={"core"})
    assert "kitchen safety" not in chat and "calculate" not in chat
    safety = LLMClient.build_system_prompt(NOW, with_tools=True, tool_groups={"core", "hazards"})
    assert "kitchen safety tools" in safety
