import logging_config
import LLMClient
import MemoryModule
import threading
import time
import sys
from datetime import datetime
import TTSModule


# Initialize logging early
logging_config.setup_logging()

memory = MemoryModule.Memory("config/memory.json")

def ask_llm_threaded(context):
    """Run the LLM call in a separate thread and return the result."""
    result = {"response": None}
    
    def target():
        result["response"] = LLMClient.askLLM(context)
        
    thread = threading.Thread(target=target)
    thread.start()
    
    # Show loading indicator while waiting
    spinner = "|/-\\"
    idx = 0
    while thread.is_alive():
        sys.stdout.write(f"\rAssistant: {spinner[idx % len(spinner)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
        
    thread.join()
    # Clear spinner
    sys.stdout.write("\rAssistant: ")
    sys.stdout.flush()
    
    return result["response"]

def get_memory():
    return memory

def clean_llm_response(text):
    """Small models hallucinate prompts and context blocks. Clean them before speaking."""
    if not text:
        return ""
        
    lines = text.split('\n')
    cleaned_lines = []
    
    # Catch phrases where the AI starts hallucinating its context or instructions
    stop_phrases = [
        "[CONSTRAINTS]",
        "CRITICAL TOOL RULE:",
        "CONTEXT",
        "DATE/TIME/CONTEXT",
        "User Profile:",
        "Assistant Profile:",
        "Context:",
        "<system_instructions>",
        "<context>",
        "Recent conversation:"
    ]
    
    for line in lines:
        # If we hit a known hallucinated block header, stop taking any more lines
        if any(phrase in line for phrase in stop_phrases):
            break
            
        # Ignore empty lines or ones that are just "Assistant:" prefixes
        clean_ln = line.strip()
        if clean_ln.startswith("Assistant:"):
            clean_ln = clean_ln.replace("Assistant:", "").strip()
            
        if clean_ln:
            cleaned_lines.append(clean_ln)
            
    # Rejoin with spaces for TTS, ensuring we only speak the first natural paragraph
    return " ".join(cleaned_lines)

def process_interaction(input_text, memory, interaction_type):
    now = datetime.now()
    import LLMClient
    import json
    
    # Parse the raw dictionaries into clean context strings so the LLM doesn't get flooded with JSON brackets
    user_prof = memory.get_user_profile()
    user_personality = user_prof.get("personality", "")
    if isinstance(user_personality, str):
        # The background LLM task sometimes wraps this in markdown blocks
        user_personality = user_personality.replace("```json", "").replace("```", "").strip()
    else:
        user_personality = "No specific user details tracked yet."
        
    assistant_prof = memory.get_assistant_profile()
    assistant_context = f"Name: {assistant_prof.get('name', 'Samantha')} | Personality: {assistant_prof.get('personality', 'Helpful')} | Purpose: {assistant_prof.get('purpose', 'assistant')}"
    
    if interaction_type == "reminder":
        system_prompt = f"""--- INSTRUCTIONS ---
You are Samantha. 
1. TEXT RESPONSE: Output a single line of plain text formatted for Text-to-Speech using natural pauses.
2. Maintain your witty, charming personality while alerting the user.

--- CONTEXT ---
Current date and time: {now.strftime("%Y-%m-%d %H:%M")}
User Profile Data:
{user_personality}
Assistant Persona:
{assistant_context}"""
        
        context = [
            {"role": "system", "content": system_prompt}
        ]
        
        for u, a in memory.conversation.to_list():
            context.append({"role": "user", "content": u})
            context.append({"role": "assistant", "content": a})
            
        alert_cmd = f"[TASK] A set reminder has gone off. Context: {input_text}\nGenerate a response to alert the user naturally."
        context.append({"role": "user", "content": alert_cmd})
        
    elif interaction_type == "conversation":
        system_prompt = f"""--- INSTRUCTIONS ---
You are Samantha, a witty, charming, and highly capable personal AI assistant. 
You possess a distinct personality: helpful, calm, concise, and humorous. You love chatting naturally and always stay in character.

1. YOUR TOOLS: You have backend tools that can check the weather, and set, list, or delete alarms/reminders. (Alarms and reminders are the exact same thing).
2. TOOL RULE: If the user asks about the weather, or asks you to set or manage an alarm/reminder, YOU MUST CALL THE CORRESPONDING TOOL. Do not answer these requests with spoken text. Let the tool do the work.
3. SPEECH RULE: If the user is just chatting normally without needing a tool, output a single line of plain text formatted for Text-to-Speech using natural pauses.
4. Never use emojis, symbols, markdown, or numbered lists in your spoken text.
5. ALWAYS use the metric system (Celsius for temperature, kilometers for distance).

--- TOOLS AVAILABLE ---
You may call the following tools if needed:
{json.dumps(LLMClient.tools, indent=2)}

--- CONTEXT ---
Date/Time: {now.strftime("%Y-%m-%d %H:%M")}
Day: {now.strftime("%A")}

User Profile Data:
{user_personality}

Assistant Persona:
{assistant_context}
"""
        
        context = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Append history as actual chat messages
        for u, a in memory.conversation.to_list():
            context.append({"role": "user", "content": u})
            context.append({"role": "assistant", "content": a})
            
        # Append current input
        current_input = input_text
        if any(word in input_text.lower() for word in ["alarm", "remind", "weather", "forecast"]):
             current_input += "\n\n[SYSTEM PUSH: The user is asking for a tool-related action. Output ONLY the raw JSON <tool_call> and NO spoken text.]"
             
        context.append({"role": "user", "content": current_input})
        system_prompt = f"""--- INSTRUCTIONS ---
You are Samantha, a witty, charming, and highly capable personal AI assistant. 
You possess a distinct personality: helpful, calm, concise, and humorous. You love chatting naturally and always stay in character.

1. YOUR TOOLS: You have backend tools that can check the weather, and set, list, or delete alarms/reminders. (Alarms and reminders are the exact same thing).
2. TOOL RULE: If the user asks about the weather, or asks you to set or manage an alarm/reminder, YOU MUST CALL THE CORRESPONDING TOOL. Do not answer these requests with spoken text. Let the tool do the work.
3. SPEECH RULE: If the user is just chatting normally without needing a tool, output a single line of plain text formatted for Text-to-Speech using natural pauses.
4. Never use emojis, symbols, markdown, or numbered lists in your spoken text.
5. ALWAYS use the metric system (Celsius for temperature, kilometers for distance).

--- TOOLS AVAILABLE ---
You may call the following tools if needed:
{json.dumps(LLMClient.tools, indent=2)}

--- CONTEXT ---
Date/Time: {now.strftime("%Y-%m-%d %H:%M")}
Day: {now.strftime("%A")}

User Profile Data:
{user_personality}

Assistant Persona:
{assistant_context}
"""


        context = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Append history as actual chat messages
        for u, a in memory.conversation.to_list():
            context.append({"role": "user", "content": u})
            context.append({"role": "assistant", "content": a})
            
        # Append current input
        current_input = input_text
        if any(word in input_text.lower() for word in ["alarm", "remind", "weather", "forecast"]):
             current_input += "\n\n[SYSTEM PUSH: The user is asking for a tool-related action. Output ONLY the raw JSON <tool_call> and NO spoken text.]"
             
        context.append({"role": "user", "content": current_input})

    response = ask_llm_threaded(context)
    
    # Clean the response of any hallucinated blocks before continuing
    cleaned_response = clean_llm_response(response)
    
    print(f"Original LLM Output:\n{response}")
    if cleaned_response != response and cleaned_response != response.replace('\n', ' '):
        print(f"\n--- Cleaned Output ---\n{cleaned_response}")
        
    TTSModule.speak(cleaned_response)

    # Save conversation pair to memory (and file) using the cleaned response
    if cleaned_response:
        memory.add_message_pair(input_text, cleaned_response)
        memory.save()
        
    return cleaned_response

def wait_for_tts_or_wake_word(stt):
    """Waits for TTS to finish, but allows wake word to interrupt it.
    Returns True if interrupted, False if TTS finished normally."""
    interrupted = [False]
    
    def listen_for_interruption():
        # Using a timeout or a non-blocking check would be better,
        # but since listen_for_wake_word is blocking, we wrap it
        # and forcefully kill the thread when TTS finishes.
        # However, pvporcupine requires processing audio chunks.
        # STTModule.py already has an internal check for TTSModule.is_speaking
        # in listen_for_wake_word that calls TTSModule.cut_off()!
        
        # We just need to call it if it's speaking.
        # Pass TTSModule.is_speaking so it automatically terminates when TTS stops
        if stt.listen_for_wake_word(until_event_cleared=TTSModule.is_speaking):
            interrupted[0] = True
            
    # Start a background thread to listen for the wake word
    interrupt_thread = threading.Thread(target=listen_for_interruption, daemon=True)
    interrupt_thread.start()
    
    # In the main thread, wait for either TTS to finish or an interrupt
    while TTSModule.is_speaking.is_set() and not interrupted[0]:
        time.sleep(0.1)
        
    return interrupted[0]

if __name__ == "__main__":
    TTSModule.warmup()

    print("1. Text Mode")
    print("2. Voice Mode")
    mode = input("Select mode (1/2): ")
    
    if mode == "1":
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            process_interaction(user_input, memory, "conversation")
    
    elif mode == "2":
        import STTModule
        import LLMClient
        stt = STTModule.STFTModule() # Initialize
        
        while True:
            print("Waiting for wake word...")
            if stt.listen_for_wake_word():
                
                while True: # Inner loop for continuous conversation
                    print("Listening for command...")
                    text = stt.listen_for_command()
                    if text:
                        print(f"You said: {text}")
                        if "exit" in text.lower():
                            break
                        
                        # Process interaction and get the AI's spoken text back
                        response_text = process_interaction(text, memory, "conversation")
                        
                        # Fallback for small models: If the AI asked a question, keep the mic open!
                        if response_text and "?" in response_text:
                            # Wait until TTS is finished speaking to avoid microphone feedback
                            # But concurrently listen for wake word to allow interruption
                            interrupted = wait_for_tts_or_wake_word(stt)
                            
                            if interrupted:
                                print("Interrupted!")
                                continue

                            time.sleep(0.5) # Buffer for room echo to dissipate
                            print("Continuing conversation (AI asked a question)...")
                            continue # Skip wake word, listen for command again
                        else:
                            # Wait until TTS is finished before returning to wake word detection
                            interrupted = wait_for_tts_or_wake_word(stt)
                                
                            if interrupted:
                                print("Interrupted!")
                                continue

                            time.sleep(0.5) # Buffer for room echo to dissipate
                            break # Go back to waiting for wake word
                    else:
                        print("Could not understand command.")
                        break # Go back to waiting for wake word

    memory.save()
