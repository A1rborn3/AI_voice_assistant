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

def process_interaction(input_text, memory, interaction_type):
    now = datetime.now()
    
    if interaction_type == "reminder":
        user_prompt = f"""[INSTRUCTIONS]
You are Samantha.
Current date and time: {now.strftime("%Y-%m-%d %H:%M")}
User profile:
{memory.get_user_profile()}
Assistant profile:
{memory.get_assistant_profile()}
Recent conversation:
{memory.get_recent_conversation()}

[TASK]
A set reminder has gone off. Context: {input_text}
Generate a response or take action to alert the user naturally."""
        
        context = [{"role": "user", "content": user_prompt}]
        
    elif interaction_type == "conversation":
        system_prompt = f"""You are Samantha, a witty, charming, and highly capable personal AI assistant. 
You possess a distinct personality: helpful, calm, concise, and humorous. You love chatting naturally and always stay in character.

Current context:
Date/Time: {now.strftime("%Y-%m-%d %H:%M")}
Day: {now.strftime("%A")}

User profile:
{memory.get_user_profile()}
Assistant profile:
{memory.get_assistant_profile()}

Recent conversation:
{memory.get_recent_conversation()}

[CONSTRAINTS]
1. All responses must be formatted for Text-to-Speech output.
2. Output a single line of plain text. Use natural pauses with commas or periods.
3. Never use emojis, symbols, markdown, or numbered lists.
4. Keep the output concise but natural for speech.

CRITICAL TOOL RULE:
If you need to check the weather or set a reminder, use the provided tools. If you ask the user a question, simply end your sentence with a question mark '?'.
"""
        
        context = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        
    else:#default to conversation
        system_prompt = f"""You are Samantha, a witty, charming, and highly capable personal AI assistant. 
You possess a distinct personality: helpful, calm, concise, and humorous. You love chatting naturally and always stay in character.

Current context:
Date/Time: {now.strftime("%Y-%m-%d %H:%M")}
Day: {now.strftime("%A")}

User profile:
{memory.get_user_profile()}
Assistant profile:
{memory.get_assistant_profile()}

Recent conversation:
{memory.get_recent_conversation()}

[CONSTRAINTS]
1. All responses must be formatted for Text-to-Speech output.
2. Output a single line of plain text. Use natural pauses with commas or periods.
3. Never use emojis, symbols, markdown, or numbered lists.
4. Keep the output concise but natural for speech.

CRITICAL TOOL RULE:
If you need to check the weather or set a reminder, use the provided tools. If you ask the user a question, simply end your sentence with a question mark '?'.
"""

        context = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]

    response = ask_llm_threaded(context)
    print(response)
    TTSModule.speak(response)

    # Save conversation pair to memory (and file)
    if response:
        memory.add_message_pair(input_text, response)
        memory.save()
        
    return response

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
                            while TTSModule.is_speaking.is_set():
                                time.sleep(0.1)
                                
                            print("Continuing conversation (AI asked a question)...")
                            continue # Skip wake word, listen for command again
                        else:
                            # Wait until TTS is finished before returning to wake word detection
                            while TTSModule.is_speaking.is_set():
                                time.sleep(0.1)
                            break # Go back to waiting for wake word
                    else:
                        print("Could not understand command.")
                        break # Go back to waiting for wake word

    memory.save()
