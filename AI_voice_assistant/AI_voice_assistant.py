import logging_config
import LLMClient
import MemoryModule
import threading
import time
import sys
from datetime import datetime
import TTSModule

#print(LLMClient.askLLM("What is the most common car in NZ"))

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
        context = f"""
        Samantha Assistant:

        Current date and time: {now.strftime("%Y-%m-%d %H:%M")}

        User profile:
        {memory.get_user_profile()}

        Assistant profile:
        {memory.get_assistant_profile()}

        Recent conversation:
        {memory.get_recent_conversation()}

        Reminder Context:
        {input_text}

        Instruction:
        a set reminder has gone off. Generate a response or take action based on the Reminder Context and the provided memory to alert the user.
        """
    elif interaction_type == "conversation":
        context = f"""
        Samantha Assistant:

        Current date and time: {now.strftime("%Y-%m-%d %H:%M")}
        Day of week: {now.strftime("%A")}

        User profile:
        {memory.get_user_profile()}

        Assistant profile:
        {memory.get_assistant_profile()}

        Recent conversation:
        {memory.get_recent_conversation()}

        Constraints:
        All responses must be formatted for TTS output:
        - Output a single line of plain text.
        - Use only normal ASCII characters.
        - Avoid emojis, symbols, and markdown.
        - Use clear grammar and natural pauses with commas or periods.
        - Do not create lists or numbered items.
        - Do not include newlines.
        - Keep output concise but natural for speech.


        The user says:
        {input_text}
        """
    else:#default to conversation
        context = f"""
        Samantha Assistant:

        Current date and time: {now.strftime("%Y-%m-%d %H:%M")}
        Day of week: {now.strftime("%A")}

        User profile:
        {memory.get_user_profile()}

        Assistant profile:
        {memory.get_assistant_profile()}

        Recent conversation:
        {memory.get_recent_conversation()}

        Constraints:
        All responses must be formatted for TTS output:
        - Output a single line of plain text.
        - Use only normal ASCII characters.
        - Avoid emojis, symbols, and markdown.
        - Use clear grammar and natural pauses with commas or periods.
        - Do not create lists or numbered items.
        - Do not include newlines.
        - Keep output concise but natural for speech.

        The user says:
        {input_text}
        """

    response = ask_llm_threaded(context)
    print(response)
    TTSModule.speak(response)

    # Save conversation pair to memory (and file)
    # We pass the original input and the assistant's response
    # The 'response' might need cleaning if it's JSON, but get_recent_conversation 
    # just stores strings, so this is fine.
    if response:
        memory.add_message_pair(input_text, response)
        memory.save() # Explicitly save to disk after every interaction for safety

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
        stt = STTModule.STFTModule() # Initialize
        
        while True:
            print("Waiting for wake word...")
            if stt.listen_for_wake_word():
                print("Listening for command...")
                text = stt.listen_for_command()
                if text:
                    print(f"You said: {text}")
                    if "exit" in text.lower():
                        break
                    process_interaction(text, memory, "conversation")
                else:
                    print("Could not understand command.")
    
    memory.save()
