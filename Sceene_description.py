import os
import threading
import queue
import torch # Keep for checking cuda availability potentially
import sounddevice as sd # Keep for PortAudioError
import numpy as np # Keep for type checks
import time
import re
# Removed direct KPipeline import here if only used via initiate_tts_model
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI # Keep for type hinting

# --- Assuming these utils exist and contain the CORRECT functions ---
# Updated function names and added initiate_tts_model
from utils.model import initiate_llm, initiate_tts_model # Import the updated function
from utils.to_base64 import encode_image_to_base64
from utils.audio_player import audio_player
from utils.audio_generator import audio_generator
from utils.snap_a_picture import capture_image # Assuming this is a utility function for image capture
# --------------------------------------------------------------------

# --- Configurations (Keep as before) ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL_IDENTIFIER = "lmstudio-community/granite-vision-3.2-2b-GGUF"
os.environ["OPENAI_API_KEY"] = "lm-studio"
TTS_VOICE = 'am_onyx'
AUDIO_SAMPLE_RATE = 24000
MIN_WORDS_FOR_AUDIO = 3
MAX_BUFFER_LENGTH = 150
AUDIO_CHUNK_TIMEOUT = 0.1
AUDIO_PLAYBACK_PAUSE = 0.05
AUDIO_PLAYER_CHECK_INTERVAL = 0.1
MAX_AUDIO_WAIT_TIME = 30

# --- Main Scene Description Function (Accept tts_pipeline as argument) ---
def sceene_description_with_tts(tts_pipeline): # Added tts_pipeline argument
    """
    Describes an image using a multimodal LLM and speaks the description
    using TTS as it streams in. Uses a pre-initialized TTS pipeline.
    Initializes threads, queues, and events for a single run.
    """
    # Check if a valid TTS pipeline was passed
    if tts_pipeline is None:
        print("Error: A valid TTS pipeline instance is required for sceene_description_with_tts.")
        return
    image_path=capture_image()
    print(f"--- Starting Scene Description with TTS for: {image_path} ---")

    # --- Initialize Queues and Events PER RUN ---
    audio_queue = queue.Queue()
    text_queue = queue.Queue()
    text_ready = threading.Event()
    generation_done = threading.Event()
    audio_playing = threading.Event()

    # 1. Initialize LLM (MUST be streaming enabled)
    print("Initializing LLM...")
    try:
        # Ensure initiate_llm returns a streaming-enabled model
        llm = initiate_llm(LM_STUDIO_BASE_URL, LM_STUDIO_MODEL_IDENTIFIER, temperature=0.5)
        # The check for streaming is now primarily within initiate_llm or should be assumed working
        print("LLM initialized.")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return # Exit if LLM fails

    # 2. Encode Image
    print("Encoding image...")
    base64_image_data_uri = encode_image_to_base64(image_path)
    if not base64_image_data_uri:
        print("Error: Could not encode image. Exiting.")
        return

    # 3. Prepare Prompt and Message
    prompt = "Describe this image in detail for a person who is blind. Focus on object positions, types, and potential obstacles or points of interest."
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": base64_image_data_uri},
            },
        ]
    )

    # 4. Start Audio Threads (Pass the tts_pipeline instance)
    print("Starting audio threads...")
    audio_player_thread = threading.Thread(
        target=audio_player,
        args=(
            audio_queue, generation_done, text_ready, audio_playing,
            AUDIO_SAMPLE_RATE, AUDIO_PLAYBACK_PAUSE, AUDIO_PLAYER_CHECK_INTERVAL, AUDIO_CHUNK_TIMEOUT
        ),
        daemon=True
    )

    # Pass the pre-initialized tts_pipeline instance to the generator thread
    audio_generator_thread = threading.Thread(
        target=audio_generator,
        args=(
            text_queue, audio_queue, generation_done, text_ready,
            tts_pipeline, TTS_VOICE, MIN_WORDS_FOR_AUDIO # Use the passed tts_pipeline
        ),
        daemon=True
    )
    audio_generator_thread.start() # Start generator thread (TTS pipeline is guaranteed non-None here)
    audio_player_thread.start() # Start player thread


    # 5. Process LLM Stream and Chunk Text (Keep this logic as before)
    print("\nSending request and processing LLM stream...")
    sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=[.?!])\s+|\n\s*')
    accumulated_text = ""
    full_response_for_debug = ""

    try:
        for chunk in llm.stream([message]):
             if chunk.content:
                # Optional console output
                # print(chunk.content, end="", flush=True)
                print(chunk.content, end="", flush=True)
                accumulated_text += chunk.content
                full_response_for_debug += chunk.content

                # --- Text Chunking Logic (identical to previous version) ---
                parts = sentence_pattern.split(accumulated_text)
                if len(parts) > 1:
                    for i in range(len(parts) - 1):
                        original_part_index = accumulated_text.find(parts[i])
                        if original_part_index != -1:
                            split_point = original_part_index + len(parts[i])
                            match = re.match(r'[.?!]', accumulated_text[split_point:].strip())
                            punctuation = match.group(0) if match else ""
                            sentence_chunk = parts[i].strip() + punctuation
                        else:
                            sentence_chunk = parts[i].strip()

                        if sentence_chunk and len(sentence_chunk.split()) >= MIN_WORDS_FOR_AUDIO:
                            text_queue.put(sentence_chunk) # Put chunk in queue
                    accumulated_text = parts[-1] # Keep remainder
                elif len(accumulated_text) > MAX_BUFFER_LENGTH and len(accumulated_text.split()) >= MIN_WORDS_FOR_AUDIO:
                    break_indices = [m.start() for m in re.finditer(r'[,;:]\s+', accumulated_text)]
                    if break_indices:
                        last_break_pos = break_indices[-1] + 1
                        chunk_to_send = accumulated_text[:last_break_pos].strip()
                        if chunk_to_send and len(chunk_to_send.split()) >= MIN_WORDS_FOR_AUDIO:
                             text_queue.put(chunk_to_send)
                             accumulated_text = accumulated_text[last_break_pos:]

        # After the loop, process any remaining text
        final_chunk = accumulated_text.strip()
        if final_chunk and len(final_chunk.split()) >= MIN_WORDS_FOR_AUDIO:
            text_queue.put(final_chunk)

    except Exception as e:
        print(f"\nError during LLM streaming or text processing: {e}")
    finally:
        # 6. Signal Generation End and Wait for Audio (Keep as before)
        print("\nLLM stream finished.")
        generation_done.set()
        print("Waiting for audio generation and playback to complete...")
        start_wait_time = time.time()
        if audio_generator_thread and audio_generator_thread.is_alive():
             print("Waiting for text queue to empty...")
             text_queue.join()
             print("Text queue empty.")
        print("Waiting for audio queue to empty...")
        audio_queue.join()
        print("Audio queue empty.")
        wait_timeout = 5
        if audio_generator_thread and audio_generator_thread.is_alive():
             print("Waiting for audio generator thread to exit...")
             audio_generator_thread.join(timeout=wait_timeout)
             if audio_generator_thread.is_alive(): print("Warning: Audio generator thread timed out.")
        if audio_player_thread.is_alive():
             print("Waiting for audio player thread to exit...")
             audio_player_thread.join(timeout=wait_timeout)
             if audio_player_thread.is_alive(): print("Warning: Audio player thread timed out.")
        end_wait_time = time.time()
        print(f"Audio processing finished in {end_wait_time - start_wait_time:.2f} seconds.")
        print(f"--- Scene description complete for: {image_path} ---")

# --- Main Execution Guard ---
# if __name__ == "__main__":
#     # --- Initiate TTS model once ---
#     # This will now use the caching logic within initiate_tts_model
#     # Set desired_device='cuda' to prefer GPU, 'cpu' otherwise.
#     tts_pipeline_main = initiate_tts_model(desired_device='cuda')

#     # --- Proceed only if TTS model loaded successfully ---
#     if tts_pipeline_main is None:
#         print("Exiting because TTS model failed to load and is required.")
#     else:
#         # Define the image path
#         IMAGE_PATH_INPUT = "C:/Users/Rohith.MR/PerceptaAI/test.jpg" # Use forward slashes or raw string
#         if not os.path.exists(IMAGE_PATH_INPUT):
#             print(f"Error: Image file not found at {IMAGE_PATH_INPUT}")
#         else:
#             # Call the main function, passing the initialized (or cached) TTS pipeline
#             sceene_description_with_tts(IMAGE_PATH_INPUT, tts_pipeline_main)

#             # Example: Call again to see caching in action (optional)
#             # print("\n--- Running again to test TTS caching ---")
#             # other_image_path = "path/to/another/image.jpg"
#             # if os.path.exists(other_image_path):
#             #      sceene_description_with_tts(other_image_path, tts_pipeline_main)
#             # else:
#             #      print(f"Skipping second run, image not found: {other_image_path}")