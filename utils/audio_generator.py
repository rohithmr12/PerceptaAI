# utils/audio_generator.py
import torch
import queue
import time
import numpy as np
from kokoro import KPipeline # Assuming KPipeline is the correct class
import threading # Import threading for Event type hint if needed

def audio_generator(
    text_queue: queue.Queue,
    audio_queue: queue.Queue,
    generation_done: threading.Event,
    text_ready: threading.Event,
    tts_pipeline: KPipeline, # Pass the initialized pipeline object
    tts_voice: str,
    min_words_for_audio: int # Pass config value
):
    """Generates audio from text chunks using the TTS model."""
    print("Audio generator thread started")
    if tts_pipeline is None:
        print("TTS model not available. Audio generator thread exiting.")
        return

    while not (generation_done.is_set() and text_queue.empty()):
        try:
            # Get text chunk, wait if necessary but with a timeout
            text_chunk = text_queue.get(block=True, timeout=0.1) # Wait briefly
            # print(f"Audio generator processing: '{text_chunk}'") # Debug

            # Basic check to avoid generating audio for very short/empty strings
            words = text_chunk.split()
            if not text_chunk or len(words) < min_words_for_audio:
                # print(f"Skipping chunk (too short/empty): '{text_chunk}'") # Debug
                text_queue.task_done()
                continue

            # Generate audio using Kokoro TTS pipeline
            # The pipeline might yield multiple results; process the audio part
            generated = False
            # Make sure to handle potential errors during TTS generation itself
            try:
                for _, _, audio in tts_pipeline(text_chunk, voice=tts_voice):
                    if audio is not None and (isinstance(audio, torch.Tensor) or isinstance(audio, np.ndarray)):
                        # Convert tensor to numpy array if needed
                        if isinstance(audio, torch.Tensor):
                            # Ensure tensor is on CPU before converting to numpy
                            audio = audio.detach().cpu().numpy()

                        # Ensure float32 for playback
                        audio_float = audio.astype(np.float32)

                        if audio_float.size > 0: # Ensure there's actual audio data
                            audio_queue.put(audio_float)
                            text_ready.set() # Signal that audio is ready for playing
                            text_ready.clear() # Reset signal immediately after setting? Or let player clear? Player loop usually just waits. Setting seems enough.
                            generated = True
                        # else:
                            # print(f"Warning: TTS generated empty audio for '{text_chunk}'") # Debug
            except Exception as tts_error:
                 print(f"Error during TTS generation for chunk '{text_chunk}': {tts_error}")
                 # Decide if you want to skip the chunk or retry

            # if not generated:
                # print(f"Warning: TTS did not generate audio for chunk: '{text_chunk}'") # Debug

            text_queue.task_done()

        except queue.Empty:
            # No text chunk available, loop again
            time.sleep(0.01) # Small sleep to prevent busy-waiting
        except Exception as e:
            print(f"Error in audio generator loop: {e}")
            # Ensure task_done is called even if there's an error processing this chunk
            # This might happen if get() succeeded but processing failed before task_done()
            try: text_queue.task_done()
            except ValueError: pass # If task_done() was already called

    print("Audio generator thread exiting")