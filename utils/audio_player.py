# utils/audio_player.py
import time
import sounddevice as sd
import queue
import numpy as np
import torch
import threading # Import threading for Event type hint if needed

def audio_player(
    audio_queue: queue.Queue,
    generation_done: threading.Event,
    text_ready: threading.Event,
    audio_playing: threading.Event,
    sample_rate: int,
    playback_pause: float,
    player_check_interval: float,
    chunk_timeout: float
):
    """Plays audio chunks from the audio_queue."""
    print("Audio player thread started")
    while not (generation_done.is_set() and audio_queue.empty()):
        try:
            # Wait briefly for text_ready if the queue is empty,
            # allowing the generator time to produce audio.
            if not text_ready.is_set() and audio_queue.empty():
                # Wait with a timeout; text_ready might be set by the generator
                signaled = text_ready.wait(timeout=chunk_timeout)
                if not signaled and audio_queue.empty(): # If timeout occurred and queue still empty
                     if generation_done.is_set(): # Check if generation is done before continuing loop
                          break # Exit loop if generation is done and queue is empty
                     continue # Otherwise, continue waiting

            # Try getting audio data without blocking indefinitely, using a timeout
            audio_data = audio_queue.get(block=True, timeout=player_check_interval)
            audio_playing.set() # Signal that we are about to play audio

            # Ensure data is numpy float32 for sounddevice
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            if not isinstance(audio_data, np.ndarray) or audio_data.dtype != np.float32:
                 audio_data = np.asarray(audio_data, dtype=np.float32)

            if audio_data.size == 0:
                print("Warning: Skipping empty audio chunk.")
                audio_queue.task_done()
                audio_playing.clear() # Clear flag if chunk was empty
                continue

            # print(f"Playing audio chunk of shape: {audio_data.shape}") # Debug
            sd.play(audio_data, sample_rate)
            sd.wait() # Wait for the current chunk to finish playing
            audio_queue.task_done()
            time.sleep(playback_pause) # Small pause between chunks

        except queue.Empty:
            # Queue was empty after waiting, clear playing flag and loop again
            audio_playing.clear()
            # Check if generation is done while queue is empty
            if generation_done.is_set():
                 break # Exit loop
        except sd.PortAudioError as pae:
            print(f"Sounddevice error: {pae}. Ensure audio output device is available/configured.")
            # Clear flags and potentially break
            audio_playing.clear()
            break # Exit thread on significant device error
        except Exception as e:
            print(f"Error in audio player: {e}")
            # Clear playing flag in case of error during playback
            audio_playing.clear()
            # Attempt to clear the problematic item from the queue if possible
            # Avoid deadlocks by ensuring task_done is called if an item was retrieved
            try:
                # Check if an item was actually retrieved before the exception
                # This is tricky; safer to just ensure flag is clear and potentially break/log
                pass
            except ValueError: pass # If task_done() was already called or item not gotten

    audio_playing.clear() # Ensure flag is clear on exit
    print("Audio player thread exiting")