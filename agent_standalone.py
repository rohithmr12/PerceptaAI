#!/usr/bin/env python3
"""
PerceptaAI Standalone Agent
Direct module calls - no server dependencies, no LangChain tools.
"""

# System imports
import os
import sys
import time
import threading
import queue
import re
from typing import Optional, Dict, Any, List
import json

print("🚀 Starting PerceptaAI Standalone Agent...")

# Core ML libraries
import torch
import whisper
import numpy as np
import cv2  # Add cv2 import for camera functionality

# Audio libraries
import sounddevice as sd
import pyaudio
import keyboard

# Import existing modules directly
from Sceene_description import sceene_description_with_tts, stream_text_with_tts
from OCR import ocr_text
from nav_core import NavigationCore
from General_Navigation import VisionAssistant
from utils.model import initiate_tts_model, get_optimal_device

# LangChain for LLM only
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Audio settings
AUDIO_SAMPLE_RATE = 22050
AUDIO_PLAYBACK_PAUSE = 0.02
CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
PUSH_TO_TALK_KEY = 'space'
MIN_RECORD_TIME = 0.3
MAX_RECORD_TIME = 20.0

# Agent Configuration
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL_IDENTIFIER = "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF/llama-3.2-3b-instruct-q8_0.gguf"
os.environ["OPENAI_API_KEY"] = "lm-studio"

# Global variables
whisper_model = None
tts_pipeline = None
llm = None
vision_assistant = None
navigation_thread = None

def check_and_initialize_models():
    """Fast model initialization"""
    global whisper_model, tts_pipeline, llm, vision_assistant, navigation_thread
    
    print("\n⚡ Fast Model Initialization")
    print("=" * 40)
    
    device = get_optimal_device()
    
    # Load Whisper
    print("🔄 Loading Whisper model...")
    whisper_model = whisper.load_model("small", device=device)
    print("✅ Whisper ready!")
    
    # Load TTS
    print("🔄 Loading TTS model...")
    tts_pipeline = initiate_tts_model(force_cpu=(device=="cpu"))
    print("✅ TTS ready!")
    
    # Initialize LLM
    print("🔄 Initializing LLM connection...")
    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        model=LM_STUDIO_MODEL_IDENTIFIER,
        temperature=0.1,
        streaming=True,
        request_timeout=15,
        max_retries=1
    )
    print("✅ LLM ready!")
    
    # Initialize Vision Assistant (for general navigation)
    print("🔄 Initializing Vision Assistant...")
    try:
        vision_assistant = VisionAssistant(force_headless=False)  # Use fast primitive TTS
        print("✅ Vision Assistant ready!")
    except Exception as e:
        print(f"⚠ Vision Assistant initialization failed: {e}")
        vision_assistant = None
    
    print(f"🎉 All models ready on {device.upper()}!")
    return True

# ===== DIRECT MODULE FUNCTIONS =====

def run_scene_description():
    """Run scene description using Sceene_description.py"""
    try:
        print("🔧 Running scene description...")
        sceene_description_with_tts(tts_pipeline)
        return "Scene description completed"
    except Exception as e:
        error_msg = f"Scene description failed: {e}"
        print(f"❌ {error_msg}")
        try:
            stream_text_with_tts(error_msg, tts_pipeline)
        except:
            print("Could not speak error message")
        return error_msg

def run_ocr(image_path: str = None, preprocessing: str = "enhanced", detect_regions: bool = True):
    """Run OCR using OCR.py and send results to LLM for analysis"""
    try:
        print("🔧 Running enhanced OCR...")
        
        # Use default test image if none provided, or capture a new one
        if not image_path:
            # Try to capture an image first
            try:
                from utils.snap_a_picture import capture_image
                image_path = capture_image()
                print(f"📷 Captured image: {image_path}")
            except:
                # Fallback to test image
                image_path = "test.png"
                if not os.path.exists(image_path):
                    error_msg = "No image available for OCR. Please ensure test.png exists or camera is working."
                    try:
                        stream_text_with_tts(error_msg, tts_pipeline)
                    except:
                        print(f"📄 {error_msg}")
                    return error_msg
        
        # Run OCR with enhanced preprocessing
        print("🔍 Extracting text with advanced preprocessing...")
        result = ocr_text(image_path, preprocessing=preprocessing, detect_regions=detect_regions)
        
        if result and result.strip() and not result.startswith("Error:") and not result.startswith("OCR Error:"):
            print(f"📄 Raw OCR Result: {result}")
            
            # Send OCR result to LLM for analysis and interpretation
            try:
                print("🤖 Sending OCR text to LLM for analysis...")
                
                # Create a prompt for the LLM to analyze the OCR text
                ocr_prompt = f"""Please analyze this text that was extracted from an image using OCR. 
                
Text extracted: "{result}"

Please provide a helpful summary or explanation of this text. Consider:
- What type of document or content this appears to be
- Key information or important details
- Any actions the user might want to take based on this text
- Make your response conversational and concise (2-3 sentences)

If the text seems incomplete or has OCR errors, mention that and provide your best interpretation."""

                messages = [HumanMessage(content=ocr_prompt)]
                llm_response = llm.invoke(messages)
                
                if llm_response.content:
                    analysis = llm_response.content.strip()
                    print(f"💬 LLM Analysis: {analysis}")
                    
                    # Speak the LLM analysis instead of just the raw OCR text
                    try:
                        stream_text_with_tts(analysis, tts_pipeline)
                    except Exception as tts_error:
                        print(f"TTS error: {tts_error}")
                        print(f"📄 {analysis}")
                    
                    return f"OCR Text: {result}\n\nAnalysis: {analysis}"
                else:
                    # Fallback to just OCR text if LLM fails
                    response_text = f"Text found: {result}"
                    try:
                        stream_text_with_tts(response_text, tts_pipeline)
                    except:
                        print(f"📄 {response_text}")
                    return result
                    
            except Exception as llm_error:
                print(f"⚠ LLM analysis failed: {llm_error}")
                # Fallback to original behavior
                response_text = f"Text found: {result}"
                try:
                    stream_text_with_tts(response_text, tts_pipeline)
                except:
                    print(f"📄 {response_text}")
                return result
        else:
            no_text_msg = "No clear text found in the image. The image might be blurry, have poor lighting, or contain no readable text."
            try:
                stream_text_with_tts(no_text_msg, tts_pipeline)
            except:
                print(f"📄 {no_text_msg}")
            return no_text_msg
            
    except Exception as e:
        error_msg = f"OCR failed: {e}"
        print(f"❌ {error_msg}")
        try:
            stream_text_with_tts(error_msg, tts_pipeline)
        except:
            print("Could not speak error message")
        return error_msg

def run_indoor_navigation(destination: str, start: str = "node1", map_file: str = None, yolo_file: str = None):
    """Run indoor navigation using nav_core.py with camera feed and real-time guidance"""
    try:
        print(f"🔧 Starting indoor navigation: {start} → {destination}")
        
        # Set defaults
        if not map_file:
            map_file = "Nav/map_data/121-5-3.json"
        if not yolo_file:
            yolo_file = "Nav/yolov8m.pt"
        
        # Check map exists
        if not os.path.exists(map_file):
            error_msg = f"Map file not found: {map_file}"
            print(f"❌ {error_msg}")
            # Use Kokoro TTS as fallback for important messages
            try:
                stream_text_with_tts(error_msg, tts_pipeline)
            except:
                print(f"📄 Could not speak: {error_msg}")
            return error_msg
        
        # Initialize navigation
        nav_core = NavigationCore(
            map_filepath=map_file,
            start_node_id=start,
            end_node_id=destination,
            yolo_model_path=yolo_file
        )
        
        print(f"📍 Navigation path: {nav_core.path}")
        
        # Test if nav_core TTS is working
        print("🔊 Testing navigation TTS system...")
        test_msg = "Navigation TTS system is ready"
        if nav_core.feedback_manager.engine is not None:
            try:
                nav_core.feedback_manager.speak(test_msg)
                print("✅ Nav_core TTS working!")
            except Exception as tts_test_error:
                print(f"⚠️ Nav_core TTS failed: {tts_test_error}")
                print("🔄 Using Kokoro TTS as backup for instructions...")
                # Use Kokoro as fallback
                try:
                    stream_text_with_tts(test_msg, tts_pipeline)
                except:
                    print("❌ Both TTS systems failed!")
        else:
            print("⚠️ Nav_core TTS engine not available, using Kokoro backup...")
            try:
                stream_text_with_tts(test_msg, tts_pipeline)
            except:
                print("❌ Kokoro TTS also failed!")
        
        # Open a video capture object (main feature from nav_core.py)
        cap = cv2.VideoCapture(0)  # 0 for default camera
        
        if not cap.isOpened():
            error_msg = "Cannot open webcam for indoor navigation"
            print(f"❌ {error_msg}")
            # Try nav_core TTS first, then fallback to Kokoro
            try:
                nav_core.feedback_manager.speak(error_msg)
            except:
                try:
                    stream_text_with_tts(error_msg, tts_pipeline)
                except:
                    print(f"📄 Could not speak: {error_msg}")
            return error_msg
        
        print("🎥 Camera opened successfully. Starting real-time navigation...")
        print("🔄 Press 'q' in the camera window to quit navigation")
        
        # Announce start with working TTS
        start_msg = f"Starting indoor navigation from {start} to {destination}"
        try:
            nav_core.feedback_manager.speak(start_msg)
        except:
            try:
                stream_text_with_tts(start_msg, tts_pipeline)
            except:
                print(f"📄 {start_msg}")
        
        try:
            navigation_completed = False
            
            while not navigation_completed:
                # Read a frame from the camera (main feature from nav_core.py)
                ret, frame = cap.read()
                if not ret:
                    error_msg = "Failed to read camera frame"
                    print(f"❌ {error_msg}")
                    break
                
                # Run navigation step with camera feed (main feature from nav_core.py)
                try:
                    navigation_completed = nav_core.navigate(frame)
                    
                    if navigation_completed:
                        success_msg = "🎉 Navigation completed! You have reached your destination."
                        print(success_msg)
                        # Ensure user hears completion message
                        completion_msg = "Congratulations! You have successfully reached your destination."
                        try:
                            nav_core.feedback_manager.speak(completion_msg)
                        except:
                            try:
                                stream_text_with_tts(completion_msg, tts_pipeline)
                            except:
                                print(f"📄 {completion_msg}")
                        break
                        
                except Exception as nav_error:
                    print(f"⚠️ Navigation step error: {nav_error}")
                
                # Check for quit condition (main feature from nav_core.py)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    quit_msg = "Navigation stopped by user"
                    print(f"🛑 {quit_msg}")
                    stop_msg = "Navigation stopped."
                    try:
                        nav_core.feedback_manager.speak(stop_msg)
                    except:
                        try:
                            stream_text_with_tts(stop_msg, tts_pipeline)
                        except:
                            print(f"📄 {stop_msg}")
                    break
                elif key == ord('s'):  # Space to skip current instruction
                    skip_msg = "Skipping to next instruction..."
                    print(f"⏭️ {skip_msg}")
                    skip_speech = "Moving to next instruction."
                    try:
                        nav_core.feedback_manager.speak(skip_speech)
                    except:
                        try:
                            stream_text_with_tts(skip_speech, tts_pipeline)
                        except:
                            print(f"📄 {skip_speech}")
                
                # Delay between instructions (main feature from nav_core.py)
                time.sleep(nav_core.feedback_interval)
                
        except Exception as loop_error:
            error_msg = f"Navigation loop error: {loop_error}"
            print(f"❌ {error_msg}")
            error_speech = f"Navigation error occurred: {loop_error}"
            try:
                nav_core.feedback_manager.speak(error_speech)
            except:
                try:
                    stream_text_with_tts(error_speech, tts_pipeline)
                except:
                    print(f"📄 {error_speech}")
            
        finally:
            # Release the video capture object and close windows (main feature from nav_core.py)
            cap.release()
            cv2.destroyAllWindows()
            print("🏁 Indoor navigation session completed")
        
        if navigation_completed:
            return "Indoor navigation completed successfully"
        else:
            return "Indoor navigation session ended"
        
    except Exception as e:
        error_msg = f"Navigation failed: {e}"
        print(f"❌ {error_msg}")
        # Use Kokoro TTS for critical errors
        try:
            stream_text_with_tts(error_msg, tts_pipeline)
        except:
            print("Could not speak error message")
        return error_msg

def run_outdoor_navigation():
    """Run outdoor navigation using outdoor_navigation.py"""
    try:
        print("🔧 Running outdoor navigation...")
        # Import outdoor navigation if it exists
        try:
            from outdoor_navigation import start_outdoor_navigation
            result = start_outdoor_navigation()
            success_msg = "Outdoor navigation started"
            try:
                stream_text_with_tts(success_msg, tts_pipeline)
            except:
                print(f"📄 {success_msg}")
            return success_msg
        except ImportError:
            error_msg = "Outdoor navigation module not available"
            try:
                stream_text_with_tts(error_msg, tts_pipeline)
            except:
                print(f"📄 {error_msg}")
            return error_msg
    except Exception as e:
        error_msg = f"Outdoor navigation failed: {e}"
        print(f"❌ {error_msg}")
        try:
            stream_text_with_tts(error_msg, tts_pipeline)
        except:
            print("Could not speak error message")
        return error_msg

def run_general_navigation():
    """Run general navigation using General_Navigation.py"""
    global navigation_thread
    
    try:
        print("🔧 Starting general navigation...")
        
        if vision_assistant is None:
            error_msg = "Vision Assistant not available"
            try:
                stream_text_with_tts(error_msg, tts_pipeline)
            except:
                print(f"📄 {error_msg}")
            return error_msg
        
        # Check if navigation is already running
        if navigation_thread and navigation_thread.is_alive():
            already_running_msg = "General navigation is already running. Say 'stop navigation' to stop it."
            try:
                stream_text_with_tts(already_running_msg, tts_pipeline)
            except:
                print(f"📄 {already_running_msg}")
            return already_running_msg
        
        # Start navigation in background thread
        def run_navigation():
            try:
                vision_assistant.running = True
                vision_assistant.run()
            except Exception as e:
                error_msg = f"General navigation error: {e}"
                print(error_msg)
                try:
                    stream_text_with_tts(error_msg, tts_pipeline)
                except:
                    print("Could not speak navigation error")
        
        navigation_thread = threading.Thread(target=run_navigation, daemon=True)
        navigation_thread.start()
        
        # Check if GUI is available and inform user
        gui_status = "with display window" if not vision_assistant.headless_mode else "in headless mode"
        success_msg = f"General navigation started {gui_status}. Say 'stop navigation' to stop it."
        try:
            stream_text_with_tts(success_msg, tts_pipeline)
        except:
            print(f"📄 {success_msg}")
        return success_msg
        
    except Exception as e:
        error_msg = f"General navigation failed: {e}"
        print(f"❌ {error_msg}")
        try:
            stream_text_with_tts(error_msg, tts_pipeline)
        except:
            print("Could not speak error message")
        return error_msg

def stop_general_navigation():
    """Stop general navigation"""
    global navigation_thread
    
    try:
        print("🔧 Stopping general navigation...")
        
        if vision_assistant is None:
            error_msg = "Vision Assistant not available"
            try:
                stream_text_with_tts(error_msg, tts_pipeline)
            except:
                print(f"📄 {error_msg}")
            return error_msg
        
        # Stop the vision assistant
        vision_assistant.stop()
        
        # Wait for thread to finish (with timeout)
        if navigation_thread and navigation_thread.is_alive():
            navigation_thread.join(timeout=3)
        
        success_msg = "General navigation stopped"
        try:
            stream_text_with_tts(success_msg, tts_pipeline)
        except:
            print(f"📄 {success_msg}")
        return success_msg
        
    except Exception as e:
        error_msg = f"Failed to stop general navigation: {e}"
        print(f"❌ {error_msg}")
        try:
            stream_text_with_tts(error_msg, tts_pipeline)
        except:
            print("Could not speak error message")
        return error_msg

# ===== AUDIO PROCESSING =====

def record_audio():
    """Record audio while push-to-talk key is held"""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("\n🎤 Recording... (Release SPACE to process)")
    frames = []
    start_time = time.time()
    
    try:
        while keyboard.is_pressed(PUSH_TO_TALK_KEY):
            if time.time() - start_time > MAX_RECORD_TIME:
                print("⏰ Maximum recording time reached")
                break
                
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
        recording_time = time.time() - start_time
        if recording_time < MIN_RECORD_TIME:
            while recording_time < MIN_RECORD_TIME:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                recording_time = time.time() - start_time
        
        if len(frames) > 0:
            print(f"📹 Recorded {len(frames)} frames ({recording_time:.1f}s)")
            return b''.join(frames)
        else:
            return b''
            
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def transcribe_audio(audio_data):
    """Convert audio to text using Whisper"""
    if not audio_data:
        return ""
    
    try:
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        print("🧠 Transcribing...")
        result = whisper_model.transcribe(
            audio_float,
            language="en",
            temperature=0.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
        )
        
        return result["text"].strip()
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

# ===== LLM PROCESSING =====

def process_user_input(user_input: str):
    """Process user input with keyword detection and direct module calls"""
    
    try:
        print("🤖 Processing user request...")
        user_lower = user_input.lower()
        
        # Stop navigation keywords (check first)
        if any(phrase in user_lower for phrase in ['stop navigation', 'stop general navigation', 'stop vision']):
            print("🔧 Detected stop navigation request")
            try:
                return stop_general_navigation()
            except Exception as e:
                error_msg = f"Stop navigation error: {e}"
                print(f"❌ {error_msg}")
                return error_msg
        
        # Scene description keywords
        if any(phrase in user_lower for phrase in ['what', 'around', 'see', 'describe', 'surroundings', 'scene']):
            print("🔧 Detected scene description request")
            try:
                return run_scene_description()
            except Exception as e:
                error_msg = f"Scene description module error: {e}"
                print(f"❌ {error_msg}")
                return error_msg
            
        # OCR keywords  
        elif any(phrase in user_lower for phrase in ['read', 'text', 'sign', 'ocr']):
            print("🔧 Detected text reading request")
            try:
                return run_ocr()
            except Exception as e:
                error_msg = f"OCR module error: {e}"
                print(f"❌ {error_msg}")
                return error_msg
        
        # General navigation keywords (check before indoor to be more specific)
        elif any(phrase in user_lower for phrase in ['general navigation', 'vision navigation', 'camera navigation', 'live navigation', 'real time navigation']):
            print("🔧 Detected general navigation request")
            try:
                return run_general_navigation()
            except Exception as e:
                error_msg = f"General navigation module error: {e}"
                print(f"❌ {error_msg}")
                return error_msg
            
        # Indoor navigation keywords
        elif any(phrase in user_lower for phrase in ['navigate', 'navigation', 'go', 'directions', 'guide', 'indoor']):
            print("🔧 Detected indoor navigation request")
            
            # Extract destination
            destination = "node5"  # Default
            if "kitchen" in user_lower:
                destination = "kitchen"
            elif "bathroom" in user_lower:
                destination = "bathroom"
            elif "office" in user_lower:
                destination = "office"
            elif "elevator" in user_lower:
                destination = "520-C"
            
            try:
                return run_indoor_navigation(destination=destination)
            except Exception as e:
                error_msg = f"Navigation module error: {e}"
                print(f"❌ {error_msg}")
                return error_msg
            
        # Outdoor navigation keywords
        elif any(phrase in user_lower for phrase in ['outdoor', 'outside', 'gps']):
            print("🔧 Detected outdoor navigation request")
            try:
                return run_outdoor_navigation()
            except Exception as e:
                error_msg = f"Outdoor navigation module error: {e}"
                print(f"❌ {error_msg}")
                return error_msg
            
        else:
            # Use LLM for general conversation (optional)
            try:
                print("🤖 Using LLM for general response...")
                messages = [HumanMessage(content=f"Please provide a helpful response to: {user_input}")]
                response = llm.invoke(messages)
                
                if response.content:
                    print(f"💬 LLM: {response.content}")
                    try:
                        stream_text_with_tts(response.content, tts_pipeline)
                    except Exception as tts_error:
                        print(f"TTS error: {tts_error}")
                        print(f"📄 {response.content}")
                    return response.content
            except Exception as llm_error:
                print(f"⚠ LLM error: {llm_error}")
            
            # Fallback response
            default_msg = "I can help you with: describing scenes, reading text, indoor navigation, general navigation, or outdoor navigation. What would you like to do?"
            print(f"💬 Assistant: {default_msg}")
            try:
                stream_text_with_tts(default_msg, tts_pipeline)
            except:
                print(f"📄 {default_msg}")
            return default_msg
        
    except Exception as e:
        error_msg = f"Error processing request: {e}"
        print(f"❌ {error_msg}")
        try:
            stream_text_with_tts(error_msg, tts_pipeline)
        except:
            print("Could not speak error message")
        return error_msg

# ===== MAIN AGENT LOOP =====

def run_interactive_agent():
    """Main interactive agent loop"""
    print("\n" + "=" * 60)
    print("🎯 PerceptaAI Standalone Agent Ready!")
    print("=" * 60)
    print("🎤 Hold SPACE to talk, release to process")
    print("🗣 Say 'quit' or 'exit' to stop")
    print("📱 Example commands:")
    print("   • 'What's around me?' → Scene description")
    print("   • 'Read that text' → OCR") 
    print("   • 'Navigate to the kitchen' → Indoor navigation")
    print("   • 'General navigation' → Live camera navigation")
    print("   • 'Stop navigation' → Stop general navigation")
    print("   • 'Outdoor navigation' → GPS navigation")
    print("-" * 60)
    
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    try:
        while True:
            try:
                # Wait for push-to-talk
                print("\n⏳ Press and hold SPACE to speak...")
                keyboard.wait(PUSH_TO_TALK_KEY)
                
                # Record audio
                audio_data = record_audio()
                
                if len(audio_data) == 0:
                    print("❌ No audio recorded")
                    continue
                
                # Transcribe
                user_input = transcribe_audio(audio_data)
                
                if not user_input:
                    print("❌ No speech detected")
                    continue
                
                print(f"👤 You said: {user_input}")
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'stop', 'goodbye']:
                    print("👋 Goodbye!")
                    try:
                        stream_text_with_tts("Goodbye! See you next time.", tts_pipeline)
                    except:
                        pass
                    break
                
                # Process with direct module calls
                try:
                    response = process_user_input(user_input)
                    consecutive_errors = 0  # Reset error count on success
                except Exception as e:
                    consecutive_errors += 1
                    print(f"❌ Processing error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        error_msg = f"Too many consecutive errors ({max_consecutive_errors}). Please check the system and try again."
                        print(f"⚠ {error_msg}")
                        try:
                            stream_text_with_tts(error_msg, tts_pipeline)
                        except:
                            pass
                        consecutive_errors = 0  # Reset and continue
                        
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as loop_error:
                consecutive_errors += 1
                print(f"❌ Loop error ({consecutive_errors}/{max_consecutive_errors}): {loop_error}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("⚠ Too many errors, restarting agent loop...")
                    consecutive_errors = 0
                    time.sleep(2)  # Brief pause before continuing
                
    except Exception as e:
        print(f"❌ Fatal error in agent loop: {e}")
        try:
            stream_text_with_tts("I encountered a critical error. Please restart me.", tts_pipeline)
        except:
            pass

def main():
    """Main entry point"""
    try:
        # Initialize models
        if not check_and_initialize_models():
            print("❌ Model initialization failed")
            sys.exit(1)
        
        # Run agent
        run_interactive_agent()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 