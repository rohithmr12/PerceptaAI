# System imports
import os
from typing import Annotated, TypedDict, List, Optional
import requests # Added for API calls
import json # Added for payload construction
import sounddevice as sd # For audio recording
import numpy as np # For audio processing
import keyboard # For push-to-talk functionality
import speech_recognition as sr # For speech-to-text
import threading # For managing audio recording
import queue # For audio data handling
import pyaudio # For audio input
import wave # For audio file handling
import time # For timing operations
import re # For regular expressions
import whisper
import torch

# Import the scene description implementation
from Sceene_description import sceene_description_with_tts, stream_text_with_tts
from utils.model import initiate_tts_model
from utils.audio_generator import audio_generator
from utils.audio_player import audio_player

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

# Audio settings
AUDIO_SAMPLE_RATE = 24000  # Sample rate for TTS audio
AUDIO_PLAYBACK_PAUSE = 0.1  # Short pause between audio chunks
AUDIO_PLAYER_CHECK_INTERVAL = 0.01  # How often to check for new audio
AUDIO_CHUNK_TIMEOUT = 0.5  # Timeout for waiting for audio chunks
MIN_WORDS_FOR_AUDIO = 3  # Minimum words needed to generate audio
TTS_VOICE = "default"  # Default voice for TTS
TEMP_AUDIO_FILE = os.path.abspath(os.path.join(os.getcwd(), "temp_recording.wav"))
CHUNK = 1024 * 4  # Larger chunk size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Keep 16kHz for Whisper
PUSH_TO_TALK_KEY = 'space'
MIN_RECORD_TIME = 0.5  # Minimum recording time in seconds
MAX_RECORD_TIME = 30.0  # Maximum recording time in seconds
audio_queue = queue.Queue()
recording = False

# --- Agent Configuration ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL_IDENTIFIER = "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF/llama-3.2-3b-instruct-q8_0.gguf"
os.environ["OPENAI_API_KEY"] = "lm-studio" # Required by ChatOpenAI, but not used by LM Studio

# LM Studio connection settings
LM_STUDIO_TIMEOUT = 10  # Timeout in seconds
LM_STUDIO_MAX_RETRIES = 1  # Minimize retries for faster failure
LM_STUDIO_REQUEST_TIMEOUT = 30  # Request timeout in seconds

PERCEPTA_SERVER_BASE_URL = "http://localhost:8000" # For calling our FastAPI server

def get_optimal_device():
    """Get the best available compute device with proper availability checks"""
    try:
        # First try Intel NPU via ONNX Runtime
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "DmlExecutionProvider" in providers:
                # Test if NPU (adapter 0) is available
                test_session = ort.InferenceSession(
                    "dummy",  # This won't actually be used for the test
                    providers=["DmlExecutionProvider"],
                    provider_options=[{"device_id": 0}]  # NPU is usually adapter 0
                )
                print("✓ Intel NPU (DirectML) is available")
                return "dml"  # Use DirectML provider
        except Exception as e:
            print(f"! Intel NPU test failed: {e}")
        
        # Then try CUDA
        if torch.cuda.is_available():
            try:
                # Test CUDA by creating a small tensor
                test_tensor = torch.tensor([1.0], device='cuda')
                del test_tensor  # Clean up test tensor
                cuda_device_name = torch.cuda.get_device_name(0)
                print(f"✓ CUDA is available and working ({cuda_device_name})")
                return "cuda"
            except Exception as e:
                print(f"! CUDA found but test failed: {e}")
        
        # CPU is always available as fallback
        print("✓ Using CPU (no GPU/NPU acceleration available)")
        return "cpu"
        
    except Exception as e:
        print(f"! Error checking device availability: {e}")
        print("✓ Defaulting to CPU for safety")
        return "cpu"

def initialize_models():
    """Initialize all models and verify they're loaded correctly"""
    print("\n=== Initializing Models ===")
    
    # Get optimal device
    device = get_optimal_device()
    print(f"Using device: {device}")
    
    try:
        # Initialize Whisper
        print("\nInitializing Whisper model (small)...")
        whisper_model = whisper.load_model("small", device=device)
        print("✓ Whisper model loaded successfully")
        
        # Initialize TTS model
        print("\nInitializing TTS model...")
        tts_pipeline = initiate_tts_model(desired_device=device)
        if tts_pipeline is None:
            raise RuntimeError("Failed to initialize TTS model")
        print("✓ TTS model loaded successfully")
        
        # Test LM Studio connection
        print("\nTesting LM Studio connection...")
        response = requests.get(LM_STUDIO_BASE_URL + "/models")
        if response.status_code != 200:
            raise ConnectionError(f"LM Studio server not responding (status {response.status_code})")
        print("✓ LM Studio connection verified")
        
        print("\nAll models initialized successfully! 🚀\n")
        return whisper_model, tts_pipeline
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Initialize all models at startup
whisper_model, tts_pipeline = initialize_models()

def audio_callback(indata, frames, time, status):
    """This is called for each audio block"""
    if recording:
        audio_queue.put(bytes(indata))

def record_audio():
    """Record audio while the push-to-talk key is held"""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("\nRecording... (Release SPACE to process)")
    frames = []
    start_time = time.time()
    
    try:
        # Record until key is released or max time reached
        while keyboard.is_pressed(PUSH_TO_TALK_KEY):
            if time.time() - start_time > MAX_RECORD_TIME:
                print("Maximum recording time reached")
                break
                
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
        # Check if we have enough audio
        recording_time = time.time() - start_time
        if recording_time < MIN_RECORD_TIME:
            # Record a bit more to meet minimum time
            while recording_time < MIN_RECORD_TIME:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                recording_time = time.time() - start_time
            
        if len(frames) > 0:
            print(f"Recorded {len(frames)} audio frames ({recording_time:.1f} seconds)")
            return b''.join(frames)
        else:
            print("No audio recorded")
            return b''
            
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    return b''.join(frames)

def transcribe_audio(audio_data):
    """Convert audio to text using Whisper"""
    if not audio_data:
        return ""
        
    try:
        # Convert audio data to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        # Convert to float32 and normalize to [-1, 1]
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # Transcribe using Whisper with optimal settings
        print("Starting Whisper transcription...")
        result = whisper_model.transcribe(
            audio_float,
            language="en",  # Specify English for better accuracy
            temperature=0.0,  # Use greedy decoding
            no_speech_threshold=0.6,  # Higher threshold for detecting speech
            condition_on_previous_text=True,  # Use context from previous transcriptions
            initial_prompt="This is a voice command for an AI assistant.",  # Give context
            compression_ratio_threshold=2.4,  # More aggressive filtering of repetitions
            logprob_threshold=-1.0,  # Stricter threshold for word selection
        )
        transcribed_text = result["text"].strip()
        print(f"Whisper transcription complete: {transcribed_text}")
        
        return transcribed_text
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

# --- Helper for API calls ---
def _call_percepta_server_tool(endpoint: str, payload: Optional[dict] = None, files: Optional[dict] = None) -> str:
    url = f"{PERCEPTA_SERVER_BASE_URL}{endpoint}"
    try:
        if files:
            response = requests.post(url, files=files, data=payload)
        elif payload:
            response = requests.post(url, json=payload)
        else:
            response = requests.post(url)

        response.raise_for_status()
        
        try:
            res_json = response.json()
            if "result" in res_json:
                return str(res_json["result"])
            elif "description" in res_json:
                return res_json["description"]
            elif "text" in res_json:
                return res_json["text"]
            elif "message" in res_json:
                return res_json["message"]
            return json.dumps(res_json)
        except json.JSONDecodeError:
            return f"Server returned non-JSON response: {response.text}"
            
    except requests.exceptions.HTTPError as http_err:
        error_detail = http_err.response.text
        try:
            error_json = http_err.response.json()
            if error_json and "detail" in error_json:
                error_detail = error_json["detail"]
        except json.JSONDecodeError:
            pass
        return f"API Call Failed ({http_err.response.status_code}): {error_detail}"
    except requests.exceptions.RequestException as req_err:
        return f"API Request Error: {req_err}"
    except Exception as e:
        return f"Unexpected error calling tool endpoint {endpoint}: {e}"

# --- Tool Definitions ---
@tool
def describe_current_scene(image_path: Optional[str] = None) -> str:
    """
    Captures an image of the current surroundings and describes it using an AI model.
    Uses the complete implementation from Sceene_description.py.
    Args:
        image_path (Optional[str]): Path to an image file. If None, captures a new image.
    """
    print(f"Tool: describe_current_scene called.")
    try:
        # Use the complete implementation with TTS
        sceene_description_with_tts(tts_pipeline)
        return "Scene description complete"
    except Exception as e:
        error_msg = f"Error in scene description: {e}"
        print(error_msg)
        return error_msg

@tool
def read_text_from_image(image_path: Optional[str] = None, preprocessing_type: str = "default", detect_regions: bool = False) -> str:
    """
    Reads text from an image.
    Args:
        image_path (Optional[str]): Path to the image file. If None, captures a new image.
        preprocessing_type (str): Type of preprocessing for OCR.
        detect_regions (bool): Whether to detect and process distinct text regions.
    """
    print(f"Tool: read_text_from_image called. Image: {image_path}, Preprocessing: {preprocessing_type}, Detect Regions: {detect_regions}")
    
    payload = {
        'preprocessing_type': preprocessing_type,
        'detect_regions': detect_regions
    }
    
    if image_path:
        if not os.path.exists(image_path):
            return f"Error: Provided image_path does not exist: {image_path}"
        payload['image_path'] = image_path
    
    result = _call_percepta_server_tool("/tools/ocr", payload=payload)
    if result:
        # Use the existing TTS streaming implementation
        stream_text_with_tts(result, tts_pipeline)
    return result

@tool
def start_indoor_navigation_guidance(destination_node_id: str, start_node_id: str = "node1", map_file: Optional[str] = None, yolo_model_file: Optional[str] = None) -> str:
    """
    Initiates indoor navigation to a specified destination node.
    Args:
        destination_node_id (str): The ID or common name of the destination.
        start_node_id (str): Optional. The ID of the starting node.
        map_file (str): Optional. Path to the map data file.
        yolo_model_file (str): Optional. Path to the YOLO model file.
    """
    print(f"Tool: start_indoor_navigation_guidance. Dest: {destination_node_id}, Start: {start_node_id}, Map: {map_file}, YOLO: {yolo_model_file}")
    
    payload = {
        "destination_node_id": destination_node_id,
        "start_node_id": start_node_id
    }
    if map_file:
        payload["map_file"] = map_file
    if yolo_model_file:
        payload["yolo_model_file"] = yolo_model_file
        
    result = _call_percepta_server_tool("/tools/start_navigation", payload=payload)
    if result:
        # Use the existing TTS streaming implementation
        stream_text_with_tts(result, tts_pipeline)
    return result

# --- LLM Setup ---
tools_list = [describe_current_scene, read_text_from_image, start_indoor_navigation_guidance]

llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    model=LM_STUDIO_MODEL_IDENTIFIER,
    temperature=0.1,
    streaming=True,
    callbacks=[],  # No callbacks needed for now
    request_timeout=LM_STUDIO_REQUEST_TIMEOUT,  # Add timeout
    max_retries=LM_STUDIO_MAX_RETRIES,  # Limit retries
    model_kwargs={
        "stream": True,  # Ensure streaming is enabled
        "timeout": LM_STUDIO_TIMEOUT,  # Add model timeout
        "functions": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to an image file"},
                        "preprocessing_type": {"type": "string", "description": "Type of preprocessing for OCR"},
                        "detect_regions": {"type": "boolean", "description": "Whether to detect text regions"},
                        "destination_node_id": {"type": "string", "description": "Destination node ID for navigation"},
                        "start_node_id": {"type": "string", "description": "Starting node ID for navigation"},
                        "map_file": {"type": "string", "description": "Path to map file"},
                        "yolo_model_file": {"type": "string", "description": "Path to YOLO model file"}
                    }
                }
            } for tool in tools_list
        ]
    }
)

llm_with_tools = llm.bind_tools(tools_list)

def process_user_input(user_input: str):
    """Process user input and get response from LLM"""
    messages = [HumanMessage(content="""SYSTEM: You are an AI assistant with access to the following tools that you MUST use when appropriate:

1. describe_current_scene()
   - Purpose: Captures and describes what's currently around the user
   - Use when: User asks about their surroundings or what's around them
   - No parameters required for basic usage

2. read_text_from_image(image_path=None, preprocessing_type="default", detect_regions=False)
   - Purpose: Reads and extracts text from images or the current scene
   - Use when: User needs text read from their surroundings
   - Parameters are optional

3. start_indoor_navigation_guidance(destination_node_id, start_node_id="node1")
   - Purpose: Helps navigate indoor spaces with voice guidance
   - Use when: User needs navigation assistance
   - Requires at least destination_node_id

IMPORTANT: You MUST use these tools when relevant. Do not just describe things - use the tools to actually perceive and interact with the environment.

Examples:
- If user asks "what's around me?" -> Use describe_current_scene()
- If user asks "can you read that sign?" -> Use read_text_from_image()
- If user asks "help me get to the kitchen" -> Use start_indoor_navigation_guidance(destination_node_id="kitchen")""")]
    
    messages.append(HumanMessage(content=user_input))
    
    try:
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        print(f"\nAssistant: {response.content}")
        
        # Send response to TTS if it's not just a tool result
        if response.content and not response.content.startswith("Tool:"):
            stream_text_with_tts(response.content, tts_pipeline)
        
        return response.content
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"\nAssistant: {error_msg}")
        stream_text_with_tts(error_msg, tts_pipeline)
        return error_msg

# --- Main Interaction Function ---
def run_interactive_agent():
    print("\n🚀 Welcome to the Interactive Assistant Agent! 🚀")
    print("This agent now calls Percepta Server for tool execution.")
    print("Ensure Percepta Server (percepta_server.py) is running.")
    print("Hold SPACE to talk, release to process.")
    print(f"Using LLM: {LM_STUDIO_MODEL_IDENTIFIER} via {LM_STUDIO_BASE_URL}")
    print("\nPress SPACE to start speaking, release to process. Say 'quit' to exit.")
    
    try:
        while True:
            # Wait for push-to-talk key press
            keyboard.wait(PUSH_TO_TALK_KEY)
            
            # Record audio while key is held
            audio_data = record_audio()
            
            if len(audio_data) == 0:
                print("No audio recorded. Please try again.")
                continue
            
            print("Processing speech...")
            user_input = transcribe_audio(audio_data)
            
            if not user_input:
                print("No speech detected. Please try again.")
                continue
                
            print(f"You said: {user_input}")
            
            if user_input.lower() == "quit":
                print("Exiting agent...")
                break
                
            try:
                response = process_user_input(user_input)
                print(f"Assistant: {response}")
                
            except Exception as e:
                print(f"An error occurred: {e}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_interactive_agent()
