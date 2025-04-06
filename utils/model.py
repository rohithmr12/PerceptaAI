# utils/model.py

import os
import torch
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from kokoro import KPipeline # Assuming KPipeline is the correct class

# --- LLM Initiation ---
def initiate_llm(base_url, model_identifier, temperature):
    """Initializes the LangChain ChatOpenAI model for LM Studio."""
    print(f"Initializing LLM: {model_identifier}")
    try:
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"), # Use get for safety
            model=model_identifier,
            temperature=temperature,
            streaming=True, # Crucial for the streaming TTS application
            # callbacks=[StreamingStdOutCallbackHandler()], # Keep if console output is desired alongside TTS
        )
        print("LLM initialized successfully (streaming enabled).")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise # Re-raise the exception to indicate failure

# --- TTS Initiation with Caching and Device Check ---

# Global variable to cache the TTS instance
_tts_pipeline_instance = None

def initiate_tts_model(force_reload=False, desired_device='cuda'):
    """
    Initializes the Kokoro TTS model. Checks for an existing instance
    and its device before loading anew.

    Args:
        force_reload (bool): If True, ignores any cached instance and reloads.
        desired_device (str): Preferred device ('cuda' or 'cpu').

    Returns:
        KPipeline instance or None if loading fails.
    """
    global _tts_pipeline_instance
    print("Initiating TTS model...")

    # Determine target device based on preference and availability
    gpu_available = torch.cuda.is_available()
    if desired_device == 'cuda' and not gpu_available:
        print("Warning: CUDA desired but not available, falling back to CPU.")
        target_device = torch.device('cpu')
    elif desired_device == 'cuda' and gpu_available:
        target_device = torch.device('cuda')
    else:
        target_device = torch.device('cpu')
    print(f"Target device for TTS: {target_device}")

    # 1. Check for existing instance (if not forcing reload)
    if not force_reload and _tts_pipeline_instance is not None:
        print("Existing TTS pipeline instance found.")
        current_device = None
        try:
            # --- Check device of existing model (KPipeline specific) ---
            if hasattr(_tts_pipeline_instance, 'device'):
                 current_device = _tts_pipeline_instance.device
            elif hasattr(_tts_pipeline_instance, 'model') and hasattr(_tts_pipeline_instance.model, 'device'):
                 current_device = _tts_pipeline_instance.model.device # Common pattern if it wraps a model
            # Add other potential checks based on KPipeline structure if needed
            # --- End KPipeline specific checks ---

            if current_device:
                print(f"Existing instance device: {current_device}")
                if current_device == target_device:
                    print("Using existing TTS model instance (matches target device).")
                    return _tts_pipeline_instance
                else:
                    print(f"Device mismatch (Existing: {current_device}, Target: {target_device}). Re-initializing.")
                    # Force re-initialization by falling through
            else:
                 print("Could not determine device of existing instance. Assuming compatible and using existing.")
                 return _tts_pipeline_instance

        except Exception as e:
            print(f"Warning: Error checking device of existing TTS instance: {e}. Using existing instance.")
            return _tts_pipeline_instance

    # 2. Load new instance if no suitable one exists or force_reload is True
    print(f"Loading new TTS model instance (force_reload={force_reload})...")
    try:
        # Initialize KPipeline - check its docs if it accepts a 'device' argument
        new_pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M") # Add device=target_device if supported

        # --- Optional: Attempt to move model to target device after loading ---
        # This also depends heavily on KPipeline's implementation
        # try:
        #     if hasattr(new_pipeline, 'to') and callable(new_pipeline.to):
        #          new_pipeline.to(target_device)
        #          print(f"Attempted to move new TTS model to {target_device}")
        #     elif hasattr(new_pipeline, 'model') and hasattr(new_pipeline.model, 'to') and callable(new_pipeline.model.to):
        #          new_pipeline.model.to(target_device)
        #          print(f"Attempted to move wrapped TTS model to {target_device}")
        # except Exception as e:
        #     print(f"Warning: Could not explicitly move new TTS model to {target_device}: {e}")
        # --- End optional move ---

        _tts_pipeline_instance = new_pipeline # Cache the new instance globally
        print("TTS model loaded successfully.")
        # You could add another device check here after loading/moving if needed
        return _tts_pipeline_instance

    except Exception as e:
        print(f"FATAL: Error loading TTS model: {e}")
        _tts_pipeline_instance = None # Ensure instance is None if loading fails
        return None # Return None to indicate failure