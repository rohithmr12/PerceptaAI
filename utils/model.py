"""
Model initialization utilities for Whisper, TTS, and LLM models.
"""

import torch
import whisper
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Global model instances to avoid reloading
_tts_pipeline = None
_whisper_model = None
_llm = None

def get_optimal_device():
    """Get the best available compute device"""
    try:
        # First try CUDA
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
        print("✓ Using CPU (no GPU acceleration available)")
        return "cpu"
        
    except Exception as e:
        print(f"! Error checking device availability: {e}")
        print("✓ Defaulting to CPU for safety")
        return "cuda"

def initiate_tts_model(desired_device: str = None, force_reload: bool = False, force_cpu: bool = True):
    """
    Initialize TTS model using Kokoro - optimized for speed
    
    Args:
        desired_device (str): Target device ('cuda', 'cpu', etc.)
        force_reload (bool): Force reload even if already cached
        force_cpu (bool): Force CPU usage (default True for speed)
        
    Returns:
        TTS pipeline object or None if failed
    """
    global _tts_pipeline
    
    # Force clear cache to ensure CPU loading
    if force_cpu:
        _tts_pipeline = None
    
    if _tts_pipeline is not None and not force_reload:
        print("✓ Using cached TTS pipeline")
        return _tts_pipeline
        
    print("⚡ Ultra-fast TTS initialization...")
    
    try:
        # Import Kokoro here to avoid import errors if not available
        from kokoro import KPipeline
        
        # Speed-optimized Kokoro settings
        print("🚀 Loading TTS model with speed optimizations...")
        _tts_pipeline = KPipeline(
            lang_code='a', 
            repo_id="hexgrad/Kokoro-82M",
            # Speed optimizations        
            # speed=1.3,  # Faster speech rate
            device='cuda'  # Force CPU to avoid CUDA hanging issues
        )
        
        # Configure pipeline for speed
        if hasattr(_tts_pipeline, 'model'):
            # Set inference mode for faster processing
            _tts_pipeline.model.eval()
            
            # Optimize for inference speed
            if hasattr(_tts_pipeline.model, 'set_speed'):
                _tts_pipeline.model.set_speed(1.2)
            
            # REMOVED: FP16 conversion that causes dtype mismatch
            # The input tensors are Float32 but model parameters become Half precision
            # This causes: "Input and parameter tensors are not the same dtype"
            # Comment out the problematic half() conversion:
            # if not force_cpu and torch.cuda.is_available():
            #     try:
            #         _tts_pipeline.model = _tts_pipeline.model.half()  # Use FP16 for speed
            #     except:
            #         pass
        
        print("⚡ TTS ready with speed optimizations!")
        return _tts_pipeline
        
    except ImportError:
        print("❌ Kokoro not available. Install with: pip install kokoro")
        return None
    except Exception as e:
        print(f"❌ Error loading TTS model: {e}")
        # Fallback to basic initialization
        try:
            print("🔄 Falling back to basic TTS initialization...")
            _tts_pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M", device='cuda')
            print("✓ Basic TTS ready!")
            return _tts_pipeline
        except Exception as fallback_e:
            print(f"❌ Fallback also failed: {fallback_e}")
            return None

def initiate_whisper_model(model_name: str = "small", force_reload: bool = False, force_cpu: bool = False):
    """
    Initialize Whisper model
    
    Args:
        model_name (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        force_reload (bool): Force reload even if already cached
        force_cpu (bool): Force CPU usage
        
    Returns:
        Whisper model object or None if failed
    """
    global _whisper_model
    
    if _whisper_model is not None and not force_reload:
        print("Existing Whisper model instance found.")
        return _whisper_model
        
    print(f"Initiating Whisper model ({model_name})...")
    
    try:
        device = "cuda" if force_cpu else get_optimal_device()
        print(f"Target device for Whisper: {device}")
        
        _whisper_model = whisper.load_model(model_name, device=device)
        print("✓ Whisper model loaded successfully")
        return _whisper_model
        
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return None

def initiate_llm(base_url: str, model_identifier: str, temperature: float = 0.1):
    """
    Initialize LLM using LangChain OpenAI interface (for LM Studio)
    
    Args:
        base_url (str): LM Studio base URL
        model_identifier (str): Model identifier
        temperature (float): Sampling temperature
        
    Returns:
        LangChain LLM object or None if failed
    """
    global _llm
    
    print("Initiating LLM...")
    
    try:
        # Set up environment
        os.environ["OPENAI_API_KEY"] = "lm-studio"  # Dummy key for LM Studio
        
        _llm = ChatOpenAI(
            base_url=base_url,
            model=model_identifier,
            temperature=temperature,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            request_timeout=30,
            max_retries=1
        )
        
        print("✓ LLM initialized successfully")
        return _llm
        
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        return None

def clear_model_cache():
    """Clear all cached models to force reload"""
    global _tts_pipeline, _whisper_model, _llm
    _tts_pipeline = None
    _whisper_model = None
    _llm = None
    print("Model cache cleared") 