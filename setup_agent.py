#!/usr/bin/env python3
"""
PerceptaAI Agent Setup Script
Ensures all dependencies and configurations are ready for smooth processing.
"""

import os
import sys
import torch
import subprocess
import platform
from pathlib import Path

def print_header():
    print("🚀 PerceptaAI Agent Setup Script")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_torch_cuda():
    """Check PyTorch and CUDA availability"""
    print("\n🔥 Checking PyTorch and CUDA...")
    print(f"✅ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        return "cuda"
    else:
        print("⚠ CUDA not available - using CPU")
        return "cpu"

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n📦 Checking required dependencies...")
    
    required_packages = [
        "whisper", "kokoro", "langchain-openai", "langchain-core",
        "sounddevice", "pyaudio", "keyboard", "opencv-python", 
        "pytesseract", "pillow", "ultralytics", "networkx", "pyttsx3"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "whisper":
                import whisper
                print(f"✅ {package}")
            elif package == "kokoro":
                from kokoro import KPipeline
                print(f"✅ {package}")
            elif package == "langchain-openai":
                from langchain_openai import ChatOpenAI
                print(f"✅ {package}")
            elif package == "langchain-core":
                from langchain_core.tools import tool
                print(f"✅ {package}")
            elif package == "sounddevice":
                import sounddevice
                print(f"✅ {package}")
            elif package == "pyaudio":
                import pyaudio
                print(f"✅ {package}")
            elif package == "keyboard":
                import keyboard
                print(f"✅ {package}")
            elif package == "opencv-python":
                import cv2
                print(f"✅ {package}")
            elif package == "pytesseract":
                import pytesseract
                print(f"✅ {package}")
            elif package == "pillow":
                from PIL import Image
                print(f"✅ {package}")
            elif package == "ultralytics":
                from ultralytics import YOLO
                print(f"✅ {package}")
            elif package == "networkx":
                import networkx
                print(f"✅ {package}")
            elif package == "pyttsx3":
                import pyttsx3
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages"""
    if not missing_packages:
        return True
    
    print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
    
    # Special handling for whisper
    install_commands = []
    for package in missing_packages:
        if package == "whisper":
            install_commands.append("pip install openai-whisper")
        else:
            install_commands.append(f"pip install {package}")
    
    for cmd in install_commands:
        print(f"⏳ Running: {cmd}")
        try:
            subprocess.run(cmd.split(), check=True)
            print("✅ Installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install: {e}")
            return False
    
    return True

def check_file_structure():
    """Check if required files exist"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "utils/model.py",
        "utils/audio_player.py", 
        "utils/audio_generator.py",
        "Sceene_description.py",
        "OCR.py",
        "nav_core.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return missing_files

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("\n👁 Checking Tesseract OCR...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract not found: {e}")
        if platform.system() == "Windows":
            print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        else:
            print("   Install with: sudo apt-get install tesseract-ocr (Ubuntu)")
            print("   Install with: brew install tesseract (macOS)")
        return False

def optimize_torch_settings():
    """Optimize PyTorch settings for performance"""
    print("\n⚡ Optimizing PyTorch settings...")
    
    # Set thread settings for better CPU performance
    torch.set_num_threads(4)
    print("✅ Set PyTorch threads to 4")
    
    # Set inference mode globally
    torch.set_grad_enabled(False)
    print("✅ Disabled gradients for inference")
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✅ Enabled CUDNN benchmark")

def create_directories():
    """Create necessary directories"""
    print("\n📂 Creating necessary directories...")
    
    directories = [
        "temp_uploads",
        "precompiled_models", 
        "logs",
        "Nav/map_data",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}/")

def check_audio_devices():
    """Check audio input/output devices"""
    print("\n🔊 Checking audio devices...")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        print(f"✅ Found {len(input_devices)} input device(s)")
        print(f"✅ Found {len(output_devices)} output device(s)")
        
        if len(input_devices) == 0:
            print("⚠ No input devices found - voice recording may not work")
        if len(output_devices) == 0:
            print("⚠ No output devices found - TTS may not work")
            
        return len(input_devices) > 0 and len(output_devices) > 0
        
    except Exception as e:
        print(f"❌ Error checking audio devices: {e}")
        return False

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check PyTorch and CUDA
    device = check_torch_cuda()
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"\n⚠ Missing packages detected: {missing_packages}")
        install = input("Install missing packages? (y/n): ").lower().strip()
        if install == 'y':
            if not install_missing_packages(missing_packages):
                print("❌ Failed to install some packages")
                sys.exit(1)
        else:
            print("⚠ Continuing with missing packages - some features may not work")
    
    # Check file structure
    missing_files = check_file_structure()
    if missing_files:
        print(f"\n⚠ Missing files: {missing_files}")
        print("   Please ensure all required files are present")
    
    # Check Tesseract
    check_tesseract()
    
    # Check audio devices
    check_audio_devices()
    
    # Optimize settings
    optimize_torch_settings()
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup complete!")
    print("=" * 50)
    print("Ready to run the PerceptaAI Agent!")
    print("Run: python agent_standalone.py")
    
    return device

if __name__ == "__main__":
    main() 