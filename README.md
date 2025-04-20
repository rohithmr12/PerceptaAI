# PerceptaAI

## Enhanced OCR System

The enhanced OCR system is designed to improve text recognition accuracy by:
1. Capturing a short video instead of a single image
2. Analyzing frame quality based on brightness, contrast, blur, and text regions
3. Selecting the best frame for OCR
4. Applying image enhancement techniques
5. Performing OCR using Tesseract

### Prerequisites
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
- Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
- Make sure Tesseract is added to your system PATH

### Usage
```python
from enhanced_ocr import enhanced_ocr

# Capture 3 seconds of video at 30fps and perform OCR
text = enhanced_ocr(duration=3, fps=30, language_code='eng')
print(text)
```

The system will:
1. Capture video from your camera
2. Analyze all frames for quality
3. Select the best frame
4. Enhance the image
5. Perform OCR and return the detected text