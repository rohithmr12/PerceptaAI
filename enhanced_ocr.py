import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import time

def capture_video(duration=3, fps=30):
    """Capture a short video using the default camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")
    
    frames = []
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            time.sleep(1/fps)  # Control frame rate
    
    cap.release()
    return frames

def analyze_frame_quality(frame):
    """
    Analyze frame quality based on multiple factors:
    - Brightness
    - Contrast
    - Blur detection
    - Text region detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Text region detection using EAST or simple edge detection
    edges = cv2.Canny(gray, 100, 200)
    text_regions = np.count_nonzero(edges)
    
    # Combine metrics into a quality score
    quality_score = (brightness/255 * 0.2 +  # Normalize brightness
                    contrast/128 * 0.3 +     # Weight contrast more
                    min(blur/1000, 1) * 0.3 + # Cap blur score at 1
                    text_regions/10000 * 0.2) # Normalize text regions
    
    return quality_score

def select_best_frame(frames):
    """Select the best frame based on quality analysis."""
    best_score = -1
    best_frame = None
    
    for frame in frames:
        score = analyze_frame_quality(frame)
        if score > best_score:
            best_score = score
            best_frame = frame
    
    return best_frame

def enhance_frame(frame):
    """Enhance the frame for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    # Apply dilation to make text more prominent
    kernel = np.ones((1,1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    return dilated

def enhanced_ocr(duration=3, fps=30, language_code='eng'):
    """
    Enhanced OCR function that:
    1. Captures video
    2. Selects best frame
    3. Enhances the frame
    4. Performs OCR using best Tesseract model
    """
    try:
        print("Initializing camera...")
        frames = capture_video(duration, fps)
        
        if not frames:
            return "Error: No frames captured"
        
        print(f"Captured {len(frames)} frames. Analyzing for best quality...")
        best_frame = select_best_frame(frames)
        
        print("Enhancing selected frame...")
        enhanced = enhance_frame(best_frame)
        
        # Save the enhanced frame temporarily
        temp_path = 'temp_enhanced_frame.png'
        cv2.imwrite(temp_path, enhanced)
        
        print(f"Performing OCR using Tesseract's best model (Language: {language_code})...")
        # Configure Tesseract path and parameters
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        
        # Custom configuration for best accuracy
        custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()[]{}-_+=\|/<>\"\'` " -c tessedit_pageseg_mode=6'
        
        # Perform OCR with best model and configuration
        img = Image.open(temp_path)
        detected_text = pytesseract.image_to_string(
            img, 
            lang=language_code,
            config=custom_config
        )
        
        # Clean up temporary file
        os.remove(temp_path)
        
        if detected_text.strip():
            print("\nDetected Text:")
            print("--------------")
            print(detected_text)
            print("--------------")
            return detected_text
        else:
            return "No text detected in the captured frames."
            
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # Example usage
    result = enhanced_ocr(duration=3, fps=30)  # Capture 3 seconds of video at 30fps
