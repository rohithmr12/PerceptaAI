import cv2
import numpy as np
import easyocr
import torch
import time
from PIL import Image

class AdvancedOCR:
    def __init__(self, languages=['en']):
        """
        Initialize EasyOCR with specified languages
        languages: list of language codes (e.g., ['en'] for English, ['en', 'hi'] for English+Hindi)
        """
        print("Initializing EasyOCR model...")
        # Use GPU if available for faster processing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reader = easyocr.Reader(
            languages,
            gpu=torch.cuda.is_available(),
            model_storage_directory='models',
            download_enabled=True
        )
        print(f"Model initialized on {self.device}")

    def capture_video(self, duration=3, fps=30):
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
                time.sleep(1/fps)
        
        cap.release()
        return frames

    def analyze_frame_quality(self, frame):
        """
        Analyze frame quality using advanced metrics:
        - Brightness and contrast
        - Blur detection
        - Text region probability using EasyOCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Get text detection confidence
        result = self.reader.readtext(frame, detail=1, paragraph=False)
        text_confidence = np.mean([detection[2] for detection in result]) if result else 0
        
        # Combine metrics with emphasis on text confidence
        quality_score = (
            brightness/255 * 0.15 +      # Brightness (15%)
            contrast/128 * 0.15 +        # Contrast (15%)
            min(blur/1000, 1) * 0.2 +    # Blur (20%)
            text_confidence * 0.5         # Text confidence (50%)
        )
        
        return quality_score, result

    def select_best_frame(self, frames):
        """Select the best frame based on quality analysis."""
        best_score = -1
        best_frame = None
        best_result = None
        
        print("Analyzing frames for text quality...")
        for i, frame in enumerate(frames):
            score, result = self.analyze_frame_quality(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
                best_result = result
        
        return best_frame, best_result

    def enhance_frame(self, frame):
        """
        Apply advanced image enhancement:
        - Adaptive histogram equalization
        - Denoising
        - Sharpening
        """
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(enhanced)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened

    def extract_text(self, duration=3, fps=30, confidence_threshold=0.5):
        """
        Main function to capture video and extract text.
        Returns both the raw text and structured data with positions and confidence scores.
        """
        try:
            print("Initializing camera...")
            frames = self.capture_video(duration, fps)
            
            if not frames:
                return "Error: No frames captured", []
            
            print(f"Captured {len(frames)} frames")
            best_frame, best_result = self.select_best_frame(frames)
            
            if best_frame is None:
                return "Error: Could not find suitable frame", []
            
            print("Enhancing selected frame...")
            enhanced_frame = self.enhance_frame(best_frame)
            
            # Perform final OCR on enhanced frame
            print("Performing final OCR analysis...")
            final_result = self.reader.readtext(
                enhanced_frame,
                detail=1,
                paragraph=True,
                contrast_ths=0.2,
                adjust_contrast=0.5,
                add_margin=0.1,
                width_ths=0.5,
                height_ths=0.5
            )
            
            # Filter results by confidence threshold
            filtered_result = [
                detection for detection in final_result 
                if detection[2] >= confidence_threshold
            ]
            
            # Extract text with positions and confidence
            structured_output = []
            full_text = []
            
            for bbox, text, confidence in filtered_result:
                structured_output.append({
                    'text': text,
                    'confidence': confidence,
                    'position': bbox
                })
                full_text.append(text)
            
            combined_text = ' '.join(full_text)
            
            if combined_text.strip():
                print("\nDetected Text:")
                print("--------------")
                print(combined_text)
                print("\nConfidence Scores:")
                for item in structured_output:
                    print(f"{item['text']}: {item['confidence']:.2f}")
                print("--------------")
            else:
                print("No text detected with sufficient confidence.")
            
            return combined_text, structured_output
            
        except Exception as e:
            return f"An error occurred: {str(e)}", []

if __name__ == "__main__":
    # Initialize with English. Add more languages as needed (e.g., ['en', 'hi'] for English + Hindi)
    ocr = AdvancedOCR(languages=['en'])
    
    # Extract text with default settings
    text, details = ocr.extract_text(
        duration=3,           # Video duration in seconds
        fps=30,              # Frames per second
        confidence_threshold=0.5  # Minimum confidence score (0-1)
    )
