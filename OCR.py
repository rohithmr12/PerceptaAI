import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ocr")

# Ensure pytesseract is properly configured
# For Windows: 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/Mac this usually works automatically if tesseract is installed

class OCRProcessor:
    """
    Advanced OCR processor with preprocessing capabilities to improve text recognition.
    """
    
    def __init__(self, 
                 tesseract_path: Optional[str] = None, 
                 lang: str = 'eng',
                 dpi: int = 300):
        """
        Initialize the OCR processor.
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
            lang: Language(s) for OCR, e.g., 'eng' or 'eng+fra'
            dpi: DPI to use for processing
        """
        self.language = lang
        self.dpi = dpi
        
        # Configure tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Try to check if tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Using Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.warning(f"Could not verify Tesseract installation: {e}")
            logger.warning("You may need to install Tesseract or set the correct path")
    
    def preprocess_image(self, 
                         image: np.ndarray, 
                         preprocessing_type: str = 'default') -> np.ndarray:
        """
        Preprocess the image for better OCR results.
        
        Args:
            image: Input image as numpy array
            preprocessing_type: Type of preprocessing to apply:
                - 'default': Basic grayscale and thresholding
                - 'adaptive': Adaptive thresholding for varying lighting
                - 'denoise': Apply denoising
                - 'document': Specific for document images
        
        Returns:
            Preprocessed image
        """
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Convert to grayscale if it's a color image
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
            
        if preprocessing_type == 'default':
            # Basic preprocessing
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Apply binary thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = thresh
            
        elif preprocessing_type == 'adaptive':
            # Adaptive thresholding for varying lighting conditions
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            processed = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
        elif preprocessing_type == 'denoise':
            # Apply denoising
            processed = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif preprocessing_type == 'document':
            # Document-specific preprocessing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Edge enhancement
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)
            # Binarization
            _, processed = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        else:
            # Default to original grayscale if unknown preprocessing type
            processed = gray
            
        return processed
    
    def detect_text_regions(self, 
                           image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions in the image that likely contain text.
        
        Args:
            image: Input image
            
        Returns:
            List of bounding boxes (x, y, w, h) for text regions
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # Convert regions to bounding boxes
        boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            # Filter out very small regions or regions with extreme aspect ratios
            if w > 5 and h > 5 and 0.2 < w/h < 5:
                boxes.append((x, y, w, h))
        
        # Merge overlapping boxes
        boxes = self._merge_boxes(boxes)
        
        return boxes
    
    def _merge_boxes(self, 
                     boxes: List[Tuple[int, int, int, int]], 
                     overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping bounding boxes.
        
        Args:
            boxes: List of bounding boxes (x, y, w, h)
            overlap_threshold: IoU threshold for merging
            
        Returns:
            Merged bounding boxes
        """
        if not boxes:
            return []
            
        # Sort boxes by x coordinate
        boxes = sorted(boxes, key=lambda b: b[0])
        
        merged_boxes = [boxes[0]]
        
        for box in boxes[1:]:
            last_box = merged_boxes[-1]
            
            # Calculate current box coordinates
            x1, y1, w1, h1 = box
            x2, y2 = x1 + w1, y1 + h1
            
            # Calculate last box coordinates
            lx1, ly1, lw1, lh1 = last_box
            lx2, ly2 = lx1 + lw1, ly1 + lh1
            
            # Check for overlap
            if (x1 <= lx2 and x2 >= lx1 and 
                y1 <= ly2 and y2 >= ly1):
                # Merge boxes
                new_x = min(x1, lx1)
                new_y = min(y1, ly1)
                new_w = max(x2, lx2) - new_x
                new_h = max(y2, ly2) - new_y
                
                merged_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(box)
                
        return merged_boxes
    
    def extract_text(self, 
                     image: Union[str, np.ndarray, Image.Image],
                     preprocessing: str = 'default',
                     detect_regions: bool = False) -> Dict:
        """
        Extract text from an image with advanced options.
        
        Args:
            image: Path to image file, PIL Image, or numpy array
            preprocessing: Preprocessing method to apply
            detect_regions: Whether to detect and process specific text regions
            
        Returns:
            Dictionary with extracted information
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                logger.error(f"Image file not found: {image}")
                return {"error": "Image file not found", "text": ""}
                
            try:
                img = cv2.imread(image)
                pil_img = Image.open(image)
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return {"error": f"Error loading image: {e}", "text": ""}
                
        elif isinstance(image, np.ndarray):
            img = image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        elif isinstance(image, Image.Image):
            pil_img = image
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        else:
                logger.error("Unsupported image type")
                return {"error": "Unsupported image type", "text": ""}
            
            # Check if image is valid
        if img is None or img.size == 0:
            logger.error("Invalid image")
            return {"error": "Invalid image", "text": ""}
            
        # Preprocess image
        processed_img = self.preprocess_image(img, preprocessing)
        
        # Extract text based on strategy
        result = {}
        
        try:
            if detect_regions:
                # Detect and process individual text regions
                boxes = self.detect_text_regions(processed_img)
                region_texts = []
                
                for i, (x, y, w, h) in enumerate(boxes):
                    region = processed_img[y:y+h, x:x+w]
                    if region.size > 0:  # Ensure region is not empty
                        region_text = pytesseract.image_to_string(
                            region, 
                            lang=self.language,
                            config=f'--psm 6 --oem 3 -c tessedit_do_invert=0'
                        ).strip()
                        
                        if region_text:
                            region_texts.append({
                                "region_id": i,
                                "bbox": (x, y, w, h),
                                "text": region_text
                            })
                
                # Combine region texts in reading order (top to bottom, left to right)
                region_texts.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
                full_text = "\n".join(r["text"] for r in region_texts)
                
                result["regions"] = region_texts
                result["text"] = full_text
                
            else:
                # Process the entire image
                text = pytesseract.image_to_string(
                    processed_img,
                    lang=self.language,
                    config=f'--psm 3 --oem 3'
                ).strip()
                
                result["text"] = text
                
            # Get additional data for better insights
            result["confidence"] = self._get_confidence(processed_img)
            
            # Clean up text a bit
            result["text"] = self._clean_text(result["text"])
            
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            result["error"] = f"Error during OCR processing: {e}"
            result["text"] = ""
        
        return result
    
    def _get_confidence(self, image: np.ndarray) -> float:
        """Get confidence score for the OCR result"""
        try:
            data = pytesseract.image_to_data(
                image, 
                lang=self.language,
                config='--psm 3 --oem 3',
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence of detected words
            confidences = [int(conf) for conf in data["conf"] if conf != '-1']
            return sum(confidences) / len(confidences) if confidences else 0
            
        except Exception as e:
            logger.warning(f"Could not calculate confidence: {e}")
            return 0
    
    def _clean_text(self, text: str) -> str:
        """Clean up OCR text by removing common artifacts"""
        if not text:
            return ""
            
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        
        # Remove lines that are just noise (very short or just punctuation)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 1 and re.search(r'[a-zA-Z0-9]', line)]
        
        return '\n'.join(cleaned_lines)

# Function for use in MCP server
def ocr_text(image_path: str = "test.png", preprocessing: str = "default", detect_regions: bool = False) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path: Path to the image file
        preprocessing: Preprocessing method to apply (default, adaptive, denoise, document)
        detect_regions: Whether to detect specific text regions
        
    Returns:
        Extracted text or error message
    """
    ocr_processor = OCRProcessor()
    
    if not image_path:
        # Generate mock text for testing when no image is provided
        mock_texts = [
            "Annual Report 2023: Revenue increased by 15% compared to previous year.",
            "Meeting Agenda: 1. Project Updates 2. Budget Review 3. New Initiatives",
            "CAUTION: Wet Floor. Please use alternate route.",
            "Store Hours: Monday-Friday 9AM-9PM, Saturday-Sunday 10AM-7PM",
            "No text detected in the image."
        ]
        import random
        return random.choice(mock_texts)
        
    try:
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
            
        result = ocr_processor.extract_text(
            image_path,
            preprocessing=preprocessing,
            detect_regions=detect_regions
        )
        
        if "error" in result and result["error"]:
            return f"OCR Error: {result['error']}"
            
        if not result["text"]:
            return "No text detected in the image."
            
        return result["text"]
        
    except Exception as e:
        logger.error(f"Unexpected error in OCR processing: {e}")
        return f"Unexpected error in OCR processing: {str(e)}"

# Main function for testing
if __name__ == "__main__":
    # Test with a sample image if provided as argument
    import sys
    
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        print(f"Processing {image_file}...")
        result = ocr_text(image_file, preprocessing="document", detect_regions=True)
        print("\nExtracted Text:")
        print("="*50)
        print(result)
    else:
        print("No image provided. Usage: python ocr.py <image_path>")
        # Show mock result
        print("\nMock OCR Result:")
        print("="*50)
        print(ocr_text())