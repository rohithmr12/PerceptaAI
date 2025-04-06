import pytesseract
from PIL import Image # Python Imaging Library (Pillow)
import os
from utils.snap_a_picture import capture_image # Ensure this module is available in your environment
def ocr():
    capture_image("test.png") # Capture an image and save it as 'test.png'
    # --- Configuration ---
    # Specify the path to your image
    image_path = 'test.png' # <--- CHANGE THIS
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    # Specify languages (e.g., 'eng' for English, 'fra' for French, 'eng+fra' for both)
    # You need to have the corresponding language data files installed for Tesseract
    language_code = 'eng'

    # **IMPORTANT**: If Tesseract is NOT in your system's PATH,
    # you MUST specify the path to the tesseract executable here.
    # Examples:
    # Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # macOS/Linux (if installed in a non-standard location): pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
    # If Tesseract is in your PATH, you can comment out or remove the line below.
    # pytesseract.pytesseract.tesseract_cmd = r'/path/to/your/tesseract' # <--- UNCOMMENT AND SET IF NEEDED
    # --------------------


    # --- Check if image exists ---
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
    else:
        try:
            print(f"Opening image: {image_path}...")
            # Open the image file using Pillow
            img = Image.open(image_path)

            print(f"Performing OCR using Tesseract (Language: {language_code})...")
            # Perform OCR using Tesseract
            # image_to_string is the simplest function, returns the detected text as a single string
            detected_text = pytesseract.image_to_string(img, lang=language_code)

            print("\nDetected Text:")
            print("--------------")
            if detected_text.strip(): # Check if any text (other than whitespace) was detected
                print(detected_text)
            else:
                print("No text detected or unable to process.")

            print("--------------")
            print("Tesseract processing complete.")

            # --- Advanced Usage Example: Get detailed data (boxes, confidence, etc.) ---
            # print("\nGetting detailed OCR data...")
            # data = pytesseract.image_to_data(img, lang=language_code, output_type=pytesseract.Output.DICT)
            # n_boxes = len(data['level'])
            # print("Detailed Data (Word level):")
            # for i in range(n_boxes):
            #     if int(data['conf'][i]) > 60: # Filter by confidence score (e.g., > 60)
            #         (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            #         text = data['text'][i]
            #         conf = data['conf'][i]
            #         print(f"- Word: \"{text}\", Confidence: {conf}%, Box: (x={x}, y={y}, w={w}, h={h})")

        except pytesseract.TesseractNotFoundError:
            print("\n--- TESSERACT ERROR ---")
            print("Tesseract executable not found. Please ensure:")
            print("1. Tesseract OCR engine is installed on your system.")
            print("2. The path to 'tesseract.exe' (Windows) or 'tesseract' (macOS/Linux)")
            print("   is either in your system's PATH environment variable OR")
            print("   explicitly set via 'pytesseract.pytesseract.tesseract_cmd = r'/path/to/...'` in the script.")
            print("-----------------------")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")