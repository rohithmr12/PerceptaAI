import cv2
def capture_image(filename='snap.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera.")
        return
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
    else:
        print("Failed to capture image.")
    
    cap.release()
    return filename
# Test the function
# capture_image("snap.jpg")