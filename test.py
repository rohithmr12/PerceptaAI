import cv2
import numpy as np
import time

# Constants
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
KNOWN_DISTANCE = 50  # cm (distance where first detection is assumed)
KNOWN_WIDTH = 14.3   # cm (width of object at known distance — person face by default)

# Load class names
with open("coco.names", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Estimate distance
def estimate_distance(focal_length, known_width, object_width):
    if object_width == 0:
        return 0
    return (known_width * focal_length) / object_width

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

focal_length = None  # Will be calculated once from first detection

print("[INFO] Starting camera stream. Move your known object (e.g., face) to about 50cm from the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            class_name = CLASSES[class_ids[i]]
            color = COLORS[class_ids[i]]

            # Calculate focal length from the first detected object
            if focal_length is None:
                focal_length = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
                print(f"[INFO] Focal Length calculated from first detection: {focal_length:.2f}")

            distance = estimate_distance(focal_length, KNOWN_WIDTH, w)
            label = f"{class_name}: {distance:.2f} cm"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            print(f"[INFO] {class_name} detected at approx. {distance:.2f} cm")

    cv2.imshow("YOLO Object Detection & Distance", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()

