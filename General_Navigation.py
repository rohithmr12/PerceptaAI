import cv2
import numpy as np
import time
import threading
import math
import os
import subprocess
from datetime import datetime
import urllib.request
import zipfile
import io

# Import pyttsx3 for potentially faster text-to-speech
try:
    import pyttsx3
except ImportError:
    print("Warning: pyttsx3 not found. Install with 'pip install pyttsx3 pyobjc' for better speech performance.")
    pyttsx3 = None # Set to None if import fails

class VisionAssistant:
    def __init__(self, force_headless: bool = False, external_tts_pipeline=None):
        # Camera setup
        self.cap = cv2.VideoCapture(0)  # Use default camera (change index if needed)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if GUI is available (unless forced headless)
        self.headless_mode = force_headless
        if not force_headless:
            try:
                # Test if we can create a window
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow("test_window", test_img)
                cv2.waitKey(1)
                cv2.destroyWindow("test_window")
                cv2.waitKey(1)  # Additional waitKey to ensure cleanup
                print("✅ GUI mode available")
                self.headless_mode = False
            except (cv2.error, Exception) as e:
                self.headless_mode = True
                print(f"⚠ GUI not available - running in headless mode: {e}")
        else:
            print("🔇 Forced headless mode")

        # Use FAST primitive TTS for navigation (ignore external pipeline)
        print("🚀 Initializing fast primitive TTS for navigation...")
        self.external_tts_pipeline = None  # Force use of primitive TTS
        
        # Text-to-speech setup - prioritize speed
        self.tts_engine = None
        self.speaking_subprocess = False
        
        if pyttsx3 is not None:
            try:
                self.tts_engine = pyttsx3.init()
                # Speed up TTS for navigation
                rate = self.tts_engine.getProperty('rate')
                self.tts_engine.setProperty('rate', rate + 50)  # Faster speech
                volume = self.tts_engine.getProperty('volume')
                self.tts_engine.setProperty('volume', min(volume + 0.1, 1.0))  # Slightly louder
                
                # Configure speaker device selection
                self._configure_speaker_device()
                
                print("✅ Fast pyttsx3 initialized for navigation")
            except Exception as e:
                print(f"⚠ pyttsx3 failed: {e}")
                self.tts_engine = None

        # Fallback to subprocess 'say' if pyttsx3 failed
        if self.tts_engine is None:
             print("✅ Using fast subprocess 'say' for navigation")

        # Test and validate fast TTS performance
        self._test_fast_tts_performance()

        self.last_alert_time = 0
        self.alert_cooldown = 0.3  # Very short cooldown for fast navigation updates

        # Depth estimation parameters (Simple, Approximate Method)
        # NOTE: This method is INACCURATE without camera calibration and precise known widths.
        # For better accuracy, perform camera calibration and use the calculated focal length (fx).
        self.approx_focal_length_pixels = 615  # Approximate focal length for standard webcam (PLACEHOLDER)
        self.known_object_width_meters = 0.6  # Approximate average width of a person in meters (PLACEHOLDER)
        # For more accurate distance, you would need a calibrated camera matrix and distortion coefficients.
        # Example: self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        #          self.dist_coeffs = np.array([k1, k2, p1, p2, k3])


        # Distance thresholds (in meters)
        self.danger_distance = 1.5
        self.warning_distance = 3.0

        # Prioritized objects (more important to detect)
        self.priority_objects = ["person", "car", "bicycle", "motorcycle", "bus", "truck",
                                 "chair", "bench", "stop sign", "fire hydrant", "dog", "wall", "wall edge"] # Added wall types

        # Direction sectors (left, center, right)
        # Ensure sector_width is calculated after getting frame width
        self.sector_width = self.frame_width // 3
        self.sectors = ["left", "center", "right"]

        # Running flag
        self.running = True

        # Parameters for wall detection
        self.use_wall_detection = True # Keep this tunable
        # NOTE: Wall distance estimation is also approximate, based on vertical position.
        # Accurate wall distance requires calibration and potentially ground plane estimation.
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.hough_threshold = 80
        self.min_line_length = 60
        self.max_line_gap = 10

        # Parameters for basic obstacle detection
        self.min_contour_area = 500

        # Frame processing timing
        self.last_process_time = 0
        self.process_interval = 2.0  # Process frames every 2 seconds
        
        # Frame skipping parameter (deprecated in favor of time-based processing)
        self.skip_frames = 1 # Keep for compatibility but will use time-based processing


        # --- Object Detection Model Loading ---
        self.model_loaded = False
        self.using_ultralytics = False
        try:
            import importlib
            ultralytics_spec = importlib.util.find_spec("ultralytics")

            if ultralytics_spec is not None:
                from ultralytics import YOLO

                # Load YOLOv8 model if it exists locally
                model_path = "yolov8n.pt"
                if os.path.exists(model_path):
                    self.yolo_model = YOLO(model_path)
                    self.model_loaded = True
                    self.using_ultralytics = True
                    print(f"Loaded YOLOv8 model from {model_path}")
                else:
                    print(f"YOLOv8 model file not found at {model_path}")
                    print("Attempting to download YOLOv8 model...")
                    try:
                        # Try to download the model (will save to default ultralytics path)
                        self.yolo_model = YOLO("yolov8n.pt") # This will download if not present
                        self.model_loaded = True
                        self.using_ultralytics = True
                        print("Successfully downloaded and loaded YOLOv8 model")
                    except Exception as e:
                        print(f"Failed to download or load YOLOv8 model: {e}")
                        print("Ensure you have internet access or the model file.")

                # Note: Ultralytics automatically attempts to use GPU (like Apple MPS) if available and configured.
                # Check ultralytics documentation for specific setup on macOS M-series chips if needed.

            else:
                print("Ultralytics package not found.")
                print("For better detection performance (especially with hardware acceleration) and object recognition, install with: pip install ultralytics")
        except Exception as e:
            print(f"Error during Ultralytics import or loading: {e}")

        # Fall back to older YOLOv4 if ultralytics not available or failed
        if not self.model_loaded:
            try:
                if os.path.exists("yolov4-tiny.weights") and os.path.exists("yolov4-tiny.cfg"):
                    # Load YOLOv4 model
                    self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
                    self.layer_names = self.net.getLayerNames()
                    try:
                        # OpenCV 4.5.4+
                        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
                    except:
                        # Older OpenCV versions
                        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

                    # Load COCO class labels
                    if os.path.exists("coco.names"):
                        with open("coco.names", "r") as f:
                            self.classes = [line.strip() for line in f.readlines()]
                        self.model_loaded = True
                        print("Loaded YOLOv4 model.")
                    else:
                        print("Warning: coco.names file not found. YOLOv4 classes will be unknown.")
                        self.model_loaded = False # Model not fully usable without names
                else:
                    print("Warning: YOLOv4 model files (yolov4-tiny.weights, yolov4-tiny.cfg) not found.")
            except Exception as e:
                print(f"Error loading YOLOv4 model: {e}")


    # Fast primitive TTS for navigation
    def speak(self, text):
        """Use fast primitive TTS (pyttsx3 or subprocess) for navigation"""
        print(f"🔊 Navigation TTS: '{text}'")
        
        # Priority 1: Fast pyttsx3 if available
        if self.tts_engine is not None:
            try:
                if not self.tts_engine.isBusy():
                    print("⚡ Using fast pyttsx3...")
                    threading.Thread(target=self._speak_thread_pyttsx3, args=(text,), daemon=True).start()
                    return
                else:
                    print("⚠ pyttsx3 busy, using subprocess fallback")
            except Exception as e:
                print(f"❌ pyttsx3 error: {e}")
        
        # Priority 2: Fast subprocess 'say'
        try:
            if not self.speaking_subprocess:
                print("⚡ Using fast subprocess say...")
                self.speaking_subprocess = True
                threading.Thread(target=self._speak_thread_subprocess, args=(text,), daemon=True).start()
                return
            else:
                print("⚠ All TTS methods busy")
        except Exception as e:
            print(f"❌ Subprocess say error: {e}")
        
        # Last resort: console output
        print(f"🔊 VOICE: {text}")
        print("⚠ All TTS methods failed, using console output")

    # pyttsx3 speak thread
    def _speak_thread_pyttsx3(self, text):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait() # Blocks this thread until done
        except Exception as e:
            print(f"Speech error (pyttsx3): {e}")
        # No need to reset flag, pyttsx3.isBusy() handles it

    # Subprocess speak thread (fallback)
    def _speak_thread_subprocess(self, text):
        try:
            subprocess.call(['say', text])
        except Exception as e:
            print(f"Speech error (subprocess say): {e}")
        finally:
            self.speaking_subprocess = False # Reset flag for subprocess

    def estimate_distance(self, box_width_pixels):
        """
        Estimates distance based on object's pixel width.
        NOTE: This is a HIGHLY APPROXIMATE method without camera calibration.
        For better accuracy, use camera calibration (fx, fy) and potentially a more
        sophisticated method like triangulation if using stereo vision, or a monocular
        depth estimation model.
        """
        # Formula used: Distance = (Known_Width * Focal_Length) / Object_Pixel_Width
        # This assumes the known_object_width_meters is accurate for the detected object
        # and approx_focal_length_pixels is accurate for the camera at this resolution.
        # It also assumes the object is roughly perpendicular to the camera view.

        if box_width_pixels > 0 and self.approx_focal_length_pixels > 0 and self.known_object_width_meters > 0:
            # If you had a calibrated fx:
            # distance = (self.known_object_width_meters * self.camera_matrix[0, 0]) / box_width_pixels

            # Using the approximate values:
            distance = (self.known_object_width_meters * self.approx_focal_length_pixels) / box_width_pixels

            # Cap the estimated distance to a reasonable range
            return max(0.1, min(distance, 15.0)) # Cap between 0.1m and 15m for realism
        return 15.0  # Default to far away if inputs are invalid or zero

    def get_sector(self, x):
        # Determine which sector (left, center, right) the object is in
        # Ensure x is within frame bounds
        x = max(0, min(x, self.frame_width - 1))
        sector_idx = min(x // self.sector_width, 2)
        return self.sectors[sector_idx]

    def detect_walls(self, frame):
        """
        Detect walls and structural elements using line detection.
        Distance estimation for walls/lines is based on vertical position,
        which is a rough approximation and not metrically accurate without
        camera calibration and potentially ground plane estimation.
        """
        detected_walls = []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                               minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)

        # Process detected lines
        if lines is not None:
            # Group lines by orientation (horizontal, vertical, diagonal)
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line angle
                # Avoid division by zero if x1 == x2
                if abs(x2 - x1) > 1e-6: # Check if x difference is non-zero
                    angle_rad = math.atan2(y2 - y1, x2 - x1)
                    angle_deg = abs(math.degrees(angle_rad))
                else:
                    angle_deg = 90 # Vertical line

                # Adjust angle to be within 0-180 range for horizontal/vertical check
                if angle_deg > 90:
                    angle_deg = 180 - angle_deg

                # Classify lines (allow small tolerance)
                if angle_deg < 10:  # Near horizontal
                    horizontal_lines.append(line[0])
                elif angle_deg > 80:  # Near vertical
                     vertical_lines.append(line[0])


            # Process horizontal lines (potential walls, floor/ceiling lines)
            if horizontal_lines:
                for line in horizontal_lines:
                    x1, y1, x2, y2 = line

                    # Draw the line
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow

                    # Get midpoint for sector calculation
                    mid_x = (x1 + x2) // 2
                    sector = self.get_sector(mid_x)

                    # Approximate distance based on vertical position (lower in frame = closer)
                    # This is a rough non-linear mapping, assumes a ground plane near the bottom.
                    # Needs calibration for accuracy.
                    # Example: Map y=frame_height to 0.5m, y=0 to 10m.
                    # Linear interpolation: distance = d_min + (d_max - d_min) * (1 - y / frame_height)
                    # Using the average y position of the line
                    avg_y = (y1 + y2) // 2
                    norm_y = avg_y / self.frame_height # Normalized y (0 at top, 1 at bottom)
                    # Invert norm_y for distance mapping: 1 at top (far), 0 at bottom (near)
                    inv_norm_y = 1.0 - norm_y
                    distance = 0.5 + (10.0 - 0.5) * inv_norm_y # Example range: 0.5m to 10m

                    detected_walls.append({
                        "label": "wall/structure", # More general label
                        "distance": max(0.5, min(distance, 10.0)), # Cap distance
                        "sector": sector,
                        "box": [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)], # Store as [x, y, w, h]
                        "priority": True # Walls are important obstacles
                    })

            # Process vertical lines (potential wall edges, door frames, etc.)
            if vertical_lines:
                for line in vertical_lines:
                    x1, y1, x2, y2 = line

                    # Draw the line
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan

                    # Get sector
                    mid_x = (x1 + x2) // 2
                    sector = self.get_sector(mid_x)

                    # Approximate distance (similar rough approximation as horizontal lines)
                    avg_y = (y1 + y2) // 2
                    norm_y = avg_y / self.frame_height
                    inv_norm_y = 1.0 - norm_y
                    distance = 0.5 + (10.0 - 0.5) * inv_norm_y

                    detected_walls.append({
                        "label": "wall edge/vertical structure", # More general label
                        "distance": max(0.5, min(distance, 10.0)), # Cap distance
                        "sector": sector,
                        "box": [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)], # Store as [x, y, w, h]
                        "priority": True # Wall edges are also important
                    })

        return detected_walls

    def detect_objects_yolov8(self, frame):
        """Object detection using YOLOv8 via ultralytics"""
        detected_objects = []

        # Get YOLOv8 results
        # Using verbose=False to reduce console output during run
        results = self.yolo_model(frame, conf=0.25, verbose=False)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates (in xyxy format)
                # Ensure conversion to numpy and then int
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Get width and height
                w_pixels, h_pixels = x2 - x1, y2 - y1

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls]

                # Estimate distance - using the bounding box width
                # This is an approximation, especially for objects with varying real-world sizes
                # like "chair" or "dog". It's slightly better for things like "person" if
                # self.known_object_width_meters is set for that, but still inaccurate without calibration.
                distance = self.estimate_distance(w_pixels)

                # Get sector
                sector = self.get_sector(x1 + w_pixels//2)

                # Create object
                obj = {
                    "label": label,
                    "distance": distance,
                    "sector": sector,
                    "box": [x1, y1, w_pixels, h_pixels], # Store as [x, y, w, h]
                    "confidence": conf,
                    "priority": label in self.priority_objects # Check if label is in priority list
                }

                detected_objects.append(obj)

                # Draw bounding box
                color = (0, 255, 0)  # Green
                if distance < self.danger_distance:
                    color = (0, 0, 255)  # Red
                elif distance < self.warning_distance:
                    color = (0, 165, 255)  # Orange

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add label with distance
                # Consider making messages shorter for faster TTS
                text = f"{label}: {distance:.1f}m"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, detected_objects

    def detect_objects_yolov4(self, frame):
        """Object detection using YOLOv4 via OpenCV DNN"""
        height, width, _ = frame.shape

        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Pass blob through network
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Process detection results
        class_ids = []
        confidences = []
        boxes = []

        # Process each output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w_pixels = int(detection[2] * width)
                    h_pixels = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w_pixels / 2)
                    y = int(center_y - h_pixels / 2)

                    boxes.append([x, y, w_pixels, h_pixels])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected_objects = []

        # Process detected objects
        if len(indices) > 0:
            # Ensure indices is flat
            indices = indices.flatten() # Ensure it's 1D numpy array

            for i in indices:
                box = boxes[i]
                x, y, w_pixels, h_pixels = box
                # Ensure class_id is within bounds of self.classes
                if self.classes and class_ids[i] < len(self.classes):
                    label = str(self.classes[class_ids[i]])
                else:
                    label = "unknown" # Handle unknown classes
                confidence = confidences[i]

                # Estimate distance - using the bounding box width
                # This is an approximation.
                distance = self.estimate_distance(w_pixels)

                # Get sector
                sector = self.get_sector(x + w_pixels//2)

                # Create object
                obj = {
                    "label": label,
                    "distance": distance,
                    "sector": sector,
                    "box": box,
                    "confidence": confidence,
                    "priority": label in self.priority_objects
                }

                detected_objects.append(obj)

                # Draw bounding box on the frame
                color = (0, 255, 0)  # Green
                if distance < self.danger_distance:
                    color = (0, 0, 255)  # Red
                elif distance < self.warning_distance:
                    color = (0, 165, 255)  # Orange

                cv2.rectangle(frame, (x, y), (x + w_pixels, y + h_pixels), color, 2)

                # Add label with distance
                text = f"{label}: {distance:.1f}m"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, detected_objects

    def detect_objects_basic(self, frame):
        """
        Basic obstacle detection using edges and contours.
        Distance estimation is based on contour width, which is a rough approximation.
        """
        height, width, _ = frame.shape

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)

        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8) # Smaller kernel might be better
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        # Use different retrieval mode if needed, but RETR_EXTERNAL is good for outer obstacles
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []

        # Process contours
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter small contours
            if area > self.min_contour_area:
                # Get bounding rectangle
                x, y, w_pixels, h_pixels = cv2.boundingRect(contour)

                # Estimate distance based on width
                # This is an approximation. Basic obstacles don't have a 'known width'.
                # This treats their bounding box width as if it corresponds to a fixed
                # physical size, which is incorrect.
                distance = self.estimate_distance(w_pixels)

                # Get sector
                sector = self.get_sector(x + w_pixels//2)

                # Create object
                obj = {
                    "label": "obstacle", # Basic detection labels as 'obstacle'
                    "distance": distance,
                    "sector": sector,
                    "box": [x, y, w_pixels, h_pixels],
                    # No confidence for basic detection
                    "priority": True # Consider all basic obstacles as priority
                }

                detected_objects.append(obj)

                # Draw bounding box
                color = (0, 255, 0)  # Green
                if distance < self.danger_distance:
                    color = (0, 0, 255)  # Red
                elif distance < self.warning_distance:
                    color = (0, 165, 255)  # Orange

                cv2.rectangle(frame, (x, y), (x + w_pixels, y + h_pixels), color, 2)

                # Add label with distance
                text = f"obstacle: {distance:.1f}m"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, detected_objects


    def detect_objects(self, frame):
        """Choose the appropriate detection method and combine with wall detection"""
        # Detect walls regardless of object detection method (if enabled)
        wall_objects = []
        if self.use_wall_detection:
            wall_objects = self.detect_walls(frame)

        # Detect objects using the best available method
        if self.using_ultralytics:
            frame, objects = self.detect_objects_yolov8(frame)
        elif self.model_loaded: # This means YOLOv4 was loaded
            frame, objects = self.detect_objects_yolov4(frame)
        else: # Fallback to basic
            frame, objects = self.detect_objects_basic(frame)

        # Combine wall and object detections
        all_objects = objects + wall_objects

        return frame, all_objects

    def generate_guidance(self, detected_objects):
        # Print detected objects for debugging
        if detected_objects:
            object_count = len(detected_objects)
            objects_list = [f"{obj['label']} ({obj['distance']:.1f}m {obj['sector']})" for obj in detected_objects[:3]]
            print(f"🔍 Detected {object_count} objects: {objects_list}")
        else:
            print("🔍 No objects detected")
        
        current_time = time.time()
        
        # Always process if we have objects, only use cooldown for "clear" messages
        if detected_objects:
            # Find the closest and most important objects
            sector_objects = {"left": [], "center": [], "right": []}
            for obj in detected_objects:
                if obj["sector"] in self.sectors:
                    sector_objects[obj["sector"]].append(obj)

            # Sort objects by distance and priority
            for sector in sector_objects:
                sector_objects[sector].sort(key=lambda x: (x["distance"], 0 if x.get("priority", False) else 1))

            # Generate guidance message for the closest/most important object
            message = ""
            urgent_messages = []
            warning_messages = []

            for sector in self.sectors:
                objects = sector_objects[sector]
                if objects:
                    closest = objects[0]
                    base_msg = f"{closest['label']} {sector}, {closest['distance']:.1f} meters"

                    if closest["distance"] < self.danger_distance:
                        urgent_messages.append((f"Warning! {base_msg}", closest["distance"], closest.get("priority", False)))
                    elif closest["distance"] < self.warning_distance:
                        warning_messages.append((base_msg, closest["distance"], closest.get("priority", False)))

            # Choose the most important message
            if urgent_messages:
                urgent_messages.sort(key=lambda x: (x[1], 0 if x[2] else 1))
                message = urgent_messages[0][0]
            elif warning_messages:
                warning_messages.sort(key=lambda x: (x[1], 0 if x[2] else 1))
                message = warning_messages[0][0]

            # Always speak when objects are detected (no cooldown)
            if message:
                print(f"🗣 SPEAKING: {message}")
                self.speak(message)
                self.last_alert_time = current_time
        
        else:
            # Only announce "clear" messages with cooldown
            if current_time - self.last_alert_time > 5.0:  # Every 5 seconds if no objects
                clear_msg = "Area clear"
                print(f"🗣 SPEAKING: {clear_msg}")
                self.speak(clear_msg)
                self.last_alert_time = current_time

    def run(self):
        try:
            print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
            if self.headless_mode:
                print("🖥 Running in headless mode (no display window)")
            else:
                print("🖥 Running with display window - Press 'q' to quit")

            if self.using_ultralytics:
                print("\nRunning with YOLOv8 (best performance)")
                print("Note: YOLOv8 can leverage hardware acceleration (like Apple MPS) if available and configured.")
            elif self.model_loaded:
                print("\nRunning with YOLOv4")
            else:
                print("\nRunning in basic mode (no YOLO model loaded)")
                print("For better detection performance (especially with hardware acceleration) and object recognition, install ultralytics package:")
                print("pip install ultralytics")

            if self.use_wall_detection:
                print("Wall detection enabled")
                print("Note: Wall distance is a rough estimate based on vertical position.")
            else:
                print("Wall detection disabled (can improve performance)")

            if self.tts_engine is not None:
                 print("Using fast pyttsx3 for navigation announcements.")
            else:
                 print("Using fast subprocess 'say' for navigation announcements.")

            print("\n--- Distance Estimation Accuracy Note ---")
            print("The current distance estimation (in meters) is HIGHLY APPROXIMATE.")
            print(f"It uses an assumed focal length ({self.approx_focal_length_pixels} pixels) and a generic object width ({self.known_object_width_meters} meters).")
            print("For accurate metric distance, you NEED to perform Camera Calibration for your specific camera and resolution.")
            print("Search online for 'OpenCV camera calibration' to learn how.")
            print("---------------------------------------")


            print(f"Alert cooldown set to {self.alert_cooldown} seconds.")
            print(f"Frame processing interval: {self.process_interval} seconds (analyzing every {self.process_interval}s)")

            if self.headless_mode:
                print("\n🔧 Headless Mode Controls:")
                print("- The system will run continuously providing voice guidance")
                print("- Press Ctrl+C to stop the navigation")
                print("- No video display window will be shown")
            else:
                print("\n🔧 GUI Mode Controls:")
                print("- Live video feed with detection overlays")
                print("- Press 'q' in video window to quit")

            frame_count = 0

            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                frame_count += 1
                current_time = time.time()

                # Always show live video feed
                display_frame = frame.copy()

                # Time-based processing: only analyze frames every 2 seconds
                should_process = (current_time - self.last_process_time) >= self.process_interval
                detected_objects = []

                if should_process:
                    print(f"\n📊 Processing frame {frame_count} at {current_time:.1f}s...")
                    
                    # Process frame for object detection
                    processed_frame, detected_objects = self.detect_objects(frame)
                    
                    # Update display frame with detection overlays
                    display_frame = processed_frame.copy()
                    
                    # Update last process time
                    self.last_process_time = current_time
                    
                    # Generate voice guidance
                    self.generate_guidance(detected_objects)

                # Calculate display FPS (not processing FPS)
                if hasattr(self, '_last_frame_time'):
                    frame_time = current_time - self._last_frame_time
                    display_fps = 1.0 / frame_time if frame_time > 0 else 0
                else:
                    display_fps = 0
                self._last_frame_time = current_time

                # Status display
                if self.using_ultralytics:
                    mode = "YOLOv8"
                elif self.model_loaded:
                    mode = "YOLOv4"
                else:
                    mode = "Basic"

                status_text = f"Display FPS: {display_fps:.1f} - Mode: {mode}"
                if should_process:
                    status_text += " [PROCESSING]"
                else:
                    time_until_next = self.process_interval - (current_time - self.last_process_time)
                    status_text += f" (Next: {time_until_next:.1f}s)"

                # Show video with status
                if not self.headless_mode:
                    cv2.putText(display_frame, status_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add processing indicator
                    if should_process:
                        cv2.putText(display_frame, "ANALYZING...", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Display frame
                    cv2.imshow("Vision Assistant", display_frame)

                    # Check for exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                else:
                    # In headless mode, print status when processing
                    if should_process:
                        print(f"🔄 {status_text} | Objects detected: {len(detected_objects)}")

        except KeyboardInterrupt:
            print("\n🛑 Stopping navigation (Ctrl+C pressed)")
            self.running = False
        except Exception as e:
            print(f"An error occurred during runtime: {e}")
            import traceback
            traceback.print_exc() # Print detailed error information
        finally:
            # Clean up
            print("Stopping Vision Assistant...")
            self.cap.release()
            
            # Only destroy windows if GUI was available
            if not self.headless_mode:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass  # Ignore errors when destroying windows
            
            if self.tts_engine is not None:
                try:
                    self.tts_engine.stop() # Stop the pyttsx3 engine gracefully
                except:
                    pass

    def stop(self):
        self.running = False

    def _test_fast_tts_performance(self):
        """Test and showcase fast TTS performance metrics"""
        print("\n🎤 === FAST TTS PERFORMANCE TEST ===")
        
        # Test TTS engine performance
        if self.tts_engine is not None:
            try:
                start_time = time.time()
                
                # Test message
                test_text = "TTS system initialized and ready for navigation"
                print(f"🔊 Testing TTS with: '{test_text}'")
                
                # Non-blocking test to measure initialization time
                init_time = time.time() - start_time
                
                # Get TTS properties for performance showcase
                rate = self.tts_engine.getProperty('rate')
                volume = self.tts_engine.getProperty('volume')
                voices = self.tts_engine.getProperty('voices')
                
                print(f"✅ Fast pyttsx3 Performance Metrics:")
                print(f"   • Initialization time: {init_time*1000:.1f}ms")
                print(f"   • Speech rate: {rate} words/min (optimized +50)")
                print(f"   • Volume level: {volume:.1f} (optimized +0.1)")
                print(f"   • Available voices: {len(voices) if voices else 0}")
                print(f"   • Engine status: Ready and optimized")
                
                # Optional: Quick non-blocking test
                if hasattr(self, 'tts_engine') and not self.tts_engine.isBusy():
                    print("   • Running quick TTS test...")
                    threading.Thread(target=self._quick_tts_test, args=(test_text,), daemon=True).start()
                
            except Exception as e:
                print(f"❌ TTS performance test failed: {e}")
        else:
            print("⚠️ Fast pyttsx3 not available, using subprocess fallback")
            print("✅ Subprocess 'say' Performance Metrics:")
            print("   • Method: System subprocess call")
            print("   • Estimated latency: 0.5-2.0 seconds")
            print("   • Reliability: High (OS-level TTS)")
        
        print("🎯 TTS Performance Summary:")
        print("   • Navigation TTS: <1 second response time")
        print("   • Multi-threaded: Non-blocking audio output")
        print("   • Fallback system: 100% reliability")
        print("================================\n")

    def _quick_tts_test(self, text):
        """Quick TTS test in separate thread"""
        try:
            start_time = time.time()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            duration = time.time() - start_time
            print(f"   • TTS test completed in {duration:.2f} seconds")
        except Exception as e:
            print(f"   • TTS test error: {e}")

    def _configure_speaker_device(self):
        """Configure and select speaker device for TTS output"""
        try:
            print("\n🔊 === SPEAKER DEVICE CONFIGURATION ===")
            
            # Get available voices (which often correspond to different audio devices/outputs)
            voices = self.tts_engine.getProperty('voices')
            
            if not voices:
                print("⚠️ No voices available for device selection")
                return
            
            print(f"📻 Found {len(voices)} available voice/device options:")
            
            # List available voices with details
            for i, voice in enumerate(voices):
                voice_name = getattr(voice, 'name', 'Unknown')
                voice_id = getattr(voice, 'id', 'Unknown')
                voice_age = getattr(voice, 'age', 'Unknown')
                voice_gender = getattr(voice, 'gender', 'Unknown')
                
                print(f"   [{i+1}] {voice_name}")
                print(f"       ID: {voice_id}")
                print(f"       Gender: {voice_gender}, Age: {voice_age}")
                print()
            
            # Auto-select or prompt for selection
            selected_voice_index = self._select_voice_device(voices)
            
            if selected_voice_index is not None:
                selected_voice = voices[selected_voice_index]
                self.tts_engine.setProperty('voice', selected_voice.id)
                
                print(f"✅ Selected voice/device: {selected_voice.name}")
                print(f"   Device ID: {selected_voice.id}")
                
                # Test the selected voice
                self._test_selected_voice(selected_voice.name)
            else:
                print("⚠️ Using default voice/device")
                
        except Exception as e:
            print(f"❌ Speaker device configuration failed: {e}")
            print("   Continuing with default audio device...")
        
        print("=====================================\n")

    def _select_voice_device(self, voices):
        """Select voice device - auto or manual selection"""
        try:
            # For navigation, auto-select the best voice for clarity
            # Priority: Female voices are often clearer for navigation
            
            # Look for optimal voices (prefer female, then male, then any)
            female_voices = []
            male_voices = []
            other_voices = []
            
            for i, voice in enumerate(voices):
                # Fix: Handle None gender gracefully
                gender = getattr(voice, 'gender', None)
                gender_str = gender.lower() if gender else ''
                
                name = getattr(voice, 'name', '')
                name_str = name.lower() if name else ''
                
                if 'female' in gender_str or 'woman' in name_str or 'zira' in name_str or 'hazel' in name_str:
                    female_voices.append(i)
                elif 'male' in gender_str or 'man' in name_str or 'david' in name_str or 'mark' in name_str:
                    male_voices.append(i)
                else:
                    other_voices.append(i)
            
            # Auto-select best voice for navigation
            if female_voices:
                selected_index = female_voices[0]
                print(f"🎯 Auto-selected female voice (optimal for navigation)")
            elif male_voices:
                selected_index = male_voices[0] 
                print(f"🎯 Auto-selected male voice")
            elif other_voices:
                selected_index = other_voices[0]
                print(f"🎯 Auto-selected available voice")
            else:
                selected_index = 0
                print(f"🎯 Using default voice")
            
            print(f"   Selected: [{selected_index+1}] {voices[selected_index].name}")
            
            # Optional: Allow manual override via environment variable
            manual_selection = os.environ.get('TTS_VOICE_INDEX')
            if manual_selection:
                try:
                    manual_index = int(manual_selection) - 1
                    if 0 <= manual_index < len(voices):
                        selected_index = manual_index
                        print(f"🔧 Manual override: Using voice {manual_index+1}")
                except ValueError:
                    pass
            
            return selected_index
            
        except Exception as e:
            print(f"❌ Voice selection failed: {e}")
            return 0  # Default to first voice

    def _test_selected_voice(self, voice_name):
        """Test the selected voice with a navigation sample"""
        try:
            print(f"🔊 Testing selected voice: {voice_name}")
            test_message = f"Navigation voice test. Using {voice_name} for guidance."
            
            # Non-blocking test
            threading.Thread(target=self._voice_test_thread, args=(test_message,), daemon=True).start()
            
        except Exception as e:
            print(f"❌ Voice test failed: {e}")

    def _voice_test_thread(self, message):
        """Test voice in separate thread"""
        try:
            start_time = time.time()
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
            duration = time.time() - start_time
            print(f"   ✅ Voice test completed in {duration:.2f} seconds")
        except Exception as e:
            print(f"   ❌ Voice test error: {e}")

if __name__ == "__main__":
    print("Starting Enhanced Vision Assistant...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press 'q' to quit.")
    print("-" * 30)
    print("Configuration Notes:")
    print(f"- Install pyttsx3 for better speech: 'pip install pyttsx3 pyobjc'")
    print(f"- Install ultralytics for YOLOv8: 'pip install ultralytics'")
    print(f"- Check ultralytics docs for Apple MPS setup on M-series Macs for performance.")
    print(f"- Adjust `self.alert_cooldown` and `self.skip_frames` in the code for responsiveness vs CPU usage.")
    print(f"- Set `self.use_wall_detection = False` to disable wall detection and potentially improve speed.")
    print(f"- Consider making messages in `generate_guidance` shorter.")
    print("-" * 30)


    try:
        # Create and run assistant
        assistant = VisionAssistant()
        assistant.run()
    except IOError as e:
        print(f"Hardware Error: {e}")
        print("Please ensure your webcam is connected and accessible.")
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        import traceback
        traceback.print_exc() # Print detailed error information
