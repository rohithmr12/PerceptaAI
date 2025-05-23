import json
import networkx as nx
import pyttsx3
import math
import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
import threading
import subprocess

# Constants
DEFAULT_YOLO_MODEL_PATH = "Nav/yolov8m.pt"

class NodeMapManager:
    def __init__(self, map_filepath):
        self.map_data = self._load_map(map_filepath)
        self.graph = nx.DiGraph()  # Use DiGraph for directed graph
        self._build_graph()

    def _load_map(self, filepath):
        with open(filepath, 'r') as f:
            map_data = json.load(f)
        return map_data

    def get_shortest_path(self, start_node, end_node):
        try:
            shortest_path = nx.shortest_path(self.graph, source=start_node, target=end_node, weight='weight')
            return shortest_path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound as e:
            print(f"Node not found: {e}")
            return None

    def get_node_data(self, node_id):
        return self.graph.nodes[node_id]

    def _build_graph(self):
        # First, add all nodes to the graph
        for node_data in self.map_data['nodes']:
            node_id = str(node_data['id'])
            self.graph.add_node(node_id, **node_data)

        # Second, add all edges to the graph
        for node_data in self.map_data['nodes']:
            node_id = str(node_data['id'])
            if 'outgoingLinks' in node_data:
                for link in node_data['outgoingLinks']:
                    end_node = str(link['endNode'])
                    # Calculate distance as weight (Euclidean distance)
                    try:
                        distance = self.calculate_distance(node_data, self.get_node_data(end_node))
                        self.graph.add_edge(node_id, end_node, weight=distance)
                    except KeyError:
                        print(f"Node {end_node} not found, skipping edge")
                    except nx.NetworkXError:
                        print(f"Node {end_node} not found, skipping edge")


    def calculate_distance(self, node1, node2):
        x1 = node1['x']
        y1 = node1['y']
        x2 = node2['x']
        y2 = node2['y']
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
class RouteTrackingManager:
    def __init__(self, node_map_manager, path):
        self.node_map_manager = node_map_manager
        self.path = path
        self.current_node_index = 0

    def get_next_node(self):
        if self.current_node_index < len(self.path) - 1:
            return self.path[self.current_node_index + 1]
        else:
            return None

    def get_direction(self, current_node, next_node):
        # Implement your direction logic here
        current_node_data = self.node_map_manager.get_node_data(current_node)
        next_node_data = self.node_map_manager.get_node_data(next_node)

        # Example: Calculate angle between nodes (requires x, y coordinates)
        angle = self.calculate_angle(current_node_data, next_node_data)
        angle = round(angle, 0)  # Round to 2 decimal places for better readability
        if angle < 0:
            angle += 360
        return f"Go straight at {angle} degrees"

    def calculate_angle(self, current_node, next_node):
        # Calculate the angle in radians
        angle_rad = math.atan2(next_node['y'] - current_node['y'], next_node['x'] - current_node['x'])

        # Convert radians to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def advance_to_next_node(self):
        if self.current_node_index < len(self.path) - 1:
            self.current_node_index += 1

class UserStatusManager:
    def __init__(self):
        self.in_intersection = False
        self.consecutive_in_intersection_count = 0
        self.consecutive_not_in_intersection_count = 0
        self.consecutive_in_intersection_threshold = 3
        self.consecutive_not_in_intersection_threshold = 5
        self.obstacle_detected = False # Track obstacle presence

    def update_status(self, is_in_intersection, obstacle_detected):
        self.in_intersection = is_in_intersection
        self.obstacle_detected = obstacle_detected
        
class PoiNameConverter:
    def __init__(self, map_data):
        # Define a dictionary to map node IDs to human-readable names
        self.name_map = {
            "node1": "the main entrance",
            "node5": "the reception desk",
            "520-C": "the elevator",
            # Add more mappings here
        }
        self.map_data = map_data

    def convert(self, node_id):
        # Convert a node ID to a human-readable name
        if node_id in self.name_map:
            return self.name_map[node_id]
        else:
            return f"node {node_id}"  # Default name if not found in the map

class FeedbackManager:
    def __init__(self, poi_converter_instance):
        self.engine = None
        self.poi_name_converter = poi_converter_instance
        
        # Initialize fast optimized pyttsx3 (simplified, no threading)
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Speed up TTS for navigation
            rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', rate + 50)  # Faster speech
            volume = self.engine.getProperty('volume')
            self.engine.setProperty('volume', min(volume + 0.1, 1.0))  # Slightly louder
            
            print("✅ Fast pyttsx3 initialized for indoor navigation (nav_core) - no threading")
        except Exception as e:
            print(f"⚠ pyttsx3 failed in nav_core: {e}")
            self.engine = None

    def speak(self, text):
        """Simple direct TTS - no threading needed"""
        if self.engine is not None:
            self.engine.say(text)
            self.engine.runAndWait()  # Fast and simple
        else:
            print(f"🔊 INDOOR NAV VOICE: {text}")  # Console fallback

    def generate_obstacle_warning(self, obstacle_type, obstacle_location):
        return f"Warning! {obstacle_type} detected {obstacle_location}."

    def generate_next_instruction(self, route_tracker, obstacle_type=None, obstacle_location=None):
        next_node = route_tracker.get_next_node()
        if next_node:
            direction = route_tracker.get_direction(route_tracker.path[route_tracker.current_node_index], next_node)
            next_node_name = self.poi_name_converter.convert(next_node)  # Convert node ID to name
            instruction = f"Go towards {next_node_name}. {direction}."
            if obstacle_type:
                instruction += " " + self.generate_obstacle_warning(obstacle_type, obstacle_location)
            return instruction
        else:
            return "You have arrived at your destination."

class IntersectionDetector:
    def __init__(self, model_path='yolov8m.pt', confidence_threshold=0.5):
        # Print the model path
        print(f"Loading YOLOv8 model from: {model_path}")
        # Load YOLOv8 model
        try:
            self.model = YOLO(model_path)  # Replace with your model path
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None  # Set model to None if loading fails
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image):
        if self.model is None:
            print("Model not loaded, skipping detection.")
            return []

        # Run YOLOv8 on the image
        results = self.model.predict(image, conf=self.confidence_threshold)

        # Extract bounding boxes and confidence scores
        objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                objects.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id
                })
        return objects

    def get_obstacle_in_path(self, objects):
        # Implement logic to determine if there's an obstacle in the user's path
        # This is a placeholder; replace with your actual obstacle detection logic
        for obj in objects:
            if obj['class_id'] == 3:  # Assuming class 3 is "person" (obstacle)
                return "person", "in the center"  # Example obstacle type and location
        return None, None

class NavigationCore:
    def __init__(self, map_filepath, start_node_id, end_node_id, yolo_model_path=DEFAULT_YOLO_MODEL_PATH):
        self.node_map_manager = NodeMapManager(map_filepath) # Can raise FileNotFoundError for map
        
        # Initialize PoiNameConverter with map_data for better naming
        poi_converter = PoiNameConverter(self.node_map_manager.map_data)
        self.feedback_manager = FeedbackManager(poi_converter) # Pass converter instance

        self.path = self.node_map_manager.get_shortest_path(str(start_node_id), str(end_node_id))
        if not self.path:
            message = f"Navigation Error: Could not find a path from '{start_node_id}' to '{end_node_id}' in map '{map_filepath}'."
            self.feedback_manager.speak(message) # Speak the error
            raise ValueError(message)
        
        self.route_tracking_manager = RouteTrackingManager(self.node_map_manager, self.path)
        self.user_status_manager = UserStatusManager()
        self.obstacle_detector = IntersectionDetector(model_path=yolo_model_path)

        self.current_node_id = self.route_tracking_manager.get_current_node()
        start_name = self.feedback_manager.poi_name_converter.convert(self.current_node_id)
        end_name = self.feedback_manager.poi_name_converter.convert(end_node_id)
        
        # Add missing attributes for compatibility
        self.user_x = None  # User's estimated x coordinate
        self.user_y = None  # User's estimated y coordinate  
        self.node_reached_threshold = 50  # Distance threshold to consider a node reached
        self.feedback_interval = 5  # Time interval between instructions (seconds)
        
        initial_message = f"Navigation system initialized. Route found from {start_name} to {end_name} via {len(self.path)} points."
        print(initial_message)
        self.feedback_manager.speak(initial_message)
        first_instruction = self.feedback_manager.generate_next_instruction(self.route_tracking_manager)
        self.feedback_manager.speak(first_instruction)

    def estimate_user_position(self, image):
        # Implement your logic to estimate the user's position based on the image
        # For now, let's simulate the user's position
        # Replace this with your actual position estimation logic
        # This is just a placeholder
        if self.user_x is None or self.user_y is None:
            # Initialize user position to the starting node
            start_node_data = self.node_map_manager.get_node_data(self.current_node_id)
            self.user_x = start_node_data['x']
            self.user_y = start_node_data['y']
        return self.user_x, self.user_y

    def calculate_distance_to_next_node(self):
        if self.route_tracking_manager.current_node_index < len(self.path) - 1:
            next_node = self.path[self.route_tracking_manager.current_node_index + 1]
            next_node_data = self.node_map_manager.get_node_data(next_node)
            return math.sqrt((next_node_data['x'] - self.user_x)**2 + (next_node_data['y'] - self.user_y)**2)
        else:
            return 0

    def has_reached_next_node(self):
        distance_to_next_node = self.calculate_distance_to_next_node()
        return distance_to_next_node <= self.node_reached_threshold

    def navigate(self, image):  # Pass the image to the navigate function
        # Estimate user position
        self.user_x, self.user_y = self.estimate_user_position(image)

        # Calculate distance to next node
        distance_to_next_node = self.calculate_distance_to_next_node()

        # Check if the user has reached the next node
        if self.has_reached_next_node():
            # Advance to the next node
            if self.route_tracking_manager.current_node_index < len(self.path) - 1:
                self.route_tracking_manager.advance_to_next_node()
                self.current_node_id = self.path[self.route_tracking_manager.current_node_index]
                print(f"Reached node: {self.current_node_id}")
            else:
                self.feedback_manager.speak("You have arrived at your destination.")
                return True  # Navigation complete

        # Detect objects in the image
        objects = self.obstacle_detector.detect_objects(image)

        # Determine if there is an obstacle in the path
        obstacle_type, obstacle_location = self.obstacle_detector.get_obstacle_in_path(objects)
        obstacle_detected = obstacle_type is not None

        # Update user status
        self.user_status_manager.update_status(False, obstacle_detected)

        # Draw bounding boxes on the image
        for obj in objects:
            box = obj['box']
            color = (0, 255, 0)  # Green for intersections
            if obj['class_id'] == 3:  # Red for obstacles
                color = (0, 0, 255)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Display the image with bounding boxes
        cv2.imshow('Navigation', image)
        cv2.waitKey(1)

        next_node = self.route_tracking_manager.get_next_node()

        # Provide feedback to guide the user
        instruction = self.feedback_manager.generate_next_instruction(self.route_tracking_manager, obstacle_type, obstacle_location)
        self.feedback_manager.speak(instruction)

        return False

    def run_navigation_loop(self):
        """Main navigation loop for continuous guidance"""
        print("🔄 Starting indoor navigation loop...")
        
        # Open a video capture object
        cap = cv2.VideoCapture(0)  # 0 for default camera
        
        if not cap.isOpened():
            error_msg = "Cannot open webcam for indoor navigation"
            print(f"❌ {error_msg}")
            self.feedback_manager.speak(error_msg)
            return
        
        try:
            while True:
                # Read a frame from the camera
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run navigation step
                navigation_complete = self.navigate(frame)
                if navigation_complete:
                    break
                
                # Check for quit condition (optional)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Delay between instructions
                time.sleep(self.feedback_interval)
                
        except Exception as e:
            error_msg = f"Navigation loop error: {e}"
            print(f"❌ {error_msg}")
            self.feedback_manager.speak(error_msg)
        finally:
            # Release the video capture object and close all windows
            cap.release()
            cv2.destroyAllWindows()
            print("🏁 Indoor navigation loop completed")

# Example usage (in main.py):
if __name__ == "__main__":
    try:
        navigation = NavigationCore("C:/Users/Rohith.MR/PERCEPTAAI/Navigation/intersection_detector/intersection_detector/nodemap/121-5-3.json", "node1", "node5")
        print("Path:", navigation.path) # Print the path

        # Open a video capture object
        cap = cv2.VideoCapture(0)  # 0 for default camera

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                break

            if navigation.navigate(frame):  # Pass the image to the navigate function
                break

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(navigation.feedback_interval) # Delay between instructions

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

    except ValueError as e:
        print(e)