import json
import networkx as nx
import pyttsx3
import math
# import torch # YOLO might handle torch internally
from ultralytics import YOLO
import cv2
import time
import os

# Default model path (can be overridden in IntersectionDetector)
# Ensure these paths are correct for your environment or provide them when calling tools.
DEFAULT_YOLO_MODEL_PATH = "Nav/yolov8m.pt" 
DEFAULT_MAP_FILE_PATH = "Nav/map_data/121-5-3.json" # Example path from notebook structure

class NodeMapManager:
    def __init__(self, map_filepath):
        self.map_filepath = map_filepath
        self.map_data = self._load_map(map_filepath)
        self.graph = nx.DiGraph()
        self._build_graph()

    def _load_map(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Map file not found: {filepath}")
        with open(filepath, 'r') as f:
            map_data = json.load(f)
        return map_data

    def get_shortest_path(self, start_node, end_node):
        try:
            shortest_path = nx.shortest_path(self.graph, source=str(start_node), target=str(end_node), weight='weight')
            return shortest_path
        except nx.NetworkXNoPath:
            print(f"No path found between {start_node} and {end_node} in map {self.map_filepath}")
            return None
        except nx.NodeNotFound as e:
            print(f"Node not found in graph for pathfinding: {e} (map: {self.map_filepath})")
            return None

    def get_node_data(self, node_id):
        node_id_str = str(node_id)
        if node_id_str in self.graph.nodes:
            return self.graph.nodes[node_id_str]
        else:
            raise nx.NodeNotFound(f"Node '{node_id_str}' not found in graph (map: {self.map_filepath}). Available nodes: {list(self.graph.nodes())[:10]}...")


    def _build_graph(self):
        for node_data in self.map_data['nodes']:
            node_id = str(node_data['id'])
            self.graph.add_node(node_id, **node_data)
        
        for node_data in self.map_data['nodes']:
            node_id = str(node_data['id'])
            if 'outgoingLinks' in node_data:
                for link in node_data['outgoingLinks']:
                    end_node_id = str(link['endNode'])
                    if not self.graph.has_node(end_node_id):
                        print(f"Warning (map: {self.map_filepath}): Edge references non-existent end node '{end_node_id}' from start node '{node_id}'. Skipping this edge.")
                        continue
                    
                    start_node_attrs = self.get_node_data(node_id)
                    end_node_attrs = self.get_node_data(end_node_id)
                    
                    distance = self.calculate_distance(start_node_attrs, end_node_attrs)
                    self.graph.add_edge(node_id, end_node_id, weight=distance)

    def calculate_distance(self, node1_attrs, node2_attrs):
        x1 = node1_attrs['x']
        y1 = node1_attrs['y']
        x2 = node2_attrs['x']
        y2 = node2_attrs['y']
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class RouteTrackingManager:
    def __init__(self, node_map_manager, path):
        self.node_map_manager = node_map_manager
        self.path = path
        self.current_node_index = 0

    def get_current_node(self):
        return self.path[self.current_node_index]

    def get_next_node(self):
        if self.current_node_index < len(self.path) - 1:
            return self.path[self.current_node_index + 1]
        return None

    def get_direction(self, current_node_id, next_node_id):
        current_node_data = self.node_map_manager.get_node_data(current_node_id)
        next_node_data = self.node_map_manager.get_node_data(next_node_id)
        angle = self.calculate_angle(current_node_data, next_node_data)
        angle = round(angle, 0)
        if angle < 0:
            angle += 360
        
        # Simplified directional cues
        if (0 <= angle < 22.5) or (337.5 <= angle < 360):
            return "go straight ahead"
        elif 22.5 <= angle < 67.5:
            return "bear slightly right"
        elif 67.5 <= angle < 112.5:
            return "turn right"
        elif 112.5 <= angle < 157.5:
            return "bear sharply right"
        elif 157.5 <= angle < 202.5:
            return "make a U-turn or go back" # Indicates going mostly backward
        elif 202.5 <= angle < 247.5:
            return "bear sharply left"
        elif 247.5 <= angle < 292.5:
            return "turn left"
        elif 292.5 <= angle < 337.5:
            return "bear slightly left"
        return f"head at {angle} degrees"


    def calculate_angle(self, current_node_attrs, next_node_attrs):
        angle_rad = math.atan2(next_node_attrs['y'] - current_node_attrs['y'], next_node_attrs['x'] - current_node_attrs['x'])
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def advance_to_next_node(self):
        if self.current_node_index < len(self.path) - 1:
            self.current_node_index += 1
            return True
        return False

class UserStatusManager: # Simplified
    def __init__(self):
        self.obstacle_detected = False

    def update_status(self, obstacle_detected):
        self.obstacle_detected = obstacle_detected

class PoiNameConverter:
    def __init__(self, map_data=None):
        self.name_map = { # Default common names
            "node1": "the main entrance",
            "node5": "the reception desk",
            "520-C": "the elevator",
        }
        if map_data and 'pois' in map_data: # Load POIs from map if available
            for poi in map_data['pois']:
                if 'id' in poi and 'name' in poi:
                    self.name_map[str(poi['id'])] = poi['name']
        elif map_data and 'nodes' in map_data: # Fallback to node IDs if they have names
             for node in map_data['nodes']:
                if 'id' in node and 'name' in node and str(node['id']) not in self.name_map : # Prioritize explicit POIs
                    self.name_map[str(node['id'])] = node['name']


    def convert(self, node_id):
        return self.name_map.get(str(node_id), f"checkpoint {node_id}")

class FeedbackManager:
    def __init__(self, poi_converter_instance):
        self.engine = None
        self.poi_name_converter = poi_converter_instance
        try:
            self.engine = pyttsx3.init()
            # Optionally set voice properties here if needed
            # voices = self.engine.getProperty('voices')
            # self.engine.setProperty('voice', voices[1].id) # Example: Set a specific voice
            self.engine.setProperty('rate', 160) # Adjust speech rate
        except Exception as e:
            print(f"Warning: Failed to initialize pyttsx3 TTS engine: {e}. Voice feedback will be printed to console.")

    def speak(self, text):
        if not text: return
        print(f"TTS: {text}")
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error during pyttsx3 speech: {e}")
        # If engine is None, text is already printed.

    def generate_obstacle_warning(self, obstacle_type, obstacle_location):
        return f"Caution! {obstacle_type} detected {obstacle_location}."

    def generate_next_instruction(self, route_tracker, obstacle_type=None, obstacle_location=None):
        current_node_id = route_tracker.get_current_node()
        next_node_id = route_tracker.get_next_node()

        if next_node_id:
            direction = route_tracker.get_direction(current_node_id, next_node_id)
            next_node_name = self.poi_name_converter.convert(next_node_id)
            instruction = f"{direction} towards {next_node_name}."
            if obstacle_type:
                instruction += " " + self.generate_obstacle_warning(obstacle_type, obstacle_location)
            return instruction
        else: # At destination
            destination_name = self.poi_name_converter.convert(current_node_id)
            return f"You have arrived at {destination_name}."

class ObstacleDetector: # Renamed from IntersectionDetector for clarity
    def __init__(self, model_path=DEFAULT_YOLO_MODEL_PATH, confidence_threshold=0.4):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        print(f"Attempting to load YOLOv8 model from: {self.model_path}")
        try:
            if not os.path.exists(self.model_path):
                # Try a common relative path if the specific one fails (e.g. if Nav/ is not in root)
                alt_path = os.path.join("Nav", os.path.basename(self.model_path))
                if os.path.exists(alt_path):
                    self.model_path = alt_path
                    print(f"Using alternative model path: {self.model_path}")
                else:
                    raise FileNotFoundError(f"YOLO model not found at {model_path} or {alt_path}")
            
            self.model = YOLO(self.model_path)
            print("YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model from '{self.model_path}': {e}. Obstacle detection will be disabled.")
            # self.model remains None

    def detect_obstacles(self, image_frame):
        if not self.model:
            return None, None # type, location

        results = self.model.predict(image_frame, conf=self.confidence_threshold, verbose=False)
        
        detected_obstacles = []
        for result in results:
            boxes = result.boxes
            names = result.names # Class names
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = names.get(class_id, "unknown object")
                
                # Focus on common obstacles like 'person', 'chair', 'table', etc.
                # Add more class_names as needed for your environment
                if class_name in ["person", "chair", "table", "backpack", "suitcase", "bottle"]: 
                    # Basic location estimation (center, left, right)
                    x1, _, x2, _ = box.xyxy[0]
                    img_width = image_frame.shape[1]
                    box_center_x = (x1 + x2) / 2
                    location = "ahead"
                    if box_center_x < img_width / 3:
                        location = "to your left"
                    elif box_center_x > 2 * img_width / 3:
                        location = "to your right"
                    detected_obstacles.append({"type": class_name, "location": location})

        if detected_obstacles:
            # Prioritize or summarize if multiple obstacles
            # For now, just report the first one clearly
            first_obstacle = detected_obstacles[0]
            return first_obstacle["type"], first_obstacle["location"]
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
        self.obstacle_detector = ObstacleDetector(model_path=yolo_model_path)

        self.current_node_id = self.route_tracking_manager.get_current_node()
        start_name = self.feedback_manager.poi_name_converter.convert(self.current_node_id)
        end_name = self.feedback_manager.poi_name_converter.convert(end_node_id)
        
        initial_message = f"Navigation system initialized. Route found from {start_name} to {end_name} via {len(self.path)} points."
        print(initial_message)
        self.feedback_manager.speak(initial_message)
        first_instruction = self.feedback_manager.generate_next_instruction(self.route_tracking_manager)
        self.feedback_manager.speak(first_instruction)


    def run_navigation_loop(self):
        cap = cv2.VideoCapture(0) # Default camera
        if not cap.isOpened():
            err_msg = "Critical Error: Cannot open webcam. Navigation cannot proceed."
            print(err_msg)
            self.feedback_manager.speak(err_msg)
            return False

        print("Navigation loop started. Press 'q' in OpenCV window to stop.")
        instruction_interval = 7 # seconds between proactive instructions if no obstacles
        last_instruction_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.feedback_manager.speak("Webcam error: Failed to capture image.")
                    print("Error: Failed to capture image from webcam.")
                    time.sleep(1) # Avoid rapid error messages
                    continue

                # Obstacle detection
                obstacle_type, obstacle_location = self.obstacle_detector.detect_obstacles(frame)
                self.user_status_manager.update_status(obstacle_detected=(obstacle_type is not None))

                current_time = time.time()
                give_instruction_now = False

                if self.user_status_manager.obstacle_detected:
                    obstacle_warning = self.feedback_manager.generate_obstacle_warning(obstacle_type, obstacle_location)
                    self.feedback_manager.speak(obstacle_warning)
                    # Potentially re-evaluate route or wait, for now just warn and pause
                    time.sleep(3) # Brief pause for user to react to obstacle
                    give_instruction_now = True # Give next step after obstacle warning

                if give_instruction_now or (current_time - last_instruction_time >= instruction_interval) :
                    # Simulate advancing to the next node for this version
                    # In a real system, this would be based on localization/CV
                    if not self.route_tracking_manager.advance_to_next_node():
                        # Already at the last node or path ended
                        final_instruction = self.feedback_manager.generate_next_instruction(self.route_tracking_manager) # Should be "You have arrived"
                        self.feedback_manager.speak(final_instruction)
                        print("Navigation finished: Arrived at destination.")
                        break 
                    
                    self.current_node_id = self.route_tracking_manager.get_current_node()
                    # print(f"Debug: Advanced to node: {self.current_node_id}")
                    
                    instruction = self.feedback_manager.generate_next_instruction(
                        self.route_tracking_manager, 
                        None, # Obstacle already handled for this turn if detected
                        None
                    )
                    self.feedback_manager.speak(instruction)
                    last_instruction_time = current_time

                # Display frame (optional, for debugging; can be commented out)
                # cv2.imshow('Navigation View', frame)
                # key = cv2.waitKey(100) & 0xFF # ~10 FPS processing, adjust as needed
                # if key == ord('q'):
                #     self.feedback_manager.speak("Navigation stopped by user.")
                #     print("Navigation loop quit by 'q' key press.")
                #     break
                # elif key == ord('n'): # Simulate reaching next node
                #     print("Debug: 'n' key pressed, forcing next instruction.")
                #     last_instruction_time = 0 # Force instruction

                time.sleep(0.1) # Small delay to prevent hogging CPU if not displaying video

        except Exception as e:
            error_msg = f"An unexpected error occurred during navigation: {e}"
            print(error_msg)
            self.feedback_manager.speak(error_msg)
            return False
        finally:
            cap.release()
            # cv2.destroyAllWindows() # Only if imshow was used
            print("Navigation loop ended.")
        return True

# Example of how NavigationCore might be called (for testing nav_core.py directly)
if __name__ == "__main__":
    print("Testing NavigationCore standalone...")
    # Ensure Nav/map_data/121-5-3.json and Nav/yolov8m.pt exist or update paths
    # Create a dummy map file if it doesn't exist for basic testing
    dummy_map_path = DEFAULT_MAP_FILE_PATH
    if not os.path.exists(os.path.dirname(dummy_map_path)):
        os.makedirs(os.path.dirname(dummy_map_path), exist_ok=True)

    if not os.path.exists(dummy_map_path):
        print(f"Creating dummy map file at {dummy_map_path} for testing...")
        dummy_map_content = {
            "nodes": [
                {"id": "node1", "name": "Entrance", "x": 0, "y": 0, "outgoingLinks": [{"endNode": "node2"}]},
                {"id": "node2", "name": "Hallway Point", "x": 10, "y": 0, "outgoingLinks": [{"endNode": "node3"}]},
                {"id": "node3", "name": "Reception Area", "x": 20, "y": 0}
            ],
            "pois": [
                {"id": "node3", "name": "the Reception Desk"}
            ]
        }
        with open(dummy_map_path, "w") as f:
            json.dump(dummy_map_content, f)
    
    # Create a dummy YOLO model file if it doesn't exist (YOLO will fail to load, but code should handle)
    dummy_yolo_path = DEFAULT_YOLO_MODEL_PATH
    if not os.path.exists(os.path.dirname(dummy_yolo_path)):
        os.makedirs(os.path.dirname(dummy_yolo_path), exist_ok=True)
    if not os.path.exists(dummy_yolo_path):
        print(f"Creating empty dummy YOLO model at {dummy_yolo_path} (obstacle detection will be off).")
        with open(dummy_yolo_path, "w") as f:
            f.write("dummy yolo model placeholder") # Content doesn't matter, YOLO checks existence/format

    try:
        nav_system = NavigationCore(
            map_filepath=dummy_map_path, # Use dummy or your actual map
            start_node_id="node1", 
            end_node_id="node3", # Ensure this node exists in your map
            yolo_model_path=dummy_yolo_path # Use dummy or your actual model
        )
        nav_system.run_navigation_loop()
    except ValueError as e:
        print(f"Setup Error: {e}")
    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except Exception as e:
        print(f"Runtime Error: {e}") 