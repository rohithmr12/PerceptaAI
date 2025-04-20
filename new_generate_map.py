import cv2
import numpy as np
import argparse
import os
import json
import math
import yaml # Added
import matplotlib # Added
from skimage import morphology # Added
from skimage import measure, color # Added
from scipy.spatial.distance import cdist # Added
import matplotlib.pyplot as plt # Added
from concurrent.futures import ThreadPoolExecutor # Added (Though used sequentially here)
import time # Added for timer decorator
import threading # For background task

# --- Optional Dependencies for Analysis/Suggestions ---
ANALYSIS_AVAILABLE = False
OCR_AVAILABLE = False
try:
    from skimage import morphology
    from scipy.spatial.distance import cdist
    ANALYSIS_AVAILABLE = True
    print("INFO: Analysis libraries (skimage, scipy) found.")
    try:
        # easyocr requires torch
        import easyocr
        import torch
        OCR_AVAILABLE = True
        print("INFO: easyocr library found. OCR suggestions enabled.")
    except ImportError:
        print("\nWARNING: easyocr or PyTorch library not found.")
        print("         Install them ('pip install easyocr torch torchvision torchaudio')")
        print("         to enable OCR-based POI suggestions. Continuing without OCR.\n")
except ImportError:
    print("\nWARNING: Analysis libraries (skimage, scipy) not found.")
    print("         Install them ('pip install scikit-image scipy')")
    print("         to enable automated intersection/POI suggestions.")
    print("         Continuing with manual mapping mode only.\n")
    ANALYSIS_AVAILABLE = False # Ensure it's false if skimage/scipy missing

# --- Optional Dependencies for Config/Threading ---
YAML_AVAILABLE = False
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("WARNING: PyYAML library not found ('pip install pyyaml').")
    print("         External configuration loading might fail if MapRecognizer relies on it.")
    print("         Using default internal settings.\n")

# --- Matplotlib backend ---
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

# --- Default Configuration (merged from interactive tool and MapRecognizer defaults) ---
# Structure matches MapRecognizer's config.yml for easier integration
DEFAULT_CONFIG = {
    "display_parameters": { # From interactive tool
        "node_color": [0, 255, 0],
        "node_radius": 5,
        "link_color": [255, 0, 0],
        "highlight_color": [0, 255, 255],
        "text_color": [255, 255, 255],
        "max_click_distance_node": 15,
        "suggestion_intersection_color": [0, 0, 255],
        "suggestion_poi_color": [255, 0, 255],
        "suggestion_radius": 4,
        "max_suggestion_snap_distance": 20,
    },
    "getpath_parameters": {
      "lower_red": [30, 30, 85], "upper_red": [50, 50, 120],
      "binary_lower_threshold": 230, "binary_upper_threshold": 255,
      "kernel_size": 5, "CC_connectivity": 4, "dilated_iteration": 4
    },
    "extract_corridor_parameters": {
      "canny_lower": 50, "canny_upper": 150, "kernel_size": 5,
      "canny_2_lower": 100, "canny_2_upper": 200, "dilation_iteration": 1,
      "erosion_iteration": 1, "area_threshold": 1000, "addWeighted_alpha": 0.5,
      "binary_threshold": 0.5, "area_threshold_2": 100,
      "measure_connectivity": 8 # <-- CHANGE THIS VALUE TO 8 (or 4)
    },
    "skeletonize_parameters": {
      "scaler": 9, "kernel1_size": 3, "kernel2_size": 4,
      "dilate_iteration": 4, "erode_iteration": 4
    },
    "extract_corner_parameters": {
      "block_size": 3, "ksize": 3, "k": 0.05,
      # Threshold logic from MapRecognizer: 0.01 * max
      "corner_threshold_ratio": 0.01
    },
    "delete_node_parameters": { # Merged into corner extraction logic
      "min_distance": 6
    },
    "extract_connection_parameters": {
      "threshold_value": 128, "theta": 0, # theta seems unused in MapRecognizer logic provided
      "threshold_node_theta": 140, "min_node_distance": 10
    },
    "find_endpoint_parameters": { # Used within connection finding
      "max_explore_distance": 1000
    },
    "ocr_parameters": { # Parameters for the OCR class
        "height": 480, # Target height for OCR bounding box resizing
        "width": 640,  # Target width for OCR bounding box resizing
        "ocr_langs": ['ja', 'en'] # Languages for easyocr
    },
    "extract_pois_parameters": { # Used after OCR
      "cnt": -1, # Initial counter for POIs
      "width": 640, # Reference width for resizing POI coords (should match OCR params)
      "height": 480 # Reference height for resizing POI coords (should match OCR params)
    },
    "poi_merge_parameters": {
      "poi_threshold": 40
    }
}

# --- Display Size Limits ---
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900

# --- Global State Variables ---
config = DEFAULT_CONFIG.copy() # Start with defaults
original_image = None
display_image = None
nodes = [] # Interactive tool nodes (used for drawing)
links = [] # Interactive tool links (used for drawing)
node_counter = 1 # Interactive tool node ID counter

mode = 'NODE'
scale_points_orig = []
scale_pixels_per_meter = None
link_start_node = None

resize_factor = 1.0
original_h, original_w = 0, 0
display_h, display_w = 0, 0

# --- Suggestion / MapRecognizer Results Variables ---
# --- Suggestion / MapRecognizer Results Variables ---
suggested_intersections_orig = [] # List of [x_orig, y_orig] for display
suggested_pois_orig = []          # List of [x_orig, y_orig, text] for display
maprec_nodes = []                 # Store full MapRecognizer nodes list for saving
maprec_pois = []                  # Store full MapRecognizer POIs list for saving
analysis_thread = None
analysis_in_progress = False
analysis_complete = False
show_suggestions = True
maprec_config = None # To store loaded config from MapRecognizer path
# --- ADDED FOR SKELETON OVERLAY ---
global_skeleton_img = None       # To store the computed skeleton (downscaled)
global_scaler_used = 1           # Scaler used for the stored skeleton
show_skeleton = False         # Toggle for skeleton visibility
# ---> END ADD <---
# --- MapRecognizer Paths (from its __init__) ---
# These will be used for saving if `save_map` is adapted
SAVE_PATH = "outputs"
IMG_PROC_PATH = os.path.join(SAVE_PATH, 'process_img')
JSON_SAVE_PATH = os.path.join(SAVE_PATH, 'json')
# Create directories if they don't exist (at the start)
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(IMG_PROC_PATH, exist_ok=True)
os.makedirs(JSON_SAVE_PATH, exist_ok=True)
# Define image-specific subdirs later when image name is known


# === Helper Functions (Adapted from MapRecognizer/util) ===

def timer_decorator(func): # From util
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'DEBUG: Function {func.__name__} took {end - start:.4f} seconds.')
        return result
    return wrapper

def theta_calc(a, b, c): # From util
    """Calculates angle ABC in degrees."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    vec_ba = a - b; vec_bc = c - b
    len_ba = np.linalg.norm(vec_ba); len_bc = np.linalg.norm(vec_bc)
    if len_ba == 0 or len_bc == 0: return 180.0 # Avoid division by zero if points coincide
    inner = np.inner(vec_ba, vec_bc)
    cos_theta = inner / (len_ba * len_bc)
    # Clip cos_theta to [-1, 1] due to potential floating point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rad = np.arccos(cos_theta)
    return np.rad2deg(rad)

# --- MapRecognizer Core Logic Functions (Adapted to run in memory) ---

@timer_decorator
def run_getpath(floormap_image_input, cfg):
    """Finds the largest non-red area, assumed to be the path."""
    floormap_image = floormap_image_input.copy() # Work on a copy
    params = cfg["getpath_parameters"]
    lower_red = np.array(params["lower_red"]); upper_red = np.array(params["upper_red"])
    kernel_size = params["kernel_size"]; kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_iteration = params["dilated_iteration"]; CC_connectivity = params["CC_connectivity"]
    binary_lower_threshold = params["binary_lower_threshold"]
    binary_upper_threshold = params["binary_upper_threshold"]

    mask = cv2.inRange(floormap_image, lower_red, upper_red)
    floormap_image[mask > 0] = [255, 255, 255] # Make red areas white

    grayscale = cv2.cvtColor(floormap_image, cv2.COLOR_BGR2GRAY)
    # Dilate might merge paths slightly, then threshold
    dilated_image = cv2.dilate(grayscale, kernel, iterations=dilated_iteration)
    _, binary_image = cv2.threshold(dilated_image, binary_lower_threshold, binary_upper_threshold, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the largest connected component (usually the floor plan itself)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=CC_connectivity)
    if num_labels <= 1: return np.zeros_like(binary_image) # No components found
    largest_area_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 # Index relative to stats[1:]
    path_mask = np.uint8(labels == largest_area_index) * 255
    return path_mask # Return the mask of the largest component

@timer_decorator
def run_extract_corridor(floormap_image_input, cfg):
    """Extracts labeled image of corridors/rooms."""
    floormap_image = floormap_image_input.copy()
    params = cfg["extract_corridor_parameters"]
    canny_lower = params["canny_lower"]; canny_upper = params["canny_upper"]
    kernel_size = params["kernel_size"]; kernel = np.ones((kernel_size, kernel_size), np.uint8)
    canny_2_lower = params["canny_2_lower"]; canny_2_upper = params["canny_2_upper"]
    dilation_iteration = params["dilation_iteration"]; erosion_iteration = params["erosion_iteration"]
    area_threshold = params["area_threshold"]; addWeighted_alpha = params["addWeighted_alpha"]
    binary_threshold_ratio = params["binary_threshold"] # Renamed to avoid clash
    area_threshold_2 = params["area_threshold_2"]; measure_connectivity = params["measure_connectivity"]

    # This part seems complex and might need tuning based on image types
    # Simplified version: Use Canny + Morphology + Connected Components
    gray = cv2.cvtColor(floormap_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_lower, canny_upper)

    # Close gaps in edges
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2) # More closing might help

    # Find contours and fill them to get potential room areas
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Invert mask to get potential walkable areas (assuming white background)
    walkable_mask = cv2.bitwise_not(mask)

    # Connected components on the walkable areas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(walkable_mask, connectivity=measure_connectivity)

    # Filter components by area
    filtered_labels = np.zeros_like(labels)
    if num_labels > 1:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > area_threshold_2:
                 filtered_labels[labels == i] = i # Keep original label number

    # Relabel consecutively if needed, or just return filtered_labels
    # For simplicity, return the labels image potentially with gaps
    return filtered_labels # Return the labeled image

@timer_decorator
def run_skeletonize(path_mask_input, cfg):
    """Skeletonizes the input path mask."""
    params = cfg["skeletonize_parameters"]
    scaler = max(1, int(params["scaler"]))
    kernel1_size = max(1, int(params["kernel1_size"])); kernel2_size = max(1, int(params["kernel2_size"]))
    kernel1 = np.ones((kernel1_size, kernel1_size), np.uint8)
    kernel2 = np.ones((kernel2_size, kernel2_size), np.uint8)
    dilate_iteration = params["dilate_iteration"]; erode_iteration = params["erode_iteration"]

    if path_mask_input is None or path_mask_input.sum() == 0:
        print("Warning: Empty path mask provided to skeletonize.")
        return None, scaler # Return None and scaler

    original_height, original_width = path_mask_input.shape[:2]
    target_width = original_width // scaler; target_height = original_height // scaler
    if target_width <=0 or target_height <= 0:
        print(f"Warning: Image too small ({original_width}x{original_height}) for scaler {scaler}. Using scaler=1.")
        scaler = 1; target_width = original_width; target_height = original_height

    # Resize the path mask (binary image)
    resized_image = cv2.resize(path_mask_input, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # Perform morphological operations on resized image
    dilated_image = cv2.dilate(resized_image, kernel2, iterations=dilate_iteration)
    eroded_image = cv2.erode(dilated_image, kernel1, iterations=erode_iteration)

    # Skeletonize using scikit-image
    # Ensure input is boolean
    skeleton = morphology.skeletonize(eroded_image > 128)
    skeleton_uint8 = (skeleton * 255).astype('uint8')
    return skeleton_uint8, scaler # Return skeleton and the scaler used

def count_outgoing_lines(image, point_yx):
    """Counts non-zero neighbors around a point (y, x)."""
    y, x = point_yx
    h, w = image.shape
    count = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and image[ny, nx] > 0:
                count += 1
    return count

def delete_node(corners_yx, skeleton_image, cfg):
    """Filters corners based on proximity and skeleton presence."""
    params_del = cfg["delete_node_parameters"]
    min_distance = params_del["min_distance"]
    params_corner = cfg["extract_corner_parameters"] # Needed? Not directly used here

    if corners_yx is None or len(corners_yx) == 0: return []

    # 1. Ensure corners are actually on the skeleton
    h, w = skeleton_image.shape
    valid_corners_yx = []
    for y, x in corners_yx:
        if 0 <= y < h and 0 <= x < w and skeleton_image[y, x] > 0:
            valid_corners_yx.append([y, x])
    corners_on_skeleton_yx = np.array(valid_corners_yx)

    if len(corners_on_skeleton_yx) <= 1: return corners_on_skeleton_yx.tolist()

    # 2. Filter based on distance (NMS - Non-Maximum Suppression based on distance)
    dist_matrix = cdist(corners_on_skeleton_yx, corners_on_skeleton_yx)
    np.fill_diagonal(dist_matrix, np.inf)

    keep_indices = np.ones(len(corners_on_skeleton_yx), dtype=bool)
    corner_strengths = [] # Placeholder if strength calculation is needed

    # Simple NMS: iterate and suppress neighbours
    # A more robust approach might consider corner strength if available (e.g., from Harris response)
    for i in range(len(corners_on_skeleton_yx)):
        if keep_indices[i]:
            # Find indices of points too close to point i
            close_indices = np.where(dist_matrix[i] < min_distance)[0]
            # Compare outgoing lines for close points (if needed, similar to MapRecognizer logic)
            # For simplicity here, just suppress neighbours
            keep_indices[close_indices] = False
            # Ensure point i itself is kept
            keep_indices[i] = True

    final_corners_yx = corners_on_skeleton_yx[keep_indices]
    return final_corners_yx.tolist()

@timer_decorator
def run_extract_corner(skeleton_image, cfg):
    """Detects and filters corners on the skeleton."""
    if skeleton_image is None: return []
    params = cfg["extract_corner_parameters"]
    block_size = params["block_size"]; ksize = params["ksize"]; k = params["k"]
    threshold_ratio = params["corner_threshold_ratio"]

    # Ensure ksize is odd
    ksize = ksize if ksize % 2 != 0 else ksize + 1
    ksize = max(1, ksize)
    block_size = max(2, block_size) # Harris needs >= 2

    try:
        # Harris corner detection needs float32 input
        skeleton_float = np.float32(skeleton_image > 128)
        harris_response = cv2.cornerHarris(skeleton_float, block_size, ksize, k)

        # Threshold based on max response
        threshold = max(threshold_ratio * harris_response.max(), 1e-7)
        # Get coordinates where response > threshold -> [[y1, x1], [y2, x2], ...]
        corners_yx = np.argwhere(harris_response > threshold)

        # Filter corners using delete_node logic
        filtered_corners_yx = delete_node(corners_yx, skeleton_image, cfg)
        return filtered_corners_yx # List of [y, x] on skeleton

    except Exception as e:
        print(f"Error during corner extraction: {e}")
        return []

def find_endpoint(start_node_index, start_yx_skel, initial_xy_skel, binary_skel_image, visited_mask, all_intersections_yx_skel, connections_list, max_explore_dist):
    """Traces a path from a starting point until an intersection or end is found."""
    current_yx = initial_xy_skel[::-1] # Use (y, x) internally
    start_y, start_x = start_yx_skel
    q = [(current_yx[0], current_yx[1])] # Queue for BFS-like exploration (y, x)
    visited_mask[current_yx[0], current_yx[1]] = True
    path_pixels = [current_yx]
    h, w = binary_skel_image.shape
    found_connection = False

    while q:
        cy, cx = q.pop(0)
        path_pixels.append((cy, cx))

        # Check if current point is another intersection (but not the starting one and sufficiently far)
        dist_from_start = np.linalg.norm(np.array([cy, cx]) - np.array(start_yx_skel))
        for idx, inter_yx in enumerate(all_intersections_yx_skel):
            inter_y, inter_x = inter_yx
            if cy == inter_y and cx == inter_x and dist_from_start > 5: # Found another intersection (threshold distance)
                # Record connection: (start_node_idx, end_node_idx, list_of_path_pixels)
                connections_list.append((start_node_index, idx, path_pixels))
                found_connection = True
                break # Stop tracing this path
        if found_connection: continue # Move to next path from start node

        # Explore neighbors
        neighbors_found = 0
        next_point = None
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                ny, nx = cy + dy, cx + dx
                # Check bounds and if it's a valid, unvisited skeleton pixel
                if 0 <= ny < h and 0 <= nx < w and \
                   binary_skel_image[ny, nx] > 0 and not visited_mask[ny, nx]:
                    visited_mask[ny, nx] = True
                    next_point = (ny, nx)
                    neighbors_found += 1
                    q.append((ny, nx)) # Add to queue for further exploration
                    break # Found one valid neighbor, move to it
            if next_point: break # Move to the first found neighbor

        if neighbors_found == 0 and not found_connection:
            # Reached end of a path (dead end), potentially store if needed, but MapRecognizer focuses on intersection links
            # print(f"Path from {start_node_index} ended at dead end ({cy},{cx})")
            pass

        # Limit exploration distance (optional)
        if len(path_pixels) > max_explore_dist:
            # print(f"Path from {start_node_index} exceeded max distance")
            break


def find_connections(intersections_yx_skel, binary_skel_image, cfg):
    """Find connections between intersection points on the skeleton."""
    params_conn = cfg["extract_connection_parameters"]
    params_find = cfg["find_endpoint_parameters"]
    max_explore_distance = params_find["max_explore_distance"]
    threshold_node_theta = params_conn["threshold_node_theta"]
    min_node_distance_sq = params_conn["min_node_distance"]**2 # Use squared distance

    connections_dict = {i: [] for i in range(len(intersections_yx_skel))}
    h, w = binary_skel_image.shape

    for i, start_yx in enumerate(intersections_yx_skel):
        start_y, start_x = start_yx
        # Explore neighbors of the intersection point
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                ny, nx = start_y + dy, start_x + dx

                if 0 <= ny < h and 0 <= nx < w and binary_skel_image[ny, nx] > 0:
                    # Start tracing from this neighbor
                    visited = np.zeros_like(binary_skel_image, dtype=bool)
                    visited[start_y, start_x] = True # Mark intersection as visited for this trace
                    
                    q = [(ny, nx)] # BFS queue (y, x)
                    visited[ny, nx] = True
                    path = [(ny, nx)]
                    found_end = False
                    
                    while q:
                        cy, cx = q.pop(0)
                        
                        # Check if current point is another intersection
                        dist_from_start = np.linalg.norm(np.array([cy,cx]) - np.array(start_yx))
                        is_intersection = False
                        end_node_index = -1
                        for j, inter_yx in enumerate(intersections_yx_skel):
                            if i != j and cy == inter_yx[0] and cx == inter_yx[1] and dist_from_start > 5: # Found a *different* intersection
                                is_intersection = True
                                end_node_index = j
                                break
                        
                        if is_intersection:
                             if end_node_index not in connections_dict[i]:
                                 connections_dict[i].append(end_node_index)
                             if i not in connections_dict[end_node_index]:
                                 connections_dict[end_node_index].append(i) # Add reciprocal link
                             found_end = True
                             break # Stop tracing this path

                        # If not intersection, explore neighbors
                        neighbor_found = False
                        for ddy in [-1, 0, 1]:
                            for ddx in [-1, 0, 1]:
                                if ddx == 0 and ddy == 0: continue
                                nny, nnx = cy + ddy, cx + ddx
                                if 0 <= nny < h and 0 <= nnx < w and \
                                   binary_skel_image[nny, nnx] > 0 and not visited[nny, nnx]:
                                    visited[nny, nnx] = True
                                    q.append((nny, nnx))
                                    path.append((nny, nnx))
                                    neighbor_found = True
                                    break # Explore one step
                            if neighbor_found: break
                        
                        if not neighbor_found and not found_end: # Dead end
                            break 
                        if len(path) > max_explore_distance: # Limit path length
                            break
                    # End of while q loop (path tracing)
            # End of neighbor exploration loop
    # End of intersection loop

    # --- Filter connections based on angle and distance (like MapRecognizer) ---
    ar_corners_yx = np.array(intersections_yx_skel) # [[y, x], ...]
    nodes_to_delete = set()
    
    # Filter corridor nodes (straight connections)
    for i, connected_indices in connections_dict.items():
        if i in nodes_to_delete: continue
        if len(connected_indices) == 2:
            idx0, idx1 = connected_indices[0], connected_indices[1]
            # Calculate angle (y,x coords are fine for angle calc)
            theta = theta_calc(ar_corners_yx[idx0], ar_corners_yx[i], ar_corners_yx[idx1])
            if theta > threshold_node_theta:
                # print(f"DEBUG: Deleting node {i} (corridor node, theta={theta:.1f}) connected to {idx0}, {idx1}")
                nodes_to_delete.add(i)
                # Reconnect neighbors
                if idx0 not in nodes_to_delete and i in connections_dict.get(idx0,[]): connections_dict[idx0].remove(i); connections_dict[idx0].append(idx1)
                if idx1 not in nodes_to_delete and i in connections_dict.get(idx1,[]): connections_dict[idx1].remove(i); connections_dict[idx1].append(idx0)


    # Filter endpoints too close to intersections
    for i, connected_indices in list(connections_dict.items()): # Iterate copy
         if i in nodes_to_delete: continue
         if len(connected_indices) == 1:
             neighbor_idx = connected_indices[0]
             if neighbor_idx in nodes_to_delete: continue # Skip if neighbor is already deleted

             dist_sq = np.sum((ar_corners_yx[i] - ar_corners_yx[neighbor_idx])**2)
             if dist_sq < min_node_distance_sq:
                 # print(f"DEBUG: Deleting node {i} (too close to {neighbor_idx}, dist_sq={dist_sq:.1f})")
                 nodes_to_delete.add(i)
                 # Remove link from neighbor
                 if neighbor_idx in connections_dict and i in connections_dict[neighbor_idx]:
                      connections_dict[neighbor_idx].remove(i)


    # Create final filtered connections dictionary
    final_connections = {}
    for i, connected_indices in connections_dict.items():
        if i not in nodes_to_delete:
            # Filter out links to deleted nodes
            valid_neighbors = [n_idx for n_idx in connected_indices if n_idx not in nodes_to_delete]
            # Remove duplicates that might arise from reconnection
            final_connections[i] = sorted(list(set(valid_neighbors)))

    return final_connections, list(nodes_to_delete) # Return connections and indices of deleted nodes

@timer_decorator
def run_extract_connection(intersections_yx_skel, skeleton_image, scaler, orig_h, orig_w, cfg):
    """Finds connections between intersections and returns nodes in MapRecognizer format."""
    if not intersections_yx_skel: return []

    params_conn = cfg["extract_connection_parameters"]
    threshold_value = params_conn["threshold_value"]
    skel_h, skel_w = skeleton_image.shape

    # Threshold skeleton image
    _, binary_skel_image = cv2.threshold(skeleton_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find connections between skeleton intersection points
    connections_dict, deleted_node_indices = find_connections(intersections_yx_skel, binary_skel_image, cfg)

    maprec_nodes_list = []
    original_indices = [i for i in range(len(intersections_yx_skel)) if i not in deleted_node_indices]
    kept_intersections_yx_skel = [intersections_yx_skel[i] for i in original_indices]

    # Create a mapping from original index to new sequential index
    orig_to_new_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(original_indices)}

    for new_idx, orig_idx in enumerate(original_indices):
        point_yx_skel = kept_intersections_yx_skel[new_idx] # y, x on skeleton
        point_y_skel, point_x_skel = point_yx_skel

        # Scale coordinates back to original image size
        # Convert (y, x)_skel -> (x, y)_orig
        orig_x = int(point_x_skel * scaler)
        orig_y = int(point_y_skel * scaler)
        # Ensure coords are within original image bounds
        orig_x = max(0, min(orig_w - 1, orig_x))
        orig_y = max(0, min(orig_h - 1, orig_y))


        # Get connected nodes (use original indices from connections_dict, then map to new indices)
        connected_orig_indices = connections_dict.get(orig_idx, [])
        connected_new_indices = [orig_to_new_idx_map[conn_orig_idx] for conn_orig_idx in connected_orig_indices if conn_orig_idx in orig_to_new_idx_map]

        # Store in MapRecognizer format: (new_sequential_id, orig_x, orig_y, list_of_connected_new_ids)
        # Using new_idx as the node ID base
        maprec_nodes_list.append((new_idx, orig_x, orig_y, connected_new_indices))

    return maprec_nodes_list # List of tuples

# --- OCR Class (Adapted from user's OCR class) ---
class BackgroundOCR:
    def __init__(self, image_input, cfg_ocr):
        self.image = image_input.copy()
        self.cfg = cfg_ocr
        self.reader = None # Initialize reader later if needed

    @timer_decorator
    def ocr_func(self):
        """Performs OCR using easyocr."""
        if not OCR_AVAILABLE:
             print("OCR Skipped: easyocr library not available.")
             self.ocr_result = []
             return self.ocr_result

        langs = self.cfg.get('ocr_langs', ['ja', 'en'])
        print(f"Initializing easyocr with languages: {langs}")
        # Initialize reader here to potentially save time if called multiple times (though not in this flow)
        try:
             if self.reader is None:
                  self.reader = easyocr.Reader(langs) # gpu=torch.cuda.is_available()) # Enable GPU if available
             # Convert to grayscale for OCR
             gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
             # Run OCR
             ocr_result_raw = self.reader.readtext(gray)
             # Format: [[bbox, text, confidence], ...]
             # bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
             self.ocr_result = ocr_result_raw
             print(f"easyocr found {len(self.ocr_result)} text boxes.")
             return self.ocr_result
        except Exception as e:
             print(f"Error during easyocr processing: {e}")
             self.ocr_result = []
             return self.ocr_result


    @staticmethod
    def is_overlap(box1, box2):
        """Checks if two bounding boxes overlap. Boxes are ((x1,y1),(x2,y2))."""
        t1, b1 = box1 # TopLeft1, BottomRight1
        t2, b2 = box2 # TopLeft2, BottomRight2
        # Check for non-overlap: one box is to the left/right/top/bottom of the other
        if b1[0] < t2[0] or b2[0] < t1[0] or b1[1] < t2[1] or b2[1] < t1[1]:
            return False
        return True # They overlap

    @staticmethod
    def merge_boxes(box1, box2):
        """Merges two bounding boxes ((x1,y1),(x2,y2))."""
        t1, b1 = box1; t2, b2 = box2
        min_x = min(t1[0], t2[0]); min_y = min(t1[1], t2[1])
        max_x = max(b1[0], b2[0]); max_y = max(b1[1], b2[1])
        return ((min_x, min_y), (max_x, max_y))

    @timer_decorator
    def apply_ocr_with_bounding_boxes(self):
        """Merges nearby OCR results."""
        if not hasattr(self, 'ocr_result') or not self.ocr_result:
             print("No OCR results to process.")
             return [], [] # Return empty lists

        merged_boxes = [] # List of ((min_x, min_y), (max_x, max_y)) tuples
        merged_texts = [] # List of corresponding merged text strings

        # Target dimensions for coordinate normalization (from config)
        target_width = self.cfg.get('width', 640)
        target_height = self.cfg.get('height', 480)
        img_h, img_w = self.image.shape[:2]

        raw_boxes_coords = [] # Store as ((x1,y1),(x2,y2)) normalized
        raw_texts = []

        # Extract and normalize initial boxes
        for detection in self.ocr_result:
            # detection[0] is the list of four points [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            points = np.array(detection[0], dtype=np.int32)
            text = detection[1]
            # Get min/max x/y to define the bounding box corners
            min_x = np.min(points[:, 0]); min_y = np.min(points[:, 1])
            max_x = np.max(points[:, 0]); max_y = np.max(points[:, 1])

            # Normalize coordinates to target W/H
            norm_min_x = int(min_x * (target_width / img_w))
            norm_min_y = int(min_y * (target_height / img_h))
            norm_max_x = int(max_x * (target_width / img_w))
            norm_max_y = int(max_y * (target_height / img_h))

            raw_boxes_coords.append(((norm_min_x, norm_min_y), (norm_max_x, norm_max_y)))
            raw_texts.append(text)

        # Merge overlapping boxes
        processed_indices = set()
        for i in range(len(raw_boxes_coords)):
            if i in processed_indices: continue

            current_box = raw_boxes_coords[i]
            current_text = raw_texts[i]
            processed_indices.add(i)

            # Check subsequent boxes for overlap
            merged_indices = [i] # Keep track of indices merged into this one
            for j in range(i + 1, len(raw_boxes_coords)):
                 if j in processed_indices: continue
                 if self.is_overlap(current_box, raw_boxes_coords[j]):
                      # print(f"Merging box {j} ('{raw_texts[j]}') into box {i} ('{current_text}')")
                      current_box = self.merge_boxes(current_box, raw_boxes_coords[j])
                      current_text += ' ' + raw_texts[j]
                      processed_indices.add(j)
                      merged_indices.append(j)

            merged_boxes.append(current_box)
            merged_texts.append(current_text)

        print(f"Merged OCR results into {len(merged_boxes)} boxes.")
        # Returns: list of boxes [((x1,y1),(x2,y2)), ...], list of texts [text1, ...]
        # Coordinates are normalized to target W/H specified in ocr_parameters
        return merged_boxes, merged_texts

def poi_merge(pois_input, labeled_image, cfg):
    """Merges POIs based on proximity within the same labeled region."""
    params = cfg["poi_merge_parameters"]
    poi_search_radius = params["poi_threshold"] # How far around POI center to look for label
    # Note: MapRecognizer uses a square search, maybe radius is better?
    h, w = labeled_image.shape[:2]

    pois_with_labels = []
    # 1. Assign a label ID to each POI
    for poi_id, poi_x, poi_y, poi_text in pois_input:
        found_label = 0 # Default to 0 (no label/background)
        # Search around the POI coordinates in the labeled image
        y_center, x_center = int(poi_y), int(poi_x)
        # Check center first
        if 0 <= y_center < h and 0 <= x_center < w:
            center_label = labeled_image[y_center, x_center]
            if center_label > 0: found_label = center_label

        # If center is background, search in a radius (more robust than square)
        if found_label == 0:
             min_r = max(0, y_center - poi_search_radius); max_r = min(h, y_center + poi_search_radius + 1)
             min_c = max(0, x_center - poi_search_radius); max_c = min(w, x_center + poi_search_radius + 1)
             region = labeled_image[min_r:max_r, min_c:max_c]
             unique_labels, counts = np.unique(region[region > 0], return_counts=True) # Find non-zero labels
             if len(unique_labels) > 0:
                 found_label = unique_labels[np.argmax(counts)] # Assign the most frequent label in the area

        pois_with_labels.append([poi_id, poi_x, poi_y, poi_text, found_label])

    # 2. Merge POIs with the same non-zero label ID
    merged_pois_dict = {} # key: label_id, value: list of POIs in that label
    unlabeled_pois = []

    for p_id, p_x, p_y, p_text, p_label in pois_with_labels:
        if p_label != 0:
            if p_label not in merged_pois_dict:
                merged_pois_dict[p_label] = []
            merged_pois_dict[p_label].append([p_id, p_x, p_y, p_text])
        else:
            # Keep unlabeled POIs separate for now
            unlabeled_pois.append([p_id, p_x, p_y, p_text, p_label]) # Keep label 0

    final_merged_pois = []
    new_poi_id_counter = pois_input[-1][0] + 1 if pois_input else 0

    # Process labeled POIs
    for label_id, poi_list in merged_pois_dict.items():
        if not poi_list: continue
        if len(poi_list) == 1:
            # Only one POI in this label, add it directly
            p_id, p_x, p_y, p_text = poi_list[0]
            final_merged_pois.append([p_id, p_x, p_y, p_text, label_id])
        else:
            # Merge multiple POIs in the same label
            combined_text = ' '.join([p[3] for p in poi_list])
            # Calculate average position
            avg_x = sum(p[1] for p in poi_list) / len(poi_list)
            avg_y = sum(p[2] for p in poi_list) / len(poi_list)
            # Use a new ID or the first POI's ID? MapRecognizer seems to use text as ID later.
            # For consistency, let's use the first poi's original ID.
            merged_id = poi_list[0][0]
            final_merged_pois.append([merged_id, int(avg_x), int(avg_y), combined_text, label_id])
            # print(f"DEBUG: Merged {len(poi_list)} POIs in label {label_id} into '{combined_text}'")

    # Add back the unlabeled POIs
    final_merged_pois.extend(unlabeled_pois)

    # Return list of [id, x, y, text, label_num]
    return final_merged_pois

@timer_decorator
def run_extract_pois(ocr_boxes, ocr_texts, labeled_image, orig_h, orig_w, cfg):
    """Processes OCR results, resizes coords, and merges based on labels."""
    params_ext = cfg["extract_pois_parameters"]
    params_ocr = cfg["ocr_parameters"]
    # Use target W/H from OCR params for consistency during resizing
    ref_width = params_ocr["width"]
    ref_height = params_ocr["height"]

    initial_pois = []
    # Convert merged OCR boxes (normalized to ref W/H) to POIs (center point)
    for i, box in enumerate(ocr_boxes):
        top_left, bottom_right = box # ((x1,y1), (x2,y2)) normalized
        center_x_norm = (top_left[0] + bottom_right[0]) / 2
        center_y_norm = (top_left[1] + bottom_right[1]) / 2
        text = ocr_texts[i]
        poi_id = i # Use simple index as initial ID

        # Scale normalized center coordinates to original image size
        orig_x = int(center_x_norm * (orig_w / ref_width))
        orig_y = int(center_y_norm * (orig_h / ref_height))
        orig_x = max(0, min(orig_w - 1, orig_x))
        orig_y = max(0, min(orig_h - 1, orig_y))

        initial_pois.append((poi_id, orig_x, orig_y, text))

    # Merge POIs based on labeled regions
    if labeled_image is not None and labeled_image.shape[0] > 0:
        final_pois = poi_merge(initial_pois, labeled_image, cfg)
    else:
        print("Warning: Labeled image is invalid, skipping POI merging based on labels.")
        # Return initial POIs with label 0 if no labeled image
        final_pois = [p + [0] for p in initial_pois] # Add label 0

    # Return list of [id, x_orig, y_orig, text, label_num]
    return final_pois


# === Interactive Tool Functions (Mostly unchanged) ===

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (tuples/lists)."""
    try:
        if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return float('inf')
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    except (TypeError, IndexError) as e: print(f"Error calculating distance between {p1} and {p2}: {e}"); return float('inf')
    except Exception as e: print(f"Unexpected error calculating distance between {p1} and {p2}: {e}"); return float('inf')

def get_display_coords(orig_x, orig_y):
    """Converts original image coordinates to display image coordinates."""
    if not isinstance(orig_x, (int, float)) or not isinstance(orig_y, (int, float)): print(f"Warning: Invalid input type for get_display_coords: x={orig_x}, y={orig_y}"); return 0, 0
    return int(orig_x * resize_factor), int(orig_y * resize_factor)

def get_original_coords(display_x, display_y):
    """Converts display image coordinates back to original image coordinates."""
    if resize_factor == 0: return 0, 0
    if not isinstance(display_x, (int, float)) or not isinstance(display_y, (int, float)): print(f"Warning: Invalid input type for get_original_coords: x={display_x}, y={display_y}"); return 0, 0
    return int(display_x / resize_factor), int(display_y / resize_factor)

def get_node_by_id(node_id):
    """Retrieves an interactive tool node dictionary by its ID."""
    if node_id is None: return None
    for node in nodes: # Search the interactive tool's node list
        if node.get('id') == node_id: return node
    return None

def find_nearest_node(display_x, display_y):
    """Finds the nearest existing interactive node within max_dist on the display."""
    nearest_node = None
    min_dist_display = float('inf')
    # Use config value for snapping distance
    max_dist_from_config = config.get('display_parameters', {}).get('max_click_distance_node', 15)

    for node in nodes: # Check the interactive tool's node list
        if 'pixel_x' not in node or 'pixel_y' not in node: continue
        try:
            node_disp_x, node_disp_y = get_display_coords(node['pixel_x'], node['pixel_y'])
            dist = calculate_distance((display_x, display_y), (node_disp_x, node_disp_y))
            if dist < min_dist_display and dist <= max_dist_from_config:
                min_dist_display = dist
                nearest_node = node
        except Exception as e: print(f"Error processing node {node.get('id', 'Unknown')} in find_nearest_node: {e}"); continue
    return nearest_node

def get_real_distance(node1, node2):
    """Calculates real-world distance between two interactive nodes using the scale."""
    if not node1 or not node2: return float('inf')
    if not all(k in node1 for k in ('pixel_x', 'pixel_y')) or \
       not all(k in node2 for k in ('pixel_x', 'pixel_y')):
        print(f"Warning: Missing pixel coordinates for distance calc between node IDs {node1.get('id')} and {node2.get('id')}")
        return float('inf')
    try:
        pixel_dist_orig = calculate_distance((node1['pixel_x'], node1['pixel_y']), (node2['pixel_x'], node2['pixel_y']))
        if scale_pixels_per_meter and scale_pixels_per_meter > 1e-6: return pixel_dist_orig / scale_pixels_per_meter
        else:
            # Return pixel distance if scale not set or invalid
            if scale_pixels_per_meter is None: pass
            elif scale_pixels_per_meter <= 1e-6: print(f"Warning: Invalid scale ({scale_pixels_per_meter}). Returning pixel distance.")
            return pixel_dist_orig
    except Exception as e: print(f"Error calculating real distance between {node1.get('id')} and {node2.get('id')}: {e}"); return float('inf')

def find_nearest_suggestion(display_x, display_y):
    """Finds the nearest intersection or POI suggestion within snap distance."""
    if not ANALYSIS_AVAILABLE or not analysis_complete: return None, None # Check analysis ran

    nearest_sug = None # Store original suggestion data [x,y] or [x,y,text]
    min_dist_display = float('inf')
    sug_type = None # 'intersection' or 'poi'

    disp_params = config.get('display_parameters', DEFAULT_CONFIG.get('display_parameters', {}))
    snap_dist = disp_params.get('max_suggestion_snap_distance', 20)

    # Check suggested intersections (list of [x_orig, y_orig])
    for pt_orig in suggested_intersections_orig:
        if isinstance(pt_orig, (list, tuple)) and len(pt_orig) >= 2 and \
           isinstance(pt_orig[0], (int, float)) and isinstance(pt_orig[1], (int, float)):
            try:
                pt_disp = get_display_coords(pt_orig[0], pt_orig[1])
                dist = calculate_distance((display_x, display_y), pt_disp)
                if dist < min_dist_display and dist <= snap_dist:
                    min_dist_display = dist
                    nearest_sug = pt_orig # Store original coords
                    sug_type = 'intersection'
            except Exception as e: print(f"Error processing intersection suggestion {pt_orig}: {e}"); continue

    # Check suggested POIs (list of [x_orig, y_orig, text])
    for poi_orig in suggested_pois_orig:
        if isinstance(poi_orig, (list, tuple)) and len(poi_orig) >= 3 and \
           isinstance(poi_orig[0], (int, float)) and isinstance(poi_orig[1], (int, float)):
            try:
                pt_disp = get_display_coords(poi_orig[0], poi_orig[1])
                dist = calculate_distance((display_x, display_y), pt_disp)
                if dist < min_dist_display and dist <= snap_dist:
                    min_dist_display = dist
                    nearest_sug = poi_orig # Store original coords + text
                    sug_type = 'poi'
            except Exception as e: print(f"Error processing POI suggestion {poi_orig}: {e}"); continue

    return sug_type, nearest_sug


# --- Drawing Function ---
def redraw_display():
    """Redraws the display window with current elements and status."""
    global display_image
    if original_image is None:
        # Create a black canvas indicating the error if image isn't loaded
        display_image = np.zeros((max(100,display_h), max(200,display_w), 3), dtype=np.uint8)
        cv2.putText(display_image, "Error: Original image not loaded", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        try:
            cv2.imshow("Floor Plan Map Generator", display_image)
        except Exception as e:
            print(f"Error showing error display: {e}")
        return

    # Create a fresh resized copy to draw on
    try:
        # Ensure display_w and display_h are valid before resizing
        if display_w <= 0 or display_h <= 0:
            raise ValueError(f"Invalid display dimensions ({display_w}x{display_h})")
        if not isinstance(original_image, np.ndarray):
             raise TypeError("Original image is not a valid NumPy array")

        display_image = cv2.resize(original_image, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"ERROR: Could not resize image for display: {e}")
        # Fallback: Create a black canvas with error message
        display_image = np.zeros((max(100, display_h), max(200, display_w), 3), dtype=np.uint8) # Ensure minimum size
        cv2.putText(display_image, f"Resize Error: {e}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)


    # Get display parameters safely using .get() with defaults
    disp_params = config.get('display_parameters', DEFAULT_CONFIG['display_parameters'])
    # Ensure colors are tuples of integers
    node_color_default = tuple(map(int, disp_params.get('node_color', [0, 255, 0])))
    node_radius = int(disp_params.get('node_radius', 5))
    link_color = tuple(map(int, disp_params.get('link_color', [255, 0, 0])))
    highlight_color = tuple(map(int, disp_params.get('highlight_color', [0, 255, 255])))
    text_color = tuple(map(int, disp_params.get('text_color', [255, 255, 255])))
    sug_intersect_color = tuple(map(int, disp_params.get('suggestion_intersection_color', [0, 0, 255])))
    sug_poi_color = tuple(map(int, disp_params.get('suggestion_poi_color', [255, 0, 255])))
    sug_radius = int(disp_params.get('suggestion_radius', 4))

    # --- SKELETON DRAWING OVERLAY ---
    if ANALYSIS_AVAILABLE and analysis_complete and show_skeleton and global_skeleton_img is not None:
        try:
            # global_skeleton_img is the downscaled skeleton (e.g., H/9, W/9)
            # We need to resize it to the *display* window's size (display_h, display_w)
            if global_skeleton_img.shape[0] > 0 and global_skeleton_img.shape[1] > 0: # Check if skeleton not empty
                skeleton_display_size = cv2.resize(global_skeleton_img, (display_w, display_h), interpolation=cv2.INTER_NEAREST)

                # Create a mask where the skeleton is present in the display size
                skeleton_mask_display = skeleton_display_size > 128 # Threshold to make boolean mask

                # Choose a color for the skeleton overlay (BGR format - Cyan)
                skeleton_overlay_color = np.array([255, 255, 0], dtype=np.uint8) # Cyan BGR

                # Apply the color directly onto the display image where the mask is True
                display_image[skeleton_mask_display] = skeleton_overlay_color
            else:
                print("DEBUG: global_skeleton_img is empty, cannot draw overlay.")

        except Exception as e:
            print(f"Error drawing skeleton overlay: {e}")
    # --- END SKELETON DRAWING ---


    # Draw Suggestions (Intersection/POI)
    if ANALYSIS_AVAILABLE and show_suggestions and analysis_complete:
        try:
            # Draw intersection suggestions (Red circles)
            for pt_orig in suggested_intersections_orig: # pt_orig = [x, y]
                if isinstance(pt_orig, (list, tuple)) and len(pt_orig) >= 2:
                    pt_disp = get_display_coords(pt_orig[0], pt_orig[1])
                    cv2.circle(display_image, pt_disp, sug_radius, sug_intersect_color, -1) # Filled circle

            # Draw POI suggestions (Magenta crosses)
            for poi_orig in suggested_pois_orig: # poi_orig = [x, y, text]
                if isinstance(poi_orig, (list, tuple)) and len(poi_orig) >= 3:
                    pt_disp = get_display_coords(poi_orig[0], poi_orig[1])
                    marker_size = max(5, sug_radius * 2 + 1) # Ensure marker size is reasonable
                    cv2.drawMarker(display_image, pt_disp, sug_poi_color, cv2.MARKER_CROSS, marker_size, 1)
        except Exception as e:
            print(f"Error drawing suggestions: {e}") # Catch potential errors during drawing

    # Draw Links (from interactive tool's link list)
    try:
        for link in links:
            node1 = get_node_by_id(link.get('startNode'))
            node2 = get_node_by_id(link.get('endNode'))
            # Check if nodes and required coords exist
            if node1 and node2 and all(k in node1 for k in ('pixel_x', 'pixel_y')) and all(k in node2 for k in ('pixel_x', 'pixel_y')):
                pt1_disp = get_display_coords(node1['pixel_x'], node1['pixel_y'])
                pt2_disp = get_display_coords(node2['pixel_x'], node2['pixel_y'])
                cv2.line(display_image, pt1_disp, pt2_disp, link_color, 1) # Line thickness 1
    except Exception as e:
        print(f"Error drawing links: {e}")

    # Draw Nodes (from interactive tool's node list)
    try:
        for node in nodes:
            if 'pixel_x' not in node or 'pixel_y' not in node: continue
            center_disp = get_display_coords(node['pixel_x'], node['pixel_y'])
            color = node_color_default
            radius = node_radius
            # Highlight start node if in LINK_END mode
            if mode == 'LINK_END' and link_start_node and node.get('id') == link_start_node.get('id'):
                color = highlight_color
                radius = node_radius + 2
            cv2.circle(display_image, center_disp, radius, color, -1) # Filled circle
            # Draw node ID slightly offset
            text_pos_disp = (center_disp[0] + radius + 2, center_disp[1] + radius + 2)
            node_id_str = str(node.get('id', '?')) # Ensure ID is string
            cv2.putText(display_image, node_id_str, text_pos_disp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    except Exception as e:
        print(f"Error drawing nodes: {e}")


    # Draw scale points if being defined
    try:
        if mode in ['SCALE_START', 'SCALE_END'] and scale_points_orig:
            for pt_orig in scale_points_orig:
                 if isinstance(pt_orig, (list, tuple)) and len(pt_orig) >= 2:
                    pt_disp = get_display_coords(pt_orig[0], pt_orig[1])
                    cv2.drawMarker(display_image, pt_disp, highlight_color, cv2.MARKER_CROSS, 15, 2)
    except Exception as e:
        print(f"Error drawing scale points: {e}")


    # --- Display Status Text ---
    status_line1 = f"Mode: {mode}"
    if scale_pixels_per_meter:
        status_line1 += f" | Scale: {scale_pixels_per_meter:.2f} px/m"
    else:
        status_line1 += " | Scale: Not Set"

    # Status line 2: Analysis Status
    status_line2 = ""
    if ANALYSIS_AVAILABLE: # Check base analysis libs
        if analysis_in_progress:
             status_line2 = "Analyzing (MapRecognizer)..."
        elif analysis_complete:
            status_line2 = f"Suggestions: {'ON' if show_suggestions else 'OFF'} ('V') "
            # --- ADD SKELETON STATUS ---
            status_line2 += f"| Skeleton: {'ON' if show_skeleton else 'OFF'} ('K')"
            # --- END ADD ---
        else:
             status_line2 = "Analysis Ready ('A' to start/re-run)"
    else:
        status_line2 = "Analysis Disabled (missing libraries)"

    # Draw status text lines
    try:
        # Add background rectangle for better visibility
        text_bg_color = (0,0,0) # Black background
        y_offset = 5 # Starting y padding
        line_height = 20 # Estimated height of a line
        max_text_w = 0

        # Draw Line 1
        (w1, h1), _ = cv2.getTextSize(status_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        max_text_w = max(max_text_w, w1)
        text_y1 = y_offset + line_height
        cv2.putText(display_image, status_line1, (10, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
        last_y = text_y1 + y_offset

        # Draw Line 2 if it exists
        if status_line2:
            (w2, h2), _ = cv2.getTextSize(status_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            max_text_w = max(max_text_w, w2)
            text_y2 = last_y + line_height
            cv2.putText(display_image, status_line2, (10, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
            last_y = text_y2 + y_offset

        # Draw background rectangle covering both lines
        cv2.rectangle(display_image, (5, 5), (15 + max_text_w, last_y), text_bg_color, -1)

        # Redraw text on top of background
        cv2.putText(display_image, status_line1, (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
        if status_line2:
            cv2.putText(display_image, status_line2, (10, y_offset + line_height*2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)


    except Exception as e:
        print(f"Error drawing status text: {e}")

    # Show the updated image in the window
    try:
        cv2.imshow("Floor Plan Map Generator", display_image)
    except Exception as e:
        # This can happen if the window was closed unexpectedly
        print(f"Error showing display image (window might be closed): {e}")


# --- Mouse Callback ---
def mouse_callback(event, x_disp, y_disp, flags, param):
    """Handles mouse clicks on the display window."""
    global mode, node_counter, link_start_node, scale_points_orig, scale_pixels_per_meter, nodes, links # Ensure nodes/links modifiable

    if not isinstance(x_disp, int) or not isinstance(y_disp, int): return
    if not (0 <= x_disp < display_w and 0 <= y_disp < display_h): return

    x_orig, y_orig = get_original_coords(x_disp, y_disp)

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"\nClicked display ({x_disp}, {y_disp}) -> original ({x_orig}, {y_orig}) | Mode: {mode}")

        # --- NODE MODE ---
        if mode == 'NODE':
            # Use interactive tool's node counter
            node_id = f"node{node_counter}"
            node_name_suggestion, node_type_suggestion = "", ""
            final_x_orig, final_y_orig = x_orig, y_orig

            # --- Try snapping to MapRecognizer suggestions ---
            sug_type, nearest_sug_data = None, None
            if ANALYSIS_AVAILABLE and show_suggestions and analysis_complete:
                try: sug_type, nearest_sug_data = find_nearest_suggestion(x_disp, y_disp)
                except Exception as e: print(f"Error finding nearest suggestion: {e}")

            if nearest_sug_data:
                final_x_orig = nearest_sug_data[0] # x_orig
                final_y_orig = nearest_sug_data[1] # y_orig
                if sug_type == 'intersection':
                    node_type_suggestion = "intersection" # Match MapRecognizer class
                    print(f"Snapped to suggested intersection @ ({final_x_orig}, {final_y_orig})")
                elif sug_type == 'poi':
                    node_type_suggestion = "poi" # Match MapRecognizer class
                    node_name_suggestion = nearest_sug_data[2] if len(nearest_sug_data) >= 3 else "POI"
                    print(f"Snapped to suggested POI '{node_name_suggestion}' @ ({final_x_orig}, {final_y_orig})")
                else: print(f"Snapped near suggestion @ ({final_x_orig}, {final_y_orig})")
            else: print(f"Adding new node near ({x_orig}, {y_orig}). Using click coordinates.")
            # --- End Suggestion Snapping ---

            # Get node details from user input (for the interactive tool's node list)
            default_name = node_name_suggestion or f'Point {node_counter}'
            default_type = node_type_suggestion or 'PointOfInterest' # Use suggestion type or generic
            name_prompt = f" > Enter node name [{default_name}]: "
            type_prompt = f" > Enter node type [{default_type}]: "

            node_name = default_name
            node_type = default_type
            try:
                node_name_input = input(name_prompt).strip()
                node_type_input = input(type_prompt).strip()
                node_name = node_name_input or default_name
                node_type = node_type_input or default_type
            except EOFError: print("\nInput interrupted. Cancelling node addition."); return
            except Exception as e: print(f"\nError during input: {e}. Using default values.")

            # Calculate real coords for interactive node (if scale is set)
            real_x, real_y = None, None
            try:
                if scale_pixels_per_meter and scale_pixels_per_meter > 1e-6:
                    real_x = final_x_orig / scale_pixels_per_meter
                    real_y = final_y_orig / scale_pixels_per_meter
                else: real_x = float(final_x_orig); real_y = float(final_y_orig) # Fallback to pixels
            except Exception as e: print(f"Error calculating real coordinates for node {node_id}: {e}"); real_x = float(final_x_orig); real_y = float(final_y_orig)

            # Add node to the *interactive tool's* list for display/linking
            interactive_node_data = {
                "id": node_id,
                "name": node_name,
                "type": node_type, # This is for the interactive display
                "pixel_x": final_x_orig, # Store original pixel coords
                "pixel_y": final_y_orig,
                "x": real_x, # Real-world coords (or pixel if no scale)
                "y": real_y
            }
            nodes.append(interactive_node_data)
            node_counter += 1 # Increment interactive counter
            print(f"Interactive Node '{node_id}' ({node_name}, Type: {node_type}) added at pixel ({final_x_orig}, {final_y_orig}).")

        # --- LINK MODE START ---
        elif mode == 'LINK_START':
            clicked_node = find_nearest_node(x_disp, y_disp) # Uses interactive node list
            if clicked_node:
                link_start_node = clicked_node
                start_node_id = link_start_node.get('id','?'); start_node_name = link_start_node.get('name','?')
                print(f"Link start: Node '{start_node_id}' ({start_node_name})."); mode = 'LINK_END'; print(f"Click the destination node.")
            else:
                click_dist = config.get('display_parameters', {}).get('max_click_distance_node', 15)
                print(f"No node found within {click_dist} pixels of click. Click closer to an existing node.")

        # --- LINK MODE END ---
        elif mode == 'LINK_END':
            if not link_start_node: print("Error: Link start node was lost. Returning to LINK_START mode."); mode = 'LINK_START'; return
            clicked_node = find_nearest_node(x_disp, y_disp) # Uses interactive node list
            if clicked_node:
                start_id = link_start_node.get('id'); end_id = clicked_node.get('id')
                if not start_id or not end_id: print("Error: Clicked nodes have missing IDs. Cannot create link."); link_start_node = None; mode = 'LINK_START'; return
                if end_id == start_id: print("Cannot link a node to itself. Click a different node.")
                else:
                    # Check if link already exists in the interactive list
                    link_exists = False
                    for l in links:
                         if 'startNode' in l and 'endNode' in l:
                              if (l['startNode'] == start_id and l['endNode'] == end_id) or \
                                 (l['startNode'] == end_id and l['endNode'] == start_id): link_exists = True; break
                    if link_exists: print(f"Link between '{start_id}' and '{end_id}' already exists.")
                    else:
                        # Add link to the *interactive tool's* list
                        links.append({"startNode": start_id, "endNode": end_id})
                        print(f"Link created: '{start_id}' <-> '{end_id}'")
                    link_start_node = None; mode = 'LINK_START'; print(f"\nMode: {mode}. Click start node for next link, or change mode ('N', 'S').")
            else:
                click_dist = config.get('display_parameters', {}).get('max_click_distance_node', 15)
                print(f"No node found within {click_dist} pixels of click for link end. Click closer.")

        # --- SCALE MODE START ---
        elif mode == 'SCALE_START':
             scale_points_orig = [(x_orig, y_orig)]; print(f"Scale: First point set at original image coordinates ({x_orig},{y_orig})."); mode = 'SCALE_END'; print(f"Click the second point for the known distance.")

        # --- SCALE MODE END ---
        elif mode == 'SCALE_END':
             if not scale_points_orig: print("Error: First scale point not set. Returning to SCALE_START."); mode = 'SCALE_START'; return
             scale_points_orig.append((x_orig, y_orig)); print(f"Scale: Second point set at original image coordinates ({x_orig},{y_orig}).")
             pixel_distance_orig = float('inf')
             try: pixel_distance_orig = calculate_distance(scale_points_orig[0], scale_points_orig[1]); print(f"Pixel distance (on original image): {pixel_distance_orig:.2f}")
             except Exception as e: print(f"Error calculating pixel distance for scale: {e}"); scale_points_orig = []; mode = 'SCALE_START'; return
             if pixel_distance_orig < 1e-6: print("Points are too close together or calculation failed. Please select two distinct points."); scale_points_orig = []; mode = 'SCALE_START'; return

             real_distance_meters = None
             while real_distance_meters is None:
                 try:
                     real_distance_str = input(" > Enter the real-world distance between these points (in METERS): ").strip()
                     if real_distance_str.lower() in ['q', 'quit', 'cancel', 'exit']: print("Scale setting cancelled by user."); scale_points_orig = []; mode = 'SCALE_START'; return
                     real_distance_meters = float(real_distance_str)
                     if real_distance_meters <= 1e-6: print("Distance must be a positive number greater than zero."); real_distance_meters = None
                 except ValueError: print("Invalid input. Please enter a number (e.g., 5.5) or 'q' to cancel.")
                 except EOFError: print("\nInput interrupted. Cancelling scale setting."); scale_points_orig = []; mode = 'SCALE_START'; return
                 except Exception as e: print(f"\nAn unexpected error occurred during input: {e}"); scale_points_orig = []; mode = 'SCALE_START'; return

             if real_distance_meters <= 1e-6: print("Error: Real distance is too small, cannot calculate scale."); scale_points_orig = []; mode = 'SCALE_START'; return
             scale_pixels_per_meter = pixel_distance_orig / real_distance_meters
             print(f"--- Scale Set: {scale_pixels_per_meter:.3f} original pixels per meter ---")

             # Update existing *interactive* nodes
             update_coords = False
             if nodes:
                 try: confirm_update = input(" > Update real (x, y) coordinates for existing interactive nodes? (y/N): ").lower().strip(); update_coords = (confirm_update == 'y')
                 except EOFError: print("\nInput interrupted. Not updating.")
                 except Exception as e: print(f"\nError during confirmation input: {e}. Not updating.")

             if update_coords:
                 print("   Updating real coordinates for existing interactive nodes..."); updated_count, skipped_count = 0, 0
                 if scale_pixels_per_meter and scale_pixels_per_meter > 1e-6:
                     for node in nodes: # Iterate interactive nodes
                         try:
                             if 'pixel_x' in node and isinstance(node['pixel_x'], (int, float)) and \
                                'pixel_y' in node and isinstance(node['pixel_y'], (int, float)):
                                node['x'] = node['pixel_x'] / scale_pixels_per_meter; node['y'] = node['pixel_y'] / scale_pixels_per_meter; updated_count += 1
                             else: print(f"   Skipping node {node.get('id', '?')} (missing pixel coords)."); skipped_count += 1
                         except Exception as e: print(f"   Error updating node {node.get('id','?')}: {e}"); skipped_count += 1
                     print(f"   Updated {updated_count} interactive nodes. Skipped {skipped_count}.")
                 else: print("   Skipping update due to invalid scale.")

             scale_points_orig = []; mode = 'NODE'; print(f"\nMode: {mode}. Continue adding nodes or change mode.")


# --- Saving Function (MODIFIED to use MapRecognizer results and format) ---
def save_map(filepath="map_generated.json", image_name_without_extension="map"):
    """Saves the map data based on MapRecognizer results."""
    global maprec_nodes, maprec_pois # Use results from background analysis

    print(f"\nSaving MapRecognizer results to JSON files...")

    if not analysis_complete:
        print("Warning: Analysis not complete. Saving may use empty or incomplete data.")
        # Optionally save interactive nodes/links here as a fallback?
        # return

    if not maprec_nodes and not maprec_pois:
        print("Warning: No nodes or POIs found by analysis. Saving empty files.")

    # Define output paths specific to this image
    image_json_path = os.path.join(JSON_SAVE_PATH, image_name_without_extension)
    os.makedirs(image_json_path, exist_ok=True)

    # --- Save Nodes (map.json / output.json in MapRecognizer) ---
    output_maprec_nodes = []
    for node_data in maprec_nodes:
        # node_data = (id, x, y, connected_ids)
        node_id, node_x, node_y, connected_indices = node_data
        node_dict = {
            "id": f"node{node_id}", # Use the sequential ID generated
            "x": int(node_x),
            "y": int(node_y),
            "nodeClass": "intersection", # Assume all are intersections from extract_connection
            "outgoingLinks": [{"endNode": f"node{link_id}"} for link_id in connected_indices]
        }
        output_maprec_nodes.append(node_dict)

    nodes_data_json = {"nodes": output_maprec_nodes}
    map_json_path = os.path.join(image_json_path, "map.json")
    # MapRecognizer also saves this as output.json in its structure? Replicate that.
    output_json_nodes_path = os.path.join(image_json_path, "output.json")

    try:
        with open(map_json_path, "w", encoding='utf-8') as outfile:
            json.dump(nodes_data_json, outfile, indent=4, ensure_ascii=False)
        print(f"Saved MapRecognizer intersection nodes to '{map_json_path}'")
        # Replicate saving to output.json
        with open(output_json_nodes_path, "w", encoding='utf-8') as outfile:
             json.dump(nodes_data_json, outfile, indent=4, ensure_ascii=False)
        print(f"Saved MapRecognizer intersection nodes to '{output_json_nodes_path}'")

    except Exception as e:
        print(f"ERROR saving MapRecognizer nodes JSON: {e}")

    # --- Save POIs (output.json in MapRecognizer's project root) ---
    poi_list_json = []
    for poi_data in maprec_pois:
        # poi_data = [id, x, y, text, label_num]
        poi_id_num, poi_x, poi_y, poi_text, _ = poi_data # Ignore label num for saving
        poi_dict = {
            # MapRecognizer uses the text as ID here, ensure uniqueness?
            # Using a combination or just text might be fragile. Let's use text for now.
            "id": poi_text,
            "x": int(poi_x),
            "y": int(poi_y),
            "nodeClass": "poi",
        }
        poi_list_json.append(poi_dict)

    pois_data_json = {"nodes": poi_list_json}
    # MapRecognizer saves this as output.json in the *root* output folder
    output_json_poi_path = os.path.join(SAVE_PATH, "output.json") # Path relative to script execution?

    try:
        with open(output_json_poi_path, "w", encoding="utf-8") as outfile:
            json.dump(pois_data_json, outfile, indent=4, ensure_ascii=False)
        print(f"Saved MapRecognizer POIs to '{output_json_poi_path}'")
    except Exception as e:
        print(f"ERROR saving MapRecognizer POIs JSON: {e}")


    # --- Save Initial Node (initial_node.json) ---
    initial_node_found = None
    for item in maprec_pois:
        # item = [id, x, y, text, label_num]
        poi_text = item[3]
        # Check for Japanese terms for "current location"
        if '現在地' in poi_text or '現在位置' in poi_text:
            initial_node_found = item
            break

    initial_node_json_path = os.path.join(image_json_path, 'initial_node.json')
    initial_output = {"nodes": []}

    if initial_node_found:
        _, initial_x, initial_y, _, _ = initial_node_found
        initial_output["nodes"].append({
            "id": "initial_node",
            "x": int(initial_x),
            "y": int(initial_y),
            "nodeClass": "initial",
            "directionX": 0.0, # Default direction
            "directionY": 0.0
        })
        print(f"Found initial location POI, saving to '{initial_node_json_path}'")
    else:
        initial_output["nodes"].append({
            "id": "initial_node",
            "x": 0, # Default location if not found
            "y": 0,
            "nodeClass": "initial",
            "directionX": 0.0,
            "directionY": 0.0
        })
        print(f"Initial location POI not found, saving default to '{initial_node_json_path}'")

    try:
        with open(initial_node_json_path, 'w', encoding='utf-8') as file:
            json.dump(initial_output, file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"ERROR saving initial node JSON: {e}")


# --- Background Analysis Function (Uses MapRecognizer Logic) ---
# --- Background Analysis Function (Uses MapRecognizer Logic) ---
def run_analysis_in_background(image_path_arg):
    """Runs the full MapRecognizer pipeline in the background."""
    global analysis_in_progress, analysis_complete, maprec_nodes, maprec_pois
    global suggested_intersections_orig, suggested_pois_orig, maprec_config
    # --- ADDED GLOBALS FOR SKELETON ---
    global global_skeleton_img, global_scaler_used
    # --- END ADD ---

    if analysis_in_progress:
        print("Analysis is already in progress.")
        return
    if original_image is None:
        print("Error: Cannot run analysis, original image not loaded.")
        return

    analysis_in_progress = True
    analysis_complete = False
    maprec_nodes = [] # Reset results
    maprec_pois = []
    suggested_intersections_orig = []
    suggested_pois_orig = []
    # --- RESET SKELETON GLOBALS ---
    global_skeleton_img = None
    global_scaler_used = 1
    # --- END RESET ---
    print("\n--- Starting Background Analysis (MapRecognizer Pipeline) ---")
    start_time = time.time()

    try:
        # Load MapRecognizer config (using hardcoded path from its __init__)
        # TODO: Make this configurable via script arguments
        maprec_config_path = "C:/Users/rohit/perceptaai/MapAnalysis/config.yml" # KEEPING YOUR HARDCODED PATH
        if YAML_AVAILABLE and os.path.exists(maprec_config_path):
            with open(maprec_config_path, "r") as ymlfile:
                maprec_config = yaml.safe_load(ymlfile)
                if not isinstance(maprec_config, dict):
                     print(f"Warning: MapRecognizer config file '{maprec_config_path}' is not a dictionary. Using defaults.")
                     maprec_config = DEFAULT_CONFIG # Fallback
        else:
             print(f"Warning: MapRecognizer config file '{maprec_config_path}' not found or PyYAML missing. Using internal defaults.")
             maprec_config = DEFAULT_CONFIG # Fallback to internal defaults

        # --- MapRecognizer Pipeline Steps (executed sequentially) ---
        print("   [1/7] Running getpath...")
        path_mask = run_getpath(original_image, maprec_config)
        if path_mask is None: raise ValueError("Getpath failed.")

        print("   [2/7] Running extract_corridor...")
        labeled_image = run_extract_corridor(original_image, maprec_config)
        # labeled_image can be None if fails, handle later

        print("   [3/7] Running skeletonize...")
        skeleton_img, scaler_used = run_skeletonize(path_mask, maprec_config)
        if skeleton_img is None: raise ValueError("Skeletonization failed.")
        # --- STORE SKELETON GLOBALLY ---
        global_skeleton_img = skeleton_img
        global_scaler_used = scaler_used
        # --- END STORE ---

        print("   [4/7] Running extract_corner...")
        intersections_yx_skel = run_extract_corner(skeleton_img, maprec_config)
        # intersections_yx_skel is list of [y, x] on skeleton image

        print("   [5/7] Running extract_connection...")
        maprec_nodes = run_extract_connection(intersections_yx_skel, skeleton_img, scaler_used, original_h, original_w, maprec_config)
        # maprec_nodes = [(id, x_orig, y_orig, [connected_ids]), ...]

        print("   [6/7] Running OCR...")
        ocr_handler = BackgroundOCR(original_image, maprec_config['ocr_parameters'])
        ocr_raw_results = ocr_handler.ocr_func() # Run easyocr
        if ocr_raw_results:
             merged_ocr_boxes, merged_ocr_texts = ocr_handler.apply_ocr_with_bounding_boxes()
        else:
             merged_ocr_boxes, merged_ocr_texts = [], []

        print("   [7/7] Running extract_pois...")
        maprec_pois = run_extract_pois(merged_ocr_boxes, merged_ocr_texts, labeled_image, original_h, original_w, maprec_config)
        # maprec_pois = [[id, x_orig, y_orig, text, label_num], ...]

        # --- Populate suggestion lists for interactive tool ---
        suggested_intersections_orig = [[int(node[1]), int(node[2])] for node in maprec_nodes]
        suggested_pois_orig = [[int(poi[1]), int(poi[2]), poi[3]] for poi in maprec_pois]

        analysis_complete = True
        end_time = time.time()
        print(f"--- Background analysis finished in {end_time - start_time:.2f} seconds ---")
        print(f"   Found {len(maprec_nodes)} intersection nodes and {len(maprec_pois)} POIs.")
        print("Suggestions available ('V' toggle), Skeleton overlay available ('K' toggle)")

    except Exception as e:
        print(f"ERROR during background analysis pipeline: {e}")
        analysis_complete = False
        maprec_nodes = []; maprec_pois = [] # Clear results on error
        suggested_intersections_orig = []; suggested_pois_orig = []
        # --- CLEAR SKELETON ON ERROR ---
        global_skeleton_img = None
        global_scaler_used = 1
        # --- END CLEAR ---
        import traceback
        print("Traceback:")
        traceback.print_exc()
    finally:
        analysis_in_progress = False
        # The main loop checks thread status, no need to reset analysis_thread here


# === Main Execution Block ===
if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Interactive floor plan map generator with MapRecognizer analysis.")
    # Default path from MapRecognizer example
    default_img_path = 'C:/Users/rohit/perceptaai/MapAnalysis/datasource/floor.png'
    parser.add_argument("image_path", nargs='?', default=default_img_path, help=f"Path to the floor plan image file (default: {default_img_path}).")
    parser.add_argument("-o", "--output", default="map_generated.json", help="Base name for output JSON files (default: map_generated.json). Actual files saved according to MapRecognizer structure.")
    parser.add_argument("--width", type=int, default=None, help=f"Maximum display window width (default: {MAX_DISPLAY_WIDTH}).")
    parser.add_argument("--height", type=int, default=None, help=f"Maximum display window height (default: {MAX_DISPLAY_HEIGHT}).")
    # Config path is currently hardcoded in the analysis function based on MapRecognizer example
    # parser.add_argument("--config", default=None, help="Path to optional YAML configuration file (Overrides internal defaults and MapRecognizer hardcoded path).")
    args = parser.parse_args()

    # --- Config Loading (Placeholder - currently uses hardcoded path in analysis) ---
    # If --config arg was used, it could override here, but analysis function needs update
    # if args.config and YAML_AVAILABLE:
    #     print("Note: --config argument provided, but analysis currently uses hardcoded path. Load attempt here is placeholder.")
        # ... (logic to load args.config into `config` potentially overriding DEFAULT_CONFIG)
    # else:
    #     print("Using internal default config or hardcoded path in analysis.")
    config = DEFAULT_CONFIG.copy() # Start with defaults, analysis will load its own

    # --- Load Image ---
    try:
        print(f"Attempting to load image: '{args.image_path}'")
        if not isinstance(args.image_path, str) or not os.path.exists(args.image_path):
             # Try constructing path relative to script if default path fails? Or just error out.
             raise FileNotFoundError(f"Image file not found or path invalid: '{args.image_path}'")
        original_image = cv2.imread(args.image_path)
        if original_image is None:
             # Add specific check for permissions or corruption if possible
             raise IOError(f"Could not decode image '{args.image_path}'. Check file format, path, and permissions.")
        original_h, original_w = original_image.shape[:2]
        print(f"Loaded image: '{args.image_path}' ({original_w}x{original_h})")
    except (FileNotFoundError, IOError, cv2.error) as e: print(f"FATAL ERROR loading image: {e}"); exit(1)
    except Exception as e: print(f"FATAL UNEXPECTED ERROR loading image: {e}"); exit(1)

    # --- Calculate Display Size ---
    try:
        max_w = int(args.width) if args.width else MAX_DISPLAY_WIDTH; max_h = int(args.height) if args.height else MAX_DISPLAY_HEIGHT
        if max_w <=0 or max_h <=0: raise ValueError("Max display dims must be positive.")
        if original_w <= 0 or original_h <= 0: raise ValueError("Original image dims invalid.")
        w_ratio = max_w / original_w if original_w > max_w else 1.0; h_ratio = max_h / original_h if original_h > max_h else 1.0
        resize_factor = min(w_ratio, h_ratio); resize_factor = max(1e-6, resize_factor) # Allow shrinking, ensure positive
        display_w = int(original_w * resize_factor); display_h = int(original_h * resize_factor)
        if display_w <= 0 or display_h <= 0: raise ValueError(f"Calculated display size invalid ({display_w}x{display_h}).")
        if abs(resize_factor - 1.0) < 1e-6: print(f"Displaying at original size.")
        else: print(f"Resized Display: {display_w}x{display_h} (Factor: {resize_factor:.3f})")
    except (ValueError, TypeError) as e: print(f"ERROR calculating display size: {e}. Using fallback."); display_w, display_h = min(original_w, MAX_DISPLAY_WIDTH), min(original_h, MAX_DISPLAY_HEIGHT); resize_factor=float(display_w)/original_w if original_w > 0 else 1.0
    except Exception as e: print(f"UNEXPECTED ERROR calculating display size: {e}"); exit(1)


    # --- Print Instructions ---
    print("\n--- Floor Plan Map Generator (with MapRecognizer Analysis) ---")
    print(" STATUS:"); print(f"  - Mode: {mode}"); print(f"  - Scale: {'Set' if scale_pixels_per_meter else 'Not Set'}")
    print(f"  - Output Base: '{args.output}' (Actual files saved in '{JSON_SAVE_PATH}')")
    print(" MODES:"); print("  [N] NODE"); print("  [L] LINK"); print("  [S] SCALE")
    print(" ACTIONS:")
    # Analysis uses MapRecognizer pipeline now
    if ANALYSIS_AVAILABLE: print("  [A] Re-Analyze (MapRecognizer)"); print("  [V] View Suggestions")
    else: print("  [A] Analyze: (Disabled)"); print("  [V] View Suggestions: (Disabled)")
    print("  [W] Write MapRecognizer JSONs"); print("  [Q] Quit"); print("  [ESC]: Quit Immediately")
    print("--------------------------------------------------------------")

    # --- OpenCV Window Setup ---
    try:
        cv2.namedWindow("Floor Plan Map Generator", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Floor Plan Map Generator", mouse_callback)
    except cv2.error as e: print(f"FATAL ERROR creating OpenCV window: {e}"); exit(1)
    except Exception as e: print(f"FATAL UNEXPECTED ERROR during OpenCV setup: {e}"); exit(1)

    # --- Automatically Start Analysis ---
    if ANALYSIS_AVAILABLE: # Check if core libs loaded
        print("\nStarting background analysis automatically...")
        if original_image is not None:
             if analysis_thread is None or not analysis_thread.is_alive():
                  # Create and start the analysis thread as a daemon thread
                  analysis_thread = threading.Thread(target=run_analysis_in_background, args=(args.image_path,), daemon=True)
                  analysis_thread.start()
             else: print("Warn: Analysis thread already exists?") # Should not happen at startup
        else: print("ERROR: Cannot start auto analysis, image invalid.")
    else: print("\nINFO: Automated analysis disabled (requirements not met)."); analysis_in_progress = False; analysis_complete = False

    keep_running = True; last_redraw_time = time.time(); min_redraw_interval = 0.03
    image_basename = os.path.basename(args.image_path)
    image_name_no_ext = os.path.splitext(image_basename)[0]

    while keep_running:
        current_time = time.time()
        force_redraw = False # Can be set to True if a key press necessitates immediate redraw

        # Redraw only if enough time has passed or forced
        if (current_time - last_redraw_time) >= min_redraw_interval or force_redraw:
             redraw_display()
             last_redraw_time = current_time

        key = cv2.waitKey(20) & 0xFF # 20ms wait

        # Check if window is still open
        try:
             if cv2.getWindowProperty("Floor Plan Map Generator", cv2.WND_PROP_VISIBLE) < 1: print("\nWindow closed."); keep_running = False; break
        except cv2.error: print("\nWindow destroyed."); keep_running = False; break
        except Exception as e: print(f"Error checking window status: {e}"); keep_running = False; break

        # --- Key Handling ---
        if key == ord('q'):
            print("\n'Q' pressed. Preparing to quit.")
            if analysis_thread and analysis_thread.is_alive():
                print("Waiting for analysis thread (max 5s)..."); analysis_thread.join(timeout=5.0)
                if analysis_thread.is_alive(): print("Warn: Analysis thread timeout.")
            save_before_quit = False
            if analysis_complete: # Only ask if analysis ran
                try: confirm = input("Save MapRecognizer results before quitting? (Y/n): ").lower().strip(); save_before_quit = (confirm != 'n')
                except EOFError: print("\nInput interrupted. Not saving.")
                except Exception as e: print(f"\nError during save confirmation: {e}. Not saving.")
            elif maprec_nodes or maprec_pois: print("Analysis incomplete, but some data exists. Not saving automatically.")
            else: print("No analysis data to save.")
            if save_before_quit: save_map(args.output, image_name_no_ext)
            keep_running = False; break

        elif key == 27: print("\nESC pressed, quitting immediately."); keep_running = False; break

        elif key == ord('n'):
             if mode != 'NODE': print("\nMODE: NODE"); mode = 'NODE'; link_start_node = None; scale_points_orig = []; force_redraw = True

        elif key == ord('l'):
             if mode != 'LINK_START' and mode != 'LINK_END':
                 if not nodes: print("\nAdd interactive nodes first!") # Link interactive nodes
                 else: print("\nMODE: LINK - Click start node"); mode = 'LINK_START'; link_start_node = None; scale_points_orig = []; force_redraw = True

        elif key == ord('s'):
             if mode != 'SCALE_START' and mode != 'SCALE_END': print("\nMODE: SCALE - Click first point"); mode = 'SCALE_START'; link_start_node = None; scale_points_orig = []; force_redraw = True

        elif key == ord('w'):
             if analysis_complete: save_map(args.output, image_name_no_ext)
             else: print("\nAnalysis not complete. Cannot save MapRecognizer results yet.")

        elif key == ord('a'):
            if ANALYSIS_AVAILABLE:
                if analysis_thread and analysis_thread.is_alive(): print("\nAnalysis is already in progress.")
                else:
                    confirm_rerun = True
                    if analysis_complete:
                        try: confirm = input(" > Analysis already completed. Re-run? (y/N): ").lower().strip(); confirm_rerun = (confirm == 'y')
                        except: confirm_rerun = False # Default no on error
                    if confirm_rerun:
                       print("\nStarting analysis (MapRecognizer)...")
                       if original_image is not None:
                           analysis_thread = threading.Thread(target=run_analysis_in_background, args=(args.image_path,), daemon=True); analysis_thread.start()
                           force_redraw = True # Update status
                       else: print("   Error: Image invalid.")
                    elif analysis_complete: print("   Keeping previous analysis results.")
            else: print("\nAnalysis disabled.")

        elif key == ord('v'): # Toggle Suggestions
             if ANALYSIS_AVAILABLE:
                 if analysis_complete:
                      show_suggestions = not show_suggestions
                      print(f"\nSuggestions Visibility: {'ON' if show_suggestions else 'OFF'}.")
                      force_redraw = True # Redraw to show/hide suggestions
                 elif analysis_in_progress: print("\nAnalysis running.")
                 else: print("\nAnalysis not complete/run yet.")
             else: print("\nAnalysis disabled.")

        # --- ADDED SKELETON TOGGLE KEY ---
        elif key == ord('k'): # Toggle Skeleton visibility
             if ANALYSIS_AVAILABLE:
                 if analysis_complete and global_skeleton_img is not None:
                     show_skeleton = not show_skeleton
                     print(f"\nSkeleton Overlay: {'ON' if show_skeleton else 'OFF'}.")
                     force_redraw = True # Redraw to show/hide skeleton
                 elif analysis_in_progress:
                     print("\nAnalysis running, skeleton not ready yet.")
                 elif global_skeleton_img is None and analysis_complete:
                      # Handle case where analysis finished but skeleton is None
                      print("\nAnalysis complete, but skeleton data is not available (check analysis steps).")
                 else: # Analysis not run/complete
                     print("\nRun analysis ('A') first to generate skeleton.")
             else:
                 print("\nAnalysis disabled.")
        # --- END ADD ---

        elif key != 0xFF: # 0xFF means no key pressed
             pass # Ignore other keys

    # --- Cleanup ---
    print("\nExiting map generator...")
    cv2.destroyAllWindows()
    if analysis_thread and analysis_thread.is_alive():
        print("Final wait for background thread..."); analysis_thread.join(timeout=2.0) # Short timeout
    print("Application closed.")
