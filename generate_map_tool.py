import cv2
import json
import math
import os
import numpy as np
import argparse

# --- Configuration ---
NODE_COLOR = (0, 255, 0)      # Green for nodes
NODE_RADIUS = 5              # Smaller radius for potentially smaller display
TEXT_COLOR = (255, 255, 255) # White text
LINK_COLOR = (255, 0, 0)      # Blue for links
HIGHLIGHT_COLOR = (0, 255, 255) # Yellow for selection
MAX_CLICK_DISTANCE_NODE = 15 # Adjust max click distance for resized display

# --- Display Size Limits ---
MAX_DISPLAY_WIDTH = 1600  # Adjust as needed for your screen
MAX_DISPLAY_HEIGHT = 900 # Adjust as needed for your screen

# --- Global State Variables ---
original_image = None       # The original loaded image (full size)
display_image = None        # Resized image copy to draw on
nodes = []                  # List to store node dictionaries (using ORIGINAL coordinates)
links = []                  # List to store link dictionaries (pairs of node IDs)
node_counter = 1            # Simple counter for unique node IDs

mode = 'NODE'               # Current interaction mode
scale_points_orig = []      # Store the two points clicked for scaling (ORIGINAL coordinates)
scale_pixels_per_meter = None # Calculated scale factor (pixels / meter) - relative to ORIGINAL image

link_start_node = None      # Temporarily store the first node clicked for linking

resize_factor = 1.0         # Factor by which the image was resized for display
original_h, original_w = 0, 0 # Dimensions of the original image
display_h, display_w = 0, 0   # Dimensions of the resized display image

# --- Helper Functions ---

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_real_distance(node1, node2):
    """Calculates real-world distance between two nodes using the scale (using original coordinates)."""
    if not node1 or not node2:
        return float('inf')
    # Use original pixel coordinates stored in nodes
    pixel_dist_orig = calculate_distance((node1['pixel_x'], node1['pixel_y']), (node2['pixel_x'], node2['pixel_y']))
    if scale_pixels_per_meter and scale_pixels_per_meter > 0:
        return pixel_dist_orig / scale_pixels_per_meter
    else:
        print("WARN: Scale not set. Using original pixel distance for link weight.")
        return pixel_dist_orig

def get_display_coords(orig_x, orig_y):
    """Converts original image coordinates to display image coordinates."""
    return int(orig_x * resize_factor), int(orig_y * resize_factor)

def get_original_coords(display_x, display_y):
    """Converts display image coordinates back to original image coordinates."""
    return int(display_x / resize_factor), int(display_y / resize_factor)

def find_nearest_node(display_x, display_y, max_dist=MAX_CLICK_DISTANCE_NODE):
    """Finds the nearest node within max_dist pixels of the click ON THE DISPLAY."""
    nearest_node = None
    min_dist_display = float('inf')

    for node in nodes:
        # Convert node's original coords to display coords for comparison
        node_disp_x, node_disp_y = get_display_coords(node['pixel_x'], node['pixel_y'])
        dist_display = calculate_distance((display_x, display_y), (node_disp_x, node_disp_y))

        if dist_display < min_dist_display and dist_display <= max_dist:
            min_dist_display = dist_display
            nearest_node = node
    return nearest_node

def get_node_by_id(node_id):
    """Retrieves a node dictionary by its ID."""
    for node in nodes:
        if node['id'] == node_id:
            return node
    return None

def redraw_display():
    """Redraws the RESIZED image with current nodes and links."""
    global display_image
    # Start fresh from a resized copy of the original
    # (Could optimize by drawing on current display_image if performance is an issue)
    display_image = cv2.resize(original_image, (display_w, display_h))

    # Draw Links (using display coordinates)
    for link in links:
        node1 = get_node_by_id(link['startNode'])
        node2 = get_node_by_id(link['endNode'])
        if node1 and node2:
            pt1_disp = get_display_coords(node1['pixel_x'], node1['pixel_y'])
            pt2_disp = get_display_coords(node2['pixel_x'], node2['pixel_y'])
            cv2.line(display_image, pt1_disp, pt2_disp, LINK_COLOR, 2)

    # Draw Nodes (and highlight selected start node for linking)
    for node in nodes:
        center_disp = get_display_coords(node['pixel_x'], node['pixel_y'])
        color = NODE_COLOR
        radius = NODE_RADIUS
        # Highlight if it's the starting node for a link
        if mode == 'LINK_END' and link_start_node and node['id'] == link_start_node['id']:
             color = HIGHLIGHT_COLOR
             radius = NODE_RADIUS + 2 # Make it slightly bigger

        cv2.circle(display_image, center_disp, radius, color, -1) # Filled circle
        # Draw node ID text nearby (on display image)
        text_pos_disp = (center_disp[0] + radius + 2, center_disp[1] + radius + 2)
        cv2.putText(display_image, node['id'], text_pos_disp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA) # Slightly smaller font

    # Draw scale points if in scaling mode (on display image)
    if mode in ['SCALE_START', 'SCALE_END'] and scale_points_orig:
        for pt_orig in scale_points_orig:
            pt_disp = get_display_coords(pt_orig[0], pt_orig[1])
            cv2.drawMarker(display_image, pt_disp, HIGHLIGHT_COLOR, cv2.MARKER_CROSS, 15, 2)

    cv2.imshow("Floor Plan Map Generator", display_image)


def mouse_callback(event, x_disp, y_disp, flags, param):
    """Handles mouse click events on the DISPLAY window."""
    global mode, node_counter, link_start_node, scale_points_orig, scale_pixels_per_meter

    # --- Convert display coordinates to original image coordinates ---
    x_orig, y_orig = get_original_coords(x_disp, y_disp)

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at display ({x_disp}, {y_disp}) -> original ({x_orig}, {y_orig}) in mode: {mode}")

        # === NODE MODE: Add a new node ===
        if mode == 'NODE':
            node_id = f"node{node_counter}"
            print(f"Adding node {node_id} at original coordinates ({x_orig}, {y_orig}).")

            # Get node details from user via console input
            while True: name = input(f" > Enter node name (e.g., Entrance, Room 101) [Leave blank for none]: ").strip(); break
            while True: node_type = input(f" > Enter node type (e.g., Room, Intersection, Elevator) [Leave blank for generic]: ").strip(); break
            while True: landmark = input(f" > Enter nearby landmark (optional) [Leave blank for none]: ").strip(); break

            # Calculate real coordinates using original coords if scale is set
            real_x, real_y = x_orig, y_orig # Default to pixel coords if no scale
            if scale_pixels_per_meter and scale_pixels_per_meter > 0:
                real_x = x_orig / scale_pixels_per_meter
                real_y = y_orig / scale_pixels_per_meter # Assuming origin is top-left
                print(f"   Scaled coordinates: ({real_x:.2f}m, {real_y:.2f}m)")

            node_data = {
                "id": node_id,
                "name": name if name else f"Point {node_counter}",
                "type": node_type if node_type else "PointOfInterest",
                "landmark": landmark if landmark else None,
                "pixel_x": x_orig,  # Store ORIGINAL pixel coords
                "pixel_y": y_orig,
                "x": real_x,       # Store potentially scaled coords for map.json
                "y": real_y,
            }
            if not node_data["landmark"]: del node_data["landmark"]

            nodes.append(node_data)
            node_counter += 1
            redraw_display()
            print(f"Node {node_id} added. Current mode: {mode}. Press 'L' for Link mode, 'S' for Scale mode.")

        # === LINK MODE: Select start node ===
        elif mode == 'LINK_START':
            # Find nearest node using DISPLAY coordinates
            clicked_node = find_nearest_node(x_disp, y_disp)
            if clicked_node:
                link_start_node = clicked_node
                print(f"Link start node selected: {link_start_node['id']} ({link_start_node['name']})")
                mode = 'LINK_END'
                redraw_display() # Highlight the selected node
                print(f"Current mode: {mode}. Click on the destination node.")
            else:
                print("No node found near click. Click closer to a node to start a link.")

        # === LINK MODE: Select end node and create link ===
        elif mode == 'LINK_END':
             # Find nearest node using DISPLAY coordinates
            clicked_node = find_nearest_node(x_disp, y_disp)
            if clicked_node and link_start_node:
                if clicked_node['id'] == link_start_node['id']:
                    print("Cannot link a node to itself. Click a different node.")
                else:
                    link_end_node = clicked_node
                    print(f"Link end node selected: {link_end_node['id']} ({link_end_node['name']})")

                    # Check if link (or reverse) already exists
                    link_exists = False
                    for lnk in links:
                        if (lnk['startNode'] == link_start_node['id'] and lnk['endNode'] == link_end_node['id']) or \
                           (lnk['startNode'] == link_end_node['id'] and lnk['endNode'] == link_start_node['id']):
                            link_exists = True
                            break

                    if link_exists:
                        print(f"Link between {link_start_node['id']} and {link_end_node['id']} already exists.")
                    else:
                        links.append({"startNode": link_start_node['id'], "endNode": link_end_node['id']})
                        print(f"Link created: {link_start_node['id']} <-> {link_end_node['id']}")
                        redraw_display()

                    # Reset for next link
                    link_start_node = None
                    mode = 'LINK_START'
                    print(f"\nLink added. Current mode: {mode}. Click start node for next link, or press 'N' for Node mode.")

            elif not clicked_node:
                print("No node found near click. Click closer to the destination node.")
            else: # link_start_node was somehow None
                print("Error: Link start node was lost. Resetting.")
                link_start_node = None
                mode = 'LINK_START'
                redraw_display()

        # === SCALE MODE: Select first point ===
        elif mode == 'SCALE_START':
             # Store ORIGINAL coordinates
             scale_points_orig = [(x_orig, y_orig)]
             print(f"Scale: First point selected at original ({x_orig},{y_orig}).")
             mode = 'SCALE_END'
             redraw_display() # Show marker on display
             print(f"Current mode: {mode}. Click the second point for scaling.")

        # === SCALE MODE: Select second point and calculate scale ===
        elif mode == 'SCALE_END':
             if not scale_points_orig:
                 mode = 'SCALE_START'; return
             # Store second point's ORIGINAL coordinates
             scale_points_orig.append((x_orig, y_orig))
             print(f"Scale: Second point selected at original ({x_orig},{y_orig}).")

             # Calculate distance using ORIGINAL coordinates
             pixel_distance_orig = calculate_distance(scale_points_orig[0], scale_points_orig[1])
             print(f"   Pixel distance between points (original image): {pixel_distance_orig:.2f} pixels")

             if pixel_distance_orig < 1:
                 print("Scale points are too close in original image. Please try again.")
                 scale_points_orig = []
                 mode = 'SCALE_START'
                 redraw_display()
                 return

             while True: # Get real-world distance input
                 try:
                     real_distance_str = input(" > Enter the real-world distance between these points in METERS: ")
                     real_distance_meters = float(real_distance_str)
                     if real_distance_meters <= 0: print("Distance must be positive.")
                     else: break
                 except ValueError: print("Invalid input. Please enter a number (e.g., 5.5).")

             # Scale is pixels per meter on the ORIGINAL image
             scale_pixels_per_meter = pixel_distance_orig / real_distance_meters
             print(f"--- Scale Set: {scale_pixels_per_meter:.2f} original pixels per meter ---")

             # Optionally update existing nodes' real coordinates
             confirm_update = input(" > Update existing node coordinates with this new scale? (y/N): ").lower()
             if confirm_update == 'y':
                 print("   Updating existing node real (x, y) coordinates...")
                 for node in nodes:
                     # Use node's stored original pixel coords
                     node['x'] = node['pixel_x'] / scale_pixels_per_meter
                     node['y'] = node['pixel_y'] / scale_pixels_per_meter
                 print("   Existing node coordinates updated.")

             scale_points_orig = [] # Clear points
             mode = 'NODE'        # Return to node mode
             redraw_display()
             print(f"Current mode: {mode}. Continue adding nodes or press 'L' for Link mode.")


def save_map(filepath="map.json"):
    """Builds the final map structure and saves it to a JSON file."""
    print(f"\nSaving map data to {filepath}...")
    if not nodes:
        print("WARN: No nodes were defined. Saving an empty map.")

    map_data = {"nodes": [], "links": []}

    # Prepare nodes for JSON (using potentially scaled 'x', 'y' relative to original)
    for node in nodes:
        json_node = node.copy()
        # Remove tool-specific pixel coordinates before saving
        if 'pixel_x' in json_node: del json_node['pixel_x']
        if 'pixel_y' in json_node: del json_node['pixel_y']
        map_data["nodes"].append(json_node)

    # Prepare links and calculate weights (using original coords)
    final_links = []
    links_added = set()
    for link in links:
        start_id, end_id = link['startNode'], link['endNode']
        if (start_id, end_id) in links_added or (end_id, start_id) in links_added: continue

        node1, node2 = get_node_by_id(start_id), get_node_by_id(end_id)
        if node1 and node2:
            # get_real_distance already uses original pixel coords
            distance = get_real_distance(node1, node2)
            final_links.append({"startNode": start_id, "endNode": end_id}) # Weight calculated by NodeMapManager later
            links_added.add((start_id, end_id))
        else:
            print(f"WARN: Could not find nodes for link {start_id} <-> {end_id}. Skipping link.")

    map_data["links"] = final_links

    # Write the JSON file
    try:
        with open(filepath, 'w') as f: json.dump(map_data, f, indent=2)
        print(f"Map successfully saved to {filepath}")
    except Exception as e: print(f"ERROR: Could not save map file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-automated floor plan map generator with display resizing.")
    parser.add_argument("image_path", help="Path to the floor plan image file.")
    parser.add_argument("-o", "--output", default="map.json", help="Output path for the map.json file.")
    parser.add_argument("--width", type=int, default=MAX_DISPLAY_WIDTH, help="Maximum display window width.")
    parser.add_argument("--height", type=int, default=MAX_DISPLAY_HEIGHT, help="Maximum display window height.")
    args = parser.parse_args()

    MAX_DISPLAY_WIDTH = args.width
    MAX_DISPLAY_HEIGHT = args.height

    if not os.path.exists(args.image_path):
        print(f"ERROR: Image file not found at '{args.image_path}'"); exit(1)

    original_image = cv2.imread(args.image_path)
    if original_image is None:
        print(f"ERROR: Could not load image from '{args.image_path}'"); exit(1)

    original_h, original_w = original_image.shape[:2]

    # --- Calculate resize factor ---
    w_ratio = MAX_DISPLAY_WIDTH / original_w if original_w > MAX_DISPLAY_WIDTH else 1.0
    h_ratio = MAX_DISPLAY_HEIGHT / original_h if original_h > MAX_DISPLAY_HEIGHT else 1.0
    resize_factor = min(w_ratio, h_ratio, 1.0) # Take the smaller ratio, but don't enlarge (<= 1.0)

    if resize_factor < 1.0:
         display_w = int(original_w * resize_factor)
         display_h = int(original_h * resize_factor)
         print(f"Original image size: {original_w}x{original_h}")
         print(f"Resizing display to: {display_w}x{display_h} (Factor: {resize_factor:.3f})")
         display_image = cv2.resize(original_image, (display_w, display_h))
    else:
        # No resizing needed
        display_w, display_h = original_w, original_h
        display_image = original_image.copy()
        resize_factor = 1.0
        print(f"Image size {original_w}x{original_h} fits within limits. No display resizing needed.")


    # --- Instructions ---
    print("\n--- Floor Plan Map Generator (Resized Display) ---")
    print("Instructions:")
    print(" - Mode 'NODE' (default): Click to add a navigation node.")
    print(" - Mode 'LINK': Click two existing nodes to create a path.")
    print(" - Mode 'SCALE': Click two points, then enter the real distance (METERS).")
    print("Controls:")
    print(" - Left Click: Perform action based on current mode.")
    print(" - 'N': Switch to NODE mode.")
    print(" - 'L': Switch to LINK mode.")
    print(" - 'S': Switch to SCALE mode.")
    print(" - 'W': Write current nodes/links to map file.")
    print(" - 'Q': Quit (prompts to save).")
    print("---------------------------------")
    print(f"Current Mode: {mode}")

    # --- OpenCV Window Setup ---
    cv2.namedWindow("Floor Plan Map Generator") # Can add WINDOW_NORMAL for manual resize later if needed
    cv2.setMouseCallback("Floor Plan Map Generator", mouse_callback)
    redraw_display() # Initial display (already resized)

    # --- Interaction Loop ---
    while True:
        key = cv2.waitKey(0) & 0xFF # Wait indefinitely until a key is pressed

        if key == ord('q'): # Quit
            confirm = input("Save before quitting? (Y/n): ").lower()
            if confirm != 'n': save_map(args.output)
            break
        elif key == ord('n'): # Switch to Node mode
             if mode != 'NODE':
                 print("\nSwitched to NODE mode. Click to add nodes.")
                 mode = 'NODE'; link_start_node = None; scale_points_orig = []
                 redraw_display()
        elif key == ord('l'): # Switch to Link mode
             if mode != 'LINK_START' and mode != 'LINK_END':
                 if not nodes: print("Add nodes first!"); continue
                 print("\nSwitched to LINK mode. Click the starting node.")
                 mode = 'LINK_START'; link_start_node = None; scale_points_orig = []
                 redraw_display()
        elif key == ord('s'): # Switch to Scale mode
             if mode != 'SCALE_START' and mode != 'SCALE_END':
                 print("\nSwitched to SCALE mode. Click the first point.")
                 mode = 'SCALE_START'; link_start_node = None; scale_points_orig = []
                 redraw_display()
        elif key == ord('w'): # Write (Save)
             save_map(args.output)
             print(f"Map saved to {args.output}. Continue editing or press 'Q' to quit.")


    cv2.destroyAllWindows()
    print("Map generator closed.")