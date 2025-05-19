import requests
import json
import os
import shlex # For smarter command splitting

SERVER_URL = "http://localhost:8000" 

def handle_response(response, action_name="Action"):
    try:
        print(f"\nRaw server status code for {action_name}: {response.status_code}")
        response.raise_for_status() 
        res_json = response.json()
        print(f"\n✅ {action_name} Successful:")
        if isinstance(res_json, dict):
            for key, value in res_json.items():
                if key.lower() != "detail" or value: 
                    print(f"   {key.replace('_', ' ').capitalize()}: {value}")
        else:
            print(f"   Response: {res_json}")
        return res_json
    except requests.exceptions.HTTPError as http_err:
        print(f"\n❌ HTTP Error for {action_name}: {http_err}")
        try:
            error_detail = response.json().get('detail', response.text)
            print(f"   Server said: {error_detail}")
        except json.JSONDecodeError:
            print(f"   Server response was not JSON: {response.text}")
    except Exception as err:
        print(f"\n❌ Error processing response for {action_name}: {err}")
    return None

def describe_scene_client(image_path: str = None, use_upload: bool = False):
    endpoint = f"{SERVER_URL}/tools/describe_scene"
    files = None
    json_payload = {}

    if use_upload and image_path and os.path.exists(image_path):
        files = {'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')} # Specify content type
        # For FastAPI, when sending files and other JSON data, it's often easier if the other data is part of the query or form-data, not JSON body.
        # Or, send everything as multipart/form-data. Here, image_path is not needed if image is uploaded.
        print(f"Uploading image {image_path} to describe scene...")
    elif image_path:
        json_payload = {'image_path': image_path}
        print(f"Requesting scene description for image path: {image_path}...")
    else:
        # Server will attempt capture if no path/upload and capture is available
        print("Requesting scene description (server may attempt capture if configured and no image data sent)...")
        # No explicit image_path or image file in this case, server decides

    try:
        if files: # If uploading, send as multipart
            response = requests.post(endpoint, files=files, data=json_payload if json_payload else None)
        else: # If only path or nothing, send as JSON
            response = requests.post(endpoint, json=json_payload)
        handle_response(response, "Scene Description")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed for Scene Description: {e}")
    finally:
        if files and files['image'] and hasattr(files['image'][1], 'close'): 
            files['image'][1].close()


def ocr_client(image_path: str = None, use_upload: bool = False, preprocessing: str = "default", detect_regions: bool = False):
    endpoint = f"{SERVER_URL}/tools/ocr"
    files = None
    json_payload = {
        'preprocessing_type': preprocessing,
        'detect_regions': detect_regions
    }

    if use_upload and image_path and os.path.exists(image_path):
        files = {'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')}
        # image_path is not needed in json_payload if image is uploaded
        print(f"Uploading image {image_path} for OCR with preprocessing '{preprocessing}'...")
    elif image_path:
        json_payload['image_path'] = image_path
        print(f"Requesting OCR for image path: {image_path} with preprocessing '{preprocessing}'...")
    else:
        print(f"Requesting OCR (server may attempt capture) with preprocessing '{preprocessing}'...")
        # Server handles capture if no image_path and no file

    try:
        if files:
            # When sending files, other data should be in `data` part of multipart, not json kwarg
            response = requests.post(endpoint, files=files, data=json_payload)
        else:
            response = requests.post(endpoint, json=json_payload)
        handle_response(response, "OCR")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed for OCR: {e}")
    finally:
        if files and files['image'] and hasattr(files['image'][1], 'close'):
            files['image'][1].close()

def start_navigation_client(destination: str, start: str = "node1", map_file:str=None, yolo_file:str=None):
    endpoint = f"{SERVER_URL}/tools/start_navigation"
    payload = {
        "destination_node_id": destination,
        "start_node_id": start,
    }
    if map_file: payload["map_file"] = map_file
    if yolo_file: payload["yolo_model_file"] = yolo_file
    
    print(f"Requesting navigation from '{start}' to '{destination}'. This may take a while and block here...")
    try:
        # Navigation can be very long, so no timeout or a very long one.
        response = requests.post(endpoint, json=payload, timeout=None) 
        handle_response(response, "Navigation")
    except requests.exceptions.Timeout:
        print("\n❌ Navigation request timed out. The server might still be processing.")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Navigation request failed: {e}")

def send_email_client(to: str, subject: str, body: str):
    endpoint = f"{SERVER_URL}/tools/send_email"
    payload = {"to": to, "subject": subject, "body": body}
    response = requests.post(endpoint, json=payload)
    handle_response(response, "Send Email (Mock)")

def send_message_client(to_contact: str, message_body: str):
    endpoint = f"{SERVER_URL}/tools/send_message"
    payload = {"to_contact": to_contact, "message_body": message_body}
    response = requests.post(endpoint, json=payload)
    handle_response(response, "Send Message (Mock)")

def get_weather_client(city: str):
    endpoint = f"{SERVER_URL}/tools/get_weather"
    payload = {"city": city}
    response = requests.post(endpoint, json=payload)
    handle_response(response, "Get Weather (Mock)")

def get_news_client(category: str = "general"):
    endpoint = f"{SERVER_URL}/tools/get_news"
    payload = {"category": category}
    response = requests.post(endpoint, json=payload)
    handle_response(response, "Get News (Mock)")

def add_todo_client(item: str):
    endpoint = f"{SERVER_URL}/tools/todos"
    payload = {"item": item}
    response = requests.post(endpoint, json=payload)
    handle_response(response, "Add To-Do")

def view_todos_client():
    endpoint = f"{SERVER_URL}/tools/todos"
    response = requests.get(endpoint)
    handle_response(response, "View To-Dos")

def remove_todo_client(item_number: int):
    endpoint = f"{SERVER_URL}/tools/todos/{item_number}"
    response = requests.delete(endpoint)
    handle_response(response, "Remove To-Do")

def calculate_client(expression: str):
    endpoint = f"{SERVER_URL}/tools/calculate"
    payload = {"expression": expression}
    response = requests.post(endpoint, json=payload)
    handle_response(response, "Calculate")

def main_cli():
    print("\n--- Percepta Client CLI ---")
    print(f"Connecting to server at: {SERVER_URL}")
    print("Type 'help' for commands, 'quit' to exit.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_test_image = os.path.join(script_dir, "test.jpg") 
    if not os.path.exists(default_test_image):
        print(f"\nNote: Default test image '{default_test_image}' not found. Image commands might require a full path.")

    while True:
        try:
            cmd_input = input("\nClient> ").strip()
            if not cmd_input: continue
            
            parts = shlex.split(cmd_input) # Use shlex for better quote handling
            command = parts[0].lower() if parts else ""

            if command == "quit": break
            if command == "help":
                print("\nAvailable commands:")
                print("  describe                       (Server attempts capture, if able)")
                print("  describe path <image_path>     (Uses local path, quote path if spaces)")
                print("  describe upload <image_path>   (Uploads local image, quote path if spaces)")
                print("  ocr                            (Server attempts capture, if able)")
                print("  ocr path <image_path> [doc]    (Optional 'doc' for document preprocessing)")
                print("  ocr upload <image_path> [doc]  (Optional 'doc' for document preprocessing)")
                print("  nav <destination_node> [start_node] [map_file_path] [yolo_model_path]")
                print("  email <to> <subject> <\"body text\">")
                print("  msg <contact> <\"message text\">")
                print("  weather <city_name>")
                print("  news [category_name]")
                print("  todo add <\"item description\">")
                print("  todo view")
                print("  todo remove <item_number>")
                print("  calc <\"expression string\">")
                print("  quit")
                continue
            
            if command == "describe":
                if len(parts) == 1: describe_scene_client()
                elif len(parts) >= 3 and parts[1] == "path": describe_scene_client(image_path=" ".join(parts[2:]))
                elif len(parts) >= 3 and parts[1] == "upload": describe_scene_client(image_path=" ".join(parts[2:]), use_upload=True)
                else: print("Usage: describe [path <image_path> | upload <image_path>]")
            
            elif command == "ocr":
                prep_type = "document" if "doc" in parts else "default"
                path_to_ocr = None
                is_upload = False
                if "path" in parts:
                    try: path_to_ocr = parts[parts.index("path") + 1]
                    except IndexError: print("Path missing after 'path' keyword."); continue
                elif "upload" in parts:
                    try: path_to_ocr = parts[parts.index("upload") + 1]; is_upload = True
                    except IndexError: print("Path missing after 'upload' keyword."); continue
                
                ocr_client(image_path=path_to_ocr, use_upload=is_upload, preprocessing=prep_type)

            elif command == "nav":
                if len(parts) >= 2:
                    dest = parts[1]
                    start = parts[2] if len(parts) > 2 else "node1"
                    map_f = parts[3] if len(parts) > 3 else None
                    yolo_f = parts[4] if len(parts) > 4 else None
                    start_navigation_client(destination=dest, start=start, map_file=map_f, yolo_file=yolo_f)
                else: print("Usage: nav <destination> [start_node] [map_file] [yolo_file]")
            
            elif command == "email":
                if len(parts) >= 4: send_email_client(to=parts[1], subject=parts[2], body=" ".join(parts[3:]))
                else: print("Usage: email <to> <subject> <body_text>")
            
            elif command == "msg":
                if len(parts) >= 3: send_message_client(to_contact=parts[1], message_body=" ".join(parts[2:]))
                else: print("Usage: msg <contact> <message_body>")

            elif command == "weather":
                if len(parts) >= 2: get_weather_client(city=" ".join(parts[1:]))
                else: print("Usage: weather <city_name>")
            
            elif command == "news":
                category = " ".join(parts[1:]) if len(parts) > 1 else "general"
                get_news_client(category=category)

            elif command == "todo":
                if len(parts) > 1:
                    action = parts[1]
                    if action == "add" and len(parts) > 2: add_todo_client(item=" ".join(parts[2:]))
                    elif action == "view": view_todos_client()
                    elif action == "remove" and len(parts) > 2:
                        try: remove_todo_client(item_number=int(parts[2]))
                        except ValueError: print("Error: Item number must be an integer.")
                    else: print("Usage: todo [add <item> | view | remove <number>]")
                else: print("Usage: todo [add <item> | view | remove <number>]")
            
            elif command == "calc":
                if len(parts) > 1: calculate_client(expression=" ".join(parts[1:]))
                else: print("Usage: calc <expression_string>")
            
            elif command: 
                print(f"Unknown command: '{command}'. Type 'help'.")

        except Exception as e:
            print(f"CLI Error: {e}")

    print("Client exited.")

if __name__ == "__main__":
    main_cli() 