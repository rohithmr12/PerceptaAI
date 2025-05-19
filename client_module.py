# client_module.py
import requests
import json
import os

SERVER_URL = "http://localhost:8000"  # Default server URL

def handle_response(response, action_name="Action"):
    """Handles and prints the server's response."""
    try:
        print(f"\nRaw server status code for {action_name}: {response.status_code}")
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        res_json = response.json()
        print(f"\n✅ {action_name} Successful:")
        if isinstance(res_json, dict):
            for key, value in res_json.items():
                # Print details, but skip if 'detail' is None or empty (FastAPI convention for no detail)
                if key.lower() != "detail" or value: 
                    print(f"   {key.replace('_', ' ').capitalize()}: {value}")
        else:
            print(f"   Response: {res_json}")
        return res_json
    except requests.exceptions.HTTPError as http_err:
        print(f"\n❌ HTTP Error for {action_name}: {http_err}")
        try:
            error_detail = response.json().get('detail', response.text) # Try to get FastAPI's detail
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
        files = {'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')}
        print(f"Uploading image {image_path} to describe scene...")
    elif image_path:
        json_payload = {'image_path': image_path}
        print(f"Requesting scene description for image path: {image_path}...")
    else:
        print("Requesting scene description (server may attempt capture)...")

    try:
        if files:
            response = requests.post(endpoint, files=files, data=json_payload if json_payload else None)
        else:
            response = requests.post(endpoint, json=json_payload)
        return handle_response(response, "Scene Description")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed for Scene Description: {e}")
    finally:
        if files and files['image'] and hasattr(files['image'][1], 'close'): 
            files['image'][1].close()
    return None

def ocr_client(image_path: str = None, use_upload: bool = False, preprocessing: str = "default", detect_regions: bool = False):
    endpoint = f"{SERVER_URL}/tools/ocr"
    files = None
    json_payload = {
        'preprocessing_type': preprocessing,
        'detect_regions': detect_regions
    }

    if use_upload and image_path and os.path.exists(image_path):
        files = {'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')}
        print(f"Uploading image {image_path} for OCR with preprocessing '{preprocessing}'...")
    elif image_path:
        json_payload['image_path'] = image_path
        print(f"Requesting OCR for image path: {image_path} with preprocessing '{preprocessing}'...")
    else:
        print(f"Requesting OCR (server may attempt capture) with preprocessing '{preprocessing}'...")

    try:
        if files:
            response = requests.post(endpoint, files=files, data=json_payload)
        else:
            response = requests.post(endpoint, json=json_payload)
        return handle_response(response, "OCR")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed for OCR: {e}")
    finally:
        if files and files['image'] and hasattr(files['image'][1], 'close'):
            files['image'][1].close()
    return None

def start_navigation_client(destination: str, start: str = "node1", map_file:str=None, yolo_file:str=None):
    endpoint = f"{SERVER_URL}/tools/start_navigation"
    payload = {
        "destination_node_id": destination,
        "start_node_id": start,
    }
    if map_file: payload["map_file"] = map_file
    if yolo_file: payload["yolo_model_file"] = yolo_file
    
    print(f"Requesting navigation from '{start}' to '{destination}'. This may take a while...")
    try:
        response = requests.post(endpoint, json=payload, timeout=None) 
        return handle_response(response, "Navigation")
    except requests.exceptions.Timeout:
        print("\n❌ Navigation request timed out. The server might still be processing.")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Navigation request failed: {e}")
    return None

def send_email_client(to: str, subject: str, body: str):
    endpoint = f"{SERVER_URL}/tools/send_email"
    payload = {"to": to, "subject": subject, "body": body}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, "Send Email (Mock)")

def send_message_client(to_contact: str, message_body: str):
    endpoint = f"{SERVER_URL}/tools/send_message"
    payload = {"to_contact": to_contact, "message_body": message_body}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, "Send Message (Mock)")

def get_weather_client(city: str):
    endpoint = f"{SERVER_URL}/tools/get_weather"
    payload = {"city": city}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, "Get Weather (Mock)")

def get_news_client(category: str = "general"):
    endpoint = f"{SERVER_URL}/tools/get_news"
    payload = {"category": category}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, "Get News (Mock)")

def add_todo_client(item: str):
    endpoint = f"{SERVER_URL}/tools/todos"
    payload = {"item": item}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, "Add To-Do")

def view_todos_client():
    endpoint = f"{SERVER_URL}/tools/todos"
    response = requests.get(endpoint)
    return handle_response(response, "View To-Dos")

def remove_todo_client(item_number: int):
    endpoint = f"{SERVER_URL}/tools/todos/{item_number}"
    response = requests.delete(endpoint)
    return handle_response(response, "Remove To-Do")

def calculate_client(expression: str):
    endpoint = f"{SERVER_URL}/tools/calculate"
    payload = {"expression": expression}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, "Calculate") 