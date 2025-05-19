import os
import shutil
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from pydantic import BaseModel
import uvicorn
import uuid # For temporary file names

# Core logic imports
from OCR import ocr_text as actual_ocr_implementation
from nav_core import NavigationCore, DEFAULT_MAP_FILE_PATH as NAV_DEFAULT_MAP_PATH, DEFAULT_YOLO_MODEL_PATH as NAV_DEFAULT_YOLO_PATH

# Daily life tools imports - using actual implementations now
from daily_tools.communication import send_email_actual, send_message_mock # Keep message as mock
from daily_tools.information import get_weather_actual, get_news_actual
from daily_tools import productivity # Productivity tools are already actual

# Utilities
UTILS_LOAD_ERROR = ""
CAPTURE_IMAGE_AVAILABLE = False
ENCODE_IMAGE_AVAILABLE = False
VISION_LLM_UTIL_AVAILABLE = False

def _dummy_initiate_llm(*args, **kwargs): return None
def _dummy_capture_image(): print("Warning: capture_image utility not found."); return None
def _dummy_encode_image_to_base64(path): print("Warning: encode_image_to_base64 utility not found."); return None

try:
    from utils.model import initiate_llm
    VISION_LLM_UTIL_AVAILABLE = True
except ImportError: UTILS_LOAD_ERROR += "Warning: 'initiate_llm' from 'utils.model' not found. "; initiate_llm = _dummy_initiate_llm
try:
    from utils.snap_a_picture import capture_image
    CAPTURE_IMAGE_AVAILABLE = True
except ImportError: UTILS_LOAD_ERROR += "Warning: 'capture_image' from 'utils.snap_a_picture' not found. "; capture_image = _dummy_capture_image
try:
    from utils.to_base64 import encode_image_to_base64
    ENCODE_IMAGE_AVAILABLE = True
except ImportError: UTILS_LOAD_ERROR += "Warning: 'encode_image_to_base64' from 'utils.to_base64' not found. "; encode_image_to_base64 = _dummy_encode_image_to_base64

# --- Configuration ---
SERVER_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio-dummy-key")
VISION_LLM_BASE_URL = "http://localhost:1234/v1" 
VISION_LLM_MODEL_ID = os.getenv("VISION_MODEL", "lmstudio-community/granite-vision-3.2-2b-GGUF")
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Percepta Assistant Server")

# Pydantic Models
class TextResponse(BaseModel): result: str; detail: Optional[str] = None
class ListResponse(BaseModel): result: List[str]; detail: Optional[str] = None
class SceneDescriptionResponse(BaseModel): description: str; image_path_used: Optional[str] = None; error: Optional[str] = None
class OCRResponse(BaseModel): text: str; image_path_used: Optional[str] = None; error: Optional[str] = None
class NavigationRequest(BaseModel): destination_node_id: str; start_node_id: Optional[str] = "node1"; map_file: Optional[str] = NAV_DEFAULT_MAP_PATH; yolo_model_file: Optional[str] = NAV_DEFAULT_YOLO_PATH
class EmailRequest(BaseModel): to: str; subject: str; body: str
class MessageRequest(BaseModel): to_contact: str; message_body: str
class WeatherRequest(BaseModel): city: str
class NewsRequest(BaseModel): category: Optional[str] = "general"; country: Optional[str] = "us" # Added country for news
class TodoItemRequest(BaseModel): item: str
class CalculateRequest(BaseModel): expression: str

async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".tmp"
        tmp_filename = os.path.join(TEMP_UPLOAD_DIR, f"{uuid.uuid4()}{suffix}")
        with open(tmp_filename, "wb") as buffer: shutil.copyfileobj(upload_file.file, buffer)
        return tmp_filename
    finally: upload_file.file.close()

vision_llm_instance = None; _vision_llm_init_error = ""
if VISION_LLM_UTIL_AVAILABLE:
    try:
        from langchain_openai import ChatOpenAI
        vision_llm_instance = initiate_llm(base_url=VISION_LLM_BASE_URL, model_id=VISION_LLM_MODEL_ID, temp=0.5, api_key_val=SERVER_API_KEY)
        if vision_llm_instance is None: _vision_llm_init_error = f"Vision LLM ({VISION_LLM_MODEL_ID}) init via util returned None. "
        else: _vision_llm_init_error = ""
    except Exception as e: _vision_llm_init_error = f"Error initializing vision LLM with initiate_llm: {e}. "; vision_llm_instance = None 
if vision_llm_instance is None: 
    _vision_llm_init_error += "Attempting fallback. "
    try:
        from langchain_openai import ChatOpenAI
        vision_llm_instance = ChatOpenAI(base_url=VISION_LLM_BASE_URL, model=VISION_LLM_MODEL_ID, api_key=SERVER_API_KEY, temperature=0.5, max_retries=1)
        _vision_llm_init_error = "" 
    except Exception as e: _vision_llm_init_error += f"Fallback init failed: {e}."; vision_llm_instance = None

# --- API Endpoints ---
@app.post("/tools/describe_scene", response_model=SceneDescriptionResponse)
async def describe_scene_endpoint(image_path: Optional[str] = Body(None, embed=True), image: Optional[UploadFile] = File(None)):
    if vision_llm_instance is None: return SceneDescriptionResponse(description="", error=f"Vision LLM not available. {_vision_llm_init_error or UTILS_LOAD_ERROR}")
    if not ENCODE_IMAGE_AVAILABLE: return SceneDescriptionResponse(description="", error="'encode_image_to_base64' utility not found.")
    processed_image_path = None; temp_file_created = False
    try:
        if image: processed_image_path = await save_upload_file_tmp(image); temp_file_created = True
        elif image_path and os.path.exists(image_path): processed_image_path = image_path
        elif image_path: return SceneDescriptionResponse(description="", error=f"image_path does not exist: {image_path}")
        else:
            if CAPTURE_IMAGE_AVAILABLE: processed_image_path = capture_image(); temp_file_created = True
            if not processed_image_path or not os.path.exists(processed_image_path): return SceneDescriptionResponse(description="", error="Failed to capture/find image for scene.")
        base64_uri = encode_image_to_base64(processed_image_path)
        if not base64_uri: return SceneDescriptionResponse(description="", image_path_used=processed_image_path, error="Failed to encode image.")
        from langchain_core.messages import HumanMessage 
        message = HumanMessage(content=[{"type": "text", "text": "Describe this image."}, {"type": "image_url", "image_url": {"url": base64_uri}}])
        response = vision_llm_instance.invoke([message])
        return SceneDescriptionResponse(description=response.content or "No description.", image_path_used=processed_image_path)
    except Exception as e: return SceneDescriptionResponse(description="", image_path_used=processed_image_path, error=f"Error: {str(e)}")
    finally:
        if temp_file_created and processed_image_path and os.path.exists(processed_image_path):
            try: os.remove(processed_image_path)
            except Exception as e_del: print(f"Error cleaning temp file {processed_image_path}: {e_del}")

@app.post("/tools/ocr", response_model=OCRResponse)
async def ocr_endpoint(image_path: Optional[str] = Body(None, embed=True), image: Optional[UploadFile] = File(None), preprocessing_type: str = Body("default", embed=True), detect_regions: bool = Body(False, embed=True)):
    processed_image_path = None; temp_file_created = False
    try:
        if image: processed_image_path = await save_upload_file_tmp(image); temp_file_created = True
        elif image_path and os.path.exists(image_path): processed_image_path = image_path
        elif image_path: return OCRResponse(text="", error=f"image_path does not exist: {image_path}")
        else:
            if CAPTURE_IMAGE_AVAILABLE: processed_image_path = capture_image(); temp_file_created = True
            if not processed_image_path or not os.path.exists(processed_image_path): return OCRResponse(text="", error="Failed to capture/find image for OCR.")
        result_text = actual_ocr_implementation(image_path=processed_image_path, preprocessing=preprocessing_type, detect_regions=detect_regions)
        return OCRResponse(text=result_text, image_path_used=processed_image_path)
    except Exception as e: return OCRResponse(text="", image_path_used=processed_image_path, error=f"Error during OCR: {str(e)}")
    finally:
        if temp_file_created and processed_image_path and os.path.exists(processed_image_path):
            try: os.remove(processed_image_path)
            except Exception as e_del: print(f"Error cleaning temp file {processed_image_path}: {e_del}")

@app.post("/tools/start_navigation", response_model=TextResponse)
async def navigation_endpoint(req: NavigationRequest):
    try:
        map_file_to_use = req.map_file or NAV_DEFAULT_MAP_PATH
        yolo_file_to_use = req.yolo_model_file or NAV_DEFAULT_YOLO_PATH
        if not os.path.exists(map_file_to_use): potential_map_path = os.path.join("Nav", "map_data", os.path.basename(map_file_to_use)); map_file_to_use = potential_map_path if os.path.exists(potential_map_path) else map_file_to_use
        if not os.path.exists(map_file_to_use): raise HTTPException(status_code=404, detail=f"Map file not found: {map_file_to_use}")
        nav_system = NavigationCore(map_filepath=map_file_to_use, start_node_id=req.start_node_id, end_node_id=req.destination_node_id, yolo_model_path=yolo_file_to_use)
        success = nav_system.run_navigation_loop()
        return TextResponse(result=f"Navigation to {req.destination_node_id} {'completed' if success else 'ended/failed'}.")
    except (ValueError, FileNotFoundError) as e: raise HTTPException(status_code=400, detail=f"Navigation Setup/File Error: {str(e)}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Navigation runtime error: {str(e)}")

@app.post("/tools/send_email", response_model=TextResponse)
async def send_email_endpoint(req: EmailRequest):
    result = send_email_actual(to_email=req.to, subject=req.subject, body=req.body)
    if "Missing configuration(s)" in result or "Invalid EMAIL_PORT" in result:
        raise HTTPException(status_code=400, detail=result)
    if "Failed to send email" in result or "Error:" in result or "Refused" in result or "Timeout" in result or "Authentication Error" in result or "SMTP" in result:
        raise HTTPException(status_code=500, detail=result)
    return TextResponse(result=result)

@app.post("/tools/send_message", response_model=TextResponse)
async def send_message_endpoint(req: MessageRequest): # Remains mock
    result = send_message_mock(to_contact=req.to_contact, message_body=req.message_body)
    return TextResponse(result=result)

@app.post("/tools/get_weather", response_model=TextResponse)
async def get_weather_endpoint(req: WeatherRequest):
    result = get_weather_actual(city=req.city)
    if "Could not find location" in result or "Error fetching weather" in result or "Error parsing weather data" in result:
        if "Could not find location" in result: raise HTTPException(status_code=404, detail=result)
        else: raise HTTPException(status_code=502, detail=result)
    return TextResponse(result=result)

@app.post("/tools/get_news", response_model=ListResponse)
async def get_news_endpoint(req: NewsRequest):
    headlines = get_news_actual(category=req.category, country=req.country)
    if headlines and isinstance(headlines, list) and headlines[0].startswith("NewsAPI key (NEWS_API_KEY) is not set"):
        raise HTTPException(status_code=400, detail=headlines[0])
    if headlines and isinstance(headlines, list) and (headlines[0].startswith("NewsAPI Error:") or headlines[0].startswith("Error fetching news") or headlines[0].startswith("Error parsing news data")):
        raise HTTPException(status_code=502, detail=". ".join(headlines))
    if not headlines or (isinstance(headlines, list) and "No news articles found" in headlines[0]):
        return ListResponse(result=headlines or ["No news found for your query."])
    return ListResponse(result=headlines)

@app.post("/tools/todos", response_model=TextResponse, status_code=201)
async def add_todo_endpoint(req: TodoItemRequest):
    try: response_data = productivity.add_todo_item(item=req.item); return TextResponse(result=f"{response_data['message']} Total: {response_data['total_items']}.")
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))

@app.get("/tools/todos", response_model=ListResponse)
async def view_todos_endpoint(): return ListResponse(result=productivity.view_todo_list())

@app.delete("/tools/todos/{item_number}", response_model=TextResponse)
async def remove_todo_endpoint(item_number: int):
    try: return TextResponse(result=productivity.remove_todo_item(item_number=item_number))
    except IndexError as e: raise HTTPException(status_code=404, detail=str(e))

@app.post("/tools/calculate", response_model=TextResponse)
async def calculate_endpoint(req: CalculateRequest):
    try: result_val = productivity.calculate_expression(expression=req.expression); return TextResponse(result=f"Result of '{req.expression}' is {result_val}.")
    except (ValueError, ZeroDivisionError) as e: raise HTTPException(status_code=400, detail=f"Calculation error: {str(e)}")

@app.get("/")
async def root(): return {"message": "Percepta Assistant Server is running. Append /docs for API details."}

if __name__ == "__main__":
    print("--- Percepta Server Starting ---")
    print(f"OpenAI API Key available: {'Yes' if SERVER_API_KEY != 'lm-studio-dummy-key' else 'No (using default)'}")
    print(f"Vision LLM ({VISION_LLM_MODEL_ID}) Initialized: {'Yes' if vision_llm_instance else 'No'}. Status: {_vision_llm_init_error or 'OK' if vision_llm_instance else _vision_llm_init_error}")
    if UTILS_LOAD_ERROR: print(f"Utility Loading Issues: {UTILS_LOAD_ERROR}")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on http://0.0.0.0:{port}. NEWS_API_KEY configured: {'Yes' if os.getenv('NEWS_API_KEY') else 'No'}. Email Conf: {'Partial/Full' if os.getenv('EMAIL_HOST') else 'No'}")
    uvicorn.run("percepta_server:app", host="0.0.0.0", port=port, reload=True) 