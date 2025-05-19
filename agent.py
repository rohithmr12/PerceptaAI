import os
from typing import Annotated, TypedDict, List, Optional

# Langchain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Tool specific imports
from OCR import ocr_text # From OCR.py
from Sceene_description import sceene_description_with_tts # From Sceene_description.py
from nav_core import NavigationCore, DEFAULT_MAP_FILE_PATH as NAV_DEFAULT_MAP_PATH, DEFAULT_YOLO_MODEL_PATH as NAV_DEFAULT_YOLO_PATH # From nav_core.py

# Attempt to import utilities (assuming they are in a 'utils' directory)
# These are crucial for some tools to function.
try:
    from utils.model import initiate_tts_model
    TTS_MODEL_AVAILABLE = True
except ImportError:
    print("Warning: 'initiate_tts_model' from 'utils.model' not found. TTS-dependent tools may fail or have limited functionality.")
    TTS_MODEL_AVAILABLE = False
    def initiate_tts_model(desired_device='cpu'): return None # Dummy function

try:
    from utils.snap_a_picture import capture_image
    CAPTURE_IMAGE_AVAILABLE = True
except ImportError:
    print("Warning: 'capture_image' from 'utils.snap_a_picture' not found. Image capture features will be disabled.")
    CAPTURE_IMAGE_AVAILABLE = False
    def capture_image(): return None # Dummy function

# --- Agent Configuration ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
# IMPORTANT: Replace with your actual model identifier from LM Studio
LM_STUDIO_MODEL_IDENTIFIER = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" 
os.environ["OPENAI_API_KEY"] = "lm-studio" # Required by ChatOpenAI, but not used by LM Studio

# --- Shared TTS Pipeline (initialized once) ---
shared_tts_pipeline_global = None
if TTS_MODEL_AVAILABLE:
    print("Initializing shared TTS model for the agent...")
    try:
        shared_tts_pipeline_global = initiate_tts_model(desired_device='cpu') 
        if shared_tts_pipeline_global is None:
            print("Warning: Shared TTS model failed to load via initiate_tts_model. TTS features will be impacted.")
    except Exception as e:
        print(f"Error initializing shared TTS model: {e}. TTS features will be impacted.")
else:
    print("TTS model utility not available. TTS-based tools will not speak.")

# --- Tool Definitions ---
@tool
def describe_current_scene() -> str:
    """Captures an image of the current surroundings, describes it using an AI model, and speaks the description aloud."""
    print("Tool: describe_current_scene called.")
    if not TTS_MODEL_AVAILABLE or shared_tts_pipeline_global is None:
        return "Error: TTS system is not available, so scene description cannot be spoken."
    if not CAPTURE_IMAGE_AVAILABLE: # sceene_description_with_tts depends on capture_image from utils
        return "Error: Image capture utility (snap_a_picture) is not available for scene description."
    try:
        # sceene_description_with_tts handles its own image capture and LLM call.
        sceene_description_with_tts(tts_pipeline=shared_tts_pipeline_global)
        return "Scene description process has been spoken."
    except Exception as e:
        print(f"Error in describe_current_scene tool: {e}")
        return f"Failed to describe scene: {str(e)}"

@tool
def read_text_from_image(image_path: Optional[str] = None, preprocessing_type: str = "default") -> str:
    """
    Reads text from an image. 
    Args:
        image_path (Optional[str]): Path to the image file. If None, an attempt will be made to capture a new image.
        preprocessing_type (str): Type of preprocessing for OCR (e.g., 'default', 'document', 'adaptive').
    """
    print(f"Tool: read_text_from_image called. Image: {image_path}, Preprocessing: {preprocessing_type}")
    current_image_path = image_path
    if not current_image_path:
        if not CAPTURE_IMAGE_AVAILABLE:
            return "Error: Image capture utility is not available, and no image_path was provided."
        try:
            print("No image path provided for OCR, attempting to capture a new image.")
            current_image_path = capture_image() # This util should save file and return path
            if not current_image_path or not os.path.exists(current_image_path):
                return "Failed to capture image for OCR, or image not saved correctly."
            print(f"Captured image for OCR: {current_image_path}")
        except Exception as e:
            return f"Error capturing image for OCR: {str(e)}"
    
    if not os.path.exists(current_image_path):
        return f"Error: Image file not found at {current_image_path}"

    try:
        extracted_text = ocr_text(image_path=current_image_path, preprocessing=preprocessing_type)
        # Let the LLM decide how to present "No text detected" or errors.
        return f"OCR Result: {extracted_text}" 
    except Exception as e:
        print(f"Error in read_text_from_image tool: {e}")
        return f"Failed to read text from image: {str(e)}"

@tool
def start_indoor_navigation_guidance(destination_node_id: str, start_node_id: str = "node1", map_file: str = NAV_DEFAULT_MAP_PATH, yolo_model_file: str = NAV_DEFAULT_YOLO_PATH) -> str:
    """
    Initiates indoor navigation to a specified destination node (e.g., "node5", "Room 101").
    Args:
        destination_node_id (str): The ID or common name of the destination.
        start_node_id (str): Optional. The ID of the starting node. Defaults to "node1" or a map default.
        map_file (str): Optional. Path to the map data file. Uses default if not specified.
        yolo_model_file (str): Optional. Path to the YOLO model file. Uses default if not specified.
    The system provides turn-by-turn voice guidance and obstacle alerts using the camera. This is a continuous process.
    """
    print(f"Tool: start_indoor_navigation_guidance. Dest: {destination_node_id}, Start: {start_node_id}, Map: {map_file}")
    try:
        if not os.path.exists(map_file):
            # Attempt to construct path relative to Nav directory if a simple filename is given for default
            potential_map_path = os.path.join("Nav", "map_data", os.path.basename(map_file))
            if os.path.exists(potential_map_path):
                map_file = potential_map_path
                print(f"Using map file from relative path: {map_file}")
            else:
                return f"Error: Map file not found at specified path '{map_file}' or default location."
        
        # NavigationCore handles its own yolo_model_file existence check

        print("Initializing NavigationCore...")
        nav_system = NavigationCore(
            map_filepath=map_file,
            start_node_id=str(start_node_id), # Ensure node IDs are strings
            end_node_id=str(destination_node_id),
            yolo_model_path=yolo_model_file
        )
        print("NavigationCore initialized. Starting navigation loop (this will block the agent's current turn)...")
        
        # run_navigation_loop is blocking and handles its own TTS and camera interaction.
        success = nav_system.run_navigation_loop() 
        
        if success:
            return f"Navigation session to {destination_node_id} has been completed."
        else:
            return f"Navigation session to {destination_node_id} was interrupted or encountered an error."

    except ValueError as ve: # Catches pathfinding/setup errors from NavigationCore
        return f"Navigation Setup Error: {str(ve)}"
    except FileNotFoundError as fnf:
        return f"Navigation File Error: {str(fnf)}"
    except Exception as e:
        print(f"Critical Error in start_indoor_navigation_guidance tool: {type(e).__name__} - {e}")
        return f"Failed to start or complete navigation due to an unexpected error: {str(e)}"

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- LLM and Tool Setup for LangGraph ---
llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    model=LM_STUDIO_MODEL_IDENTIFIER,
    temperature=0.1, # Lower for more predictable tool use
)

tools_list = [describe_current_scene, read_text_from_image, start_indoor_navigation_guidance]
llm_with_tools = llm.bind_tools(tools_list)

# --- Agent Logic Nodes for LangGraph ---
def agent_node(state: AgentState):
    """Invokes the LLM to get the next action or response based on the current state."""
    print("\n--- Agent Node --- Calling LLM...")
    # print(f"Current state messages: {state['messages']}")
    response = llm_with_tools.invoke(state["messages"])
    # print(f"LLM Response: {response}")
    return {"messages": [response]} # Append LLM's response (AIMessage with content or tool_calls)

# ToolNode handles the execution of the LLM-chosen tool
tool_node = ToolNode(tools_list)

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent") # Start with the agent node

# Conditional edge: after agent node, check if LLM called a tool or responded directly
workflow.add_conditional_edges(
    "agent",
    tools_condition, # Built-in function to check for tool calls
    {
        "tools": "tools", # If tool_calls exist, go to tools node
        END: END        # Otherwise (no tool_calls, direct AIMessage), end the turn
    }
)
workflow.add_edge("tools", "agent") # After tool execution, go back to agent node for next step

# Compile the graph
agent_graph = workflow.compile()

# --- Main Interaction Function ---
def run_interactive_agent():
    print("\n🚀 Welcome to the Interactive Assistant Agent! 🚀")
    print("Say 'quit' to exit.")
    print(f"Using LLM: {LM_STUDIO_MODEL_IDENTIFIER} via {LM_STUDIO_BASE_URL}")
    print("Make sure LM Studio server is running and the model is loaded.")
    print("Ensure map files and YOLO models are in their expected locations (e.g., Nav/map_data/, Nav/)")
    print("Example commands:")
    print("  - describe the scene")
    print("  - read text from image")
    print("  - read text from image data/my_sign.jpg")
    print("  - navigate to node5")
    print("  - guide me to the reception desk from the main entrance")
    
    config = {"recursion_limit": 150} # Standard LangGraph config

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "quit":
            print("Exiting agent...")
            break
        if not user_input.strip():
            continue

        try:
            print("--- Invoking Agent Graph ---")
            # Stream events from the graph execution
            events = agent_graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            
            final_ai_response_content = ""
            for event in events:
                # You can print event names to see the flow: print(f"Event type: {list(event.keys())[0]}")
                if "agent" in event:
                    agent_messages = event["agent"].get("messages", [])
                    if agent_messages:
                        last_msg = agent_messages[-1]
                        if isinstance(last_msg, AIMessage):
                            if last_msg.content:
                                final_ai_response_content = last_msg.content # Capture direct AI response
                                print(f"\nAI: {last_msg.content}")
                                # Decide if direct AI responses should be spoken
                                # if shared_tts_pipeline_global and not last_msg.tool_calls:
                                # shared_tts_pipeline_global.speak(last_msg.content)
                            if last_msg.tool_calls:
                                print(f"AI is requesting to use tool(s):")
                                for tc in last_msg.tool_calls:
                                    print(f"  - Tool: {tc['name']}, Args: {tc['args']}")
                elif "tools" in event:
                    tool_outputs = event["tools"].get("messages", [])
                    if tool_outputs:
                        for tool_msg in tool_outputs:
                            if isinstance(tool_msg, ToolMessage):
                                print(f"\nTool Output ({tool_msg.name if hasattr(tool_msg, 'name') else 'unknown_tool'}):\n{tool_msg.content}")
            
            # If the graph ended without a direct AI content response (e.g. after a tool call)
            # the conversation continues in the next loop based on the full message history.
            # If the final step was a direct AI message, it's already printed.

        except Exception as e:
            print(f"An error occurred during agent execution: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_interactive_agent()
