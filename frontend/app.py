from flask import Flask, request, send_file
import os
import tempfile
import sys
import io
from werkzeug.utils import secure_filename

# Import LangChain components for agent integration
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Configure CORS to allow requests from the frontend
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

# Mock tools similar to those in the agent.ipynb
@tool
def indoor_navigation(start_location, destination):
    """Navigate from one indoor location to another."""
    # This is a simplified mock of the indoor navigation tool
    return f"To get from {start_location} to {destination}, go straight ahead for 10 meters, then turn right."

@tool
def general_navigation(start_location, destination):
    """Navigate from one general location to another using GPS."""
    # This is a simplified mock of the general navigation tool
    return f"The route from {start_location} to {destination} is approximately 5 miles. Head north on Main Street."

@tool
def scene_description(image_path=None):
    """Describe the scene in an image or the current view."""
    # This is a simplified mock of the scene description tool
    descriptions = [
        "I see a living room with a couch, coffee table, and TV.",
        "This appears to be a kitchen with modern appliances.",
        "I can see an outdoor park with trees and a playground."
    ]
    import random
    return random.choice(descriptions)

# Setup the agent with tools
tools = [indoor_navigation, general_navigation, scene_description]

# Initialize the LLM (in a real scenario, this would connect to your actual LLM)
def get_llm():
    # Mock LLM for demonstration purposes
    # In a real scenario, this would be your actual LLM connection
    class MockLLM:
        def invoke(self, prompt):
            # Simple response generation based on keywords in the prompt
            if "where" in prompt.lower() or "how to get" in prompt.lower():
                return "I'll help you navigate. Please specify your starting point and destination."
            elif "what" in prompt.lower() or "describe" in prompt.lower():
                return "I can see your surroundings. There appears to be a room with furniture and some windows."
            else:
                return "I'm your assistant. How can I help you today?"
    
    return MockLLM()

# Setup the agent
def setup_agent():
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant that can use tools to help users. {input}"
    )
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Process audio and get agent response
def process_audio_with_agent(audio_file_path):
    """
    Process the audio file with speech-to-text, 
    send the text to the agent, and convert the response to speech.
    
    In a real implementation, this would:
    1. Use a speech-to-text service to convert audio to text
    2. Send the text to the agent for processing
    3. Convert the agent's text response back to audio
    """
    # Mock implementation for demonstration
    
    # Mock speech-to-text (would use a real STT service in production)
    # Pretend we extracted "Where is the nearest exit?" from the audio
    transcribed_text = "Where is the nearest exit?"
    
    # Process with agent
    agent = setup_agent()
    agent_response = agent.invoke({"input": transcribed_text})
    response_text = agent_response.get("output", "I'm processing your request.")
    
    # Mock text-to-speech (would use a real TTS service in production)
    # In a real implementation, this would convert the text to an audio file
    
    # For demonstration, we'll create a simple audio file with a message
    from scipy.io import wavfile
    import numpy as np
    
    # Generate a simple tone as a placeholder for the TTS output
    sample_rate = 22050
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a simple sine wave as a placeholder
    tone = np.sin(2 * np.pi * 440 * t) * 0.3
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    wavfile.write(temp_file.name, sample_rate, tone.astype(np.float32))
    
    return temp_file.name, response_text

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return {"error": "No audio file provided"}, 400
    
    audio_file = request.files['audio']
    
    # Save the uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, secure_filename(audio_file.filename))
    audio_file.save(temp_path)
    
    try:
        # Process the audio with the agent
        response_audio_path, response_text = process_audio_with_agent(temp_path)
        
        # Log the response for debugging
        app.logger.info(f"Agent response: {response_text}")
        
        # Return the audio file
        return send_file(
            response_audio_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='response.wav'
        )
    
    except Exception as e:
        app.logger.error(f"Error processing audio: {str(e)}")
        return {"error": str(e)}, 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Install required packages if not already installed
    try:
        import scipy
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    
    app.run(host='0.0.0.0', port=5000, debug=True)
