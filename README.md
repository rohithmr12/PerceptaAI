# MCP Server for Percepta AI

This is a Multi-Client Protocol (MCP) server implementation that provides access to the tools from the Percepta AI agent.

## Features

- Socket-based server supporting multiple concurrent clients
- Access to four key tools:
  - `indoor_navigation`: Provides directions based on indoor floor plans
  - `general_navigation`: Provides outdoor navigation guidance
  - `scene_description`: Describes surroundings based on object detection
  - `ocr_text`: Extracts text from documents using OCR
- Integration with LLM Studio for natural language processing
- Text-to-speech capabilities using Kokoro TTS

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd PerceptaAI
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Ensure you have LM Studio running locally (for LLM functionality)
   - Download from [https://lmstudio.ai/](https://lmstudio.ai/)
   - Start the server on port 1234
   - Load the Llama-3.2-3B-Instruct model

## Usage

### Starting the Server

Run the server with:

```
python mcp_server.py
```

The server will start on port 5000 by default and accept connections from any interface.

### Using the Client

Run the test client with:

```
python mcp_client.py
```

The client provides a menu-based interface to test all available tools:

1. Indoor Navigation
2. General Navigation
3. Scene Description
4. OCR Text Recognition
5. Ask LLM a question
6. Text-to-speech

## API Documentation

Clients can communicate with the server using JSON messages over TCP sockets. 

### Function Call Request

To directly call a tool function:

```json
{
  "type": "function_call",
  "function": "scene_description",
  "arguments": {
    "image_path": ""
  }
}
```

### LLM Query Request

To process a natural language query through the LLM:

```json
{
  "type": "llm_query",
  "query": "What's around me?"
}
```

### Text-to-Speech Request

To convert text to speech:

```json
{
  "type": "tts",
  "text": "Hello, how can I help you?"
}
```

## Extending the Server

To add new tools, modify the `TOOLS` dictionary in `mcp_server.py` and implement the corresponding function.

## License

This project is licensed under the MIT License - see the LICENSE file for details.