# VLM AutoEval Robot Benchmark

API-driven VLM server to tunnel requests/responses to the real-world robot evaluation project [AutoEval](https://github.com/AutoEval/AutoEval).

## Overview

This package creates a server that accepts requests from AutoEval, sends them to a Vision Language Model (VLM) API, and then sends the responses back to AutoEval. The VLM responses (high-level natural language commands) are translated into robot actions.

## Features

- Support for multiple VLM providers (OpenAI, Anthropic, Google, etc.)
- Rate limiting for API calls with configurable limits per provider/model
- Asynchronous parallel requests for efficient handling of multiple VLM calls
- Translation of natural language commands to robot actions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vlm-autoeval-robot-benchmark.git
cd vlm-autoeval-robot-benchmark

# Install the package
pip install -e .
```

## Configuration

Create a `.env` file based on the `.env.example` template:

```bash
cp .env.example .env
```

Edit the `.env` file to set your API keys and other configuration options.

## Usage

### Starting the Server

```bash
python -m vlm_autoeval_robot_benchmark --host 0.0.0.0 --port 8000
```

Or using the installed package:

```bash
vlm-autoeval-server --host 0.0.0.0 --port 8000
```

### API Endpoints

- `GET /`: Root endpoint, returns basic server information
- `POST /health`: Health check endpoint
- `POST /command`: Main endpoint to generate robot commands from observations
- `GET /stats`: Get statistics about the server and rate limits

### Example Request

```python
import requests
import base64
import json

# Load an image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Prepare the request
data = {
    "observation": {
        "image": image_data,
        "proprio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    },
    "task_description": "Pick up the red block and place it in the blue box.",
    "model": "gpt-4-vision-preview"
}

# Send the request
response = requests.post("http://localhost:8000/command", json=data)
result = response.json()

# Print the result
print(json.dumps(result, indent=2))
```

## Command Translation

The server translates VLM natural language commands into robot actions using the following mapping:

- "move forward/backward" → x-axis (action[0])
- "move left/right" → y-axis (action[1])
- "move up/down" → z-axis (action[2])
- "tilt up/down" → roll/x-axis rotation (action[3])
- "rotate clockwise/counterclockwise" → yaw/z-axis rotation (action[5])
- "open/close gripper" → gripper (action[6])

The action vector format is [x, y, z, roll, pitch, yaw, gripper], with values typically in the range [-1.0, 1.0].

## Development

### Project Structure

```
vlm_autoeval_robot_benchmark/
├── __init__.py             # Package initialization
├── __main__.py             # CLI entry point
├── server.py               # FastAPI server implementation
├── models/                 # VLM models
│   ├── __init__.py
│   └── vlm.py              # VLM implementation
└── utils/                  # Utility functions
    ├── __init__.py
    └── rate_limit.py       # Rate limiting implementation
```

### Adding New VLM Providers

To add support for a new VLM provider, update the following:

1. Add API key environment variable to `.env.example`
2. Add provider detection in `VLM._get_provider_from_model()` method
3. Add default rate limits in `VLM.DEFAULT_RATE_LIMITS` if needed

## License

MIT License 