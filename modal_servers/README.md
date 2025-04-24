# Modal Servers

 This directory contains Modal deployments for two types of robot policy servers using FastAPI:

 - **VLA (Vision-Language-Action)** via OpenVLA model
 - **VLM (Vision Language Model)** via VLMPolicyServer

 ## Directory Structure

 modal_servers/
 ├── vla/
 │   ├── __init__.py                   # VLA server defaults and image/volume setup
 │   ├── vla_modal_server.py           # VLA FastAPI ASGI app for Modal deployment
 │   └── vla_modal_server_with_tunnel.py # VLA server with Modal tunnels for ports
 └── vlm/
     ├── __init__.py                   # VLM server defaults, image, and secrets setup
     ├── vlm_modal_server.py           # VLM FastAPI ASGI app for Modal deployment
     └── vlm_modal_server_with_tunnel.py # VLM server with Modal tunnels for ports

 ## Prerequisites
 TODO MODAL SETUP + SECRETS

 ## Configuration

 You can customize defaults in the `__init__.py` files:

 **VLA (`vla/__init__.py`):**
 - `DEFAULT_APP_NAME`: Modal App name
 - `DEFAULT_OPENVLA_PATH`: HuggingFace path to OpenVLA model
 - `DEFAULT_CONCURRENCY`, `DEFAULT_GPU`, `DEFAULT_TIMEOUT`

 **VLM (`vlm/__init__.py`):**
 - `DEFAULT_MODEL`, `DEFAULT_CONCURRENCY`, `DEFAULT_TIMEOUT`

 ## Usage

 ### Deploying the VLA Server as a Modal App
 ```bash
 modal deploy modal_servers/vla/vla_modal_server.py
 ```

 To expose the service with a port via Modal tunnels:
 ```bash
 modal run modal_servers/vla/vla_modal_server_with_tunnel.py
 ```
 After running the deployment, you can find the host and port in the logs; it should look something like `xxx.modal.host:yyyyy`. Note that the tunnel approach creates ONE Modal endpoint that must be manually closed (or allow the timeout to wind down), as opposed to the typical Modal deployment that autoscales on its own.

 ### Deploying the VLM Server

 ```bash
 # (Optional) Set a custom model
MODEL="your-vlm-model-name" modal deploy modal_servers/vlm/vlm_modal_server.py
 ```

 To expose the VLM service via a forwarded port:
 ```bash
 modal run modal_servers/vlm/vlm_modal_server_with_tunnel.py
 ```
 After running the deployment, you can find the host and port in the logs; it should look something like `xxx.modal.host:yyyyy`. Note that the tunnel approach creates ONE Modal endpoint that must be manually closed (or allow the timeout to wind down), as opposed to the typical Modal deployment that autoscales on its own.

 ## API Endpoints

 Both sets of servers expose a FastAPI app with:

 - `GET /health`
   Health check returning service status, model, and timestamp.
 - `POST /act`
   Predict an action for a given image and instruction.
