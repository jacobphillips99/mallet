# Modal Servers
See the main [README](https://github.com/jacobphillips99/mallet/blob/main/README.md) for more information on how to deploy the Modal apps.

 ## Prerequisites

 - [Modal](https://modal.com/docs/guide/getting-started)
 - [MALLET](https://github.com/jacobphillips99/mallet/blob/main/README.md)

 Using MALLET Modal VLM Servers requires adding any API keys to the Modal secrets manager. Go to the Modal Secrets dashboard at `https://modal.com/secrets/your-name-here/main` and add the necessary secrets for your model provider. The default VLM modal server provisions API keys for OpenAI, Anthropic, Gemini, and XAI.

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

 This will create a Modal web function along a URL; it should look like 

 ```bash
 Created web function fastapi_app => https://your-name-here.modal.run
 ```

 Use this link as the `host` and port `-1`.

 To expose the service with a port via Modal tunnels:
 ```bash
 modal run modal_servers/vla/vla_modal_server_with_tunnel.py
 ```
 After running the deployment, you can find the host and port in the logs; it should look something like `xxx.modal.host:yyyyy`. Note that the tunnel approach creates ONE Modal endpoint that must be manually closed (or allow the timeout to wind down), as opposed to the typical Modal deployment that autoscales on its own.

 ### Deploying the VLM Server

 ```bash
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
