 # VLM AutoEval Robot Benchmark

 API-driven servers for robot action policy prediction using Vision-Language Models (VLM) and Vision-Language Action (VLA) models, integrated with the AutoEval benchmarking framework.

 ## Repository Structure

- `vlm_autoeval_robot_benchmark/` — Core Python package implementing:
  - `VLMPolicyServer`: FastAPI server for VLM-based robot commands.
  - `OpenVLAServer`: FastAPI server for OpenVLA-based vision-to-action predictions.
  - Models, utilities, rate limiting, and translation modules.
  - See `vlm_autoeval_robot_benchmark/README.md` for package details.
- `vlm_modal_server.py` — Modal deployment script for the VLM policy server.
- `vla_modal_server.py` — Modal deployment script for the OpenVLA policy server.
- `examples/` — Example client scripts.
- `assets/` — Sample goal images and trajectory data for primitive move testing.
- Jupyter notebooks (`bridge_explore.ipynb`, `mse_ecot.ipynb`) for data exploration.
- Test and utility scripts (`primitive_moves_tester.py`, `vlm_tester.py`, `test_utils.py`, `test_server_via_client.py`).
- `setup.py`, `pyproject.toml`, `requirements.txt` — Packaging and dependencies.

 ## Features

- Integration with multiple VLM providers (OpenAI, Anthropic, Gemini, HuggingFace) via `litellm`.
- OpenVLA-based direct vision-to-action prediction.
- Configurable rate limiting per provider/model (`rate_limits.yaml`).
- Translation of natural language commands into 7-DOF robot action vectors.
- REST API with FastAPI and Uvicorn (async endpoints).
- Modal deployment support for serverless hosting.

 ## Installation

 1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/vlm-autoeval-robot-benchmark.git
    cd vlm-autoeval-robot-benchmark
    ```
 2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
 3. (Optional) Install the package in editable mode:
    ```bash
    pip install -e .
    ```

 ## Configuration

 Configuration and defaults are documented in `vlm_autoeval_robot_benchmark/config/README-config.md`.

 ### Environment Variables

 Set API keys for VLM providers:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `HUGGINGFACE_API_KEY` (for HuggingFace integration)

 ### Rate Limits

 Define rate limits in a `rate_limits.yaml` file. The system searches in:
 1. Current working directory
 2. `~/.vlm_autoeval/`
 3. Package config directory (`vlm_autoeval_robot_benchmark/config`)

 ## Usage

 ### VLM-Based Server

 Run locally:
 ```bash
 python -m vlm_autoeval_robot_benchmark.servers.server --host 0.0.0.0 --port 8000 --model gpt-4o-mini
 ```

 **Endpoints**:
 - `GET /` — Server info
 - `GET /health` — Health check
 - `POST /act` — Predict an action
   - Request JSON:
     ```json
     {
       "image": <numpy array or base64 string>,
       "instruction": "Your instruction",
       "proprio": [optional array of length 7],
       "history": [optional history]
     }
     ```
   - Response: JSON array of 7 action values.

 ### VLA-Based Server (OpenVLA)

 Deploy locally via Modal:
 ```bash
 modal serve vla_modal_server.py
 ```
 Or deploy to Modal Cloud:
 ```bash
 modal deploy vla_modal_server.py
 ```

 **Endpoints**:
 - `GET /health` — Health check
 - `POST /act` — Predict an action
   - Request JSON:
     ```json
     {
       "image": <numpy array>,
       "instruction": "Your instruction",
       "unnorm_key": "optional unnormalization key"
     }
     ```
   - Response: JSON array of action values.

 ### Example Client

 Send a request using the example client script:
 ```bash
 python examples/example_client.py \
   --server http://localhost:8000 \
   --image path/to/image.png \
   --task "Pick up the red block and place it in the blue box." \
   --model gpt-4-vision-preview
 ```

 ### Testing

 Run all tests with pytest:
 ```bash
 pytest
 ```
 Or use provided scripts:
 ```bash
 python primitive_moves_tester.py
 python vlm_tester.py
 python test_server_via_client.py
 ```

 ## Notebooks

 - `bridge_explore.ipynb`: Explore ECOT primitive action trajectories.
 - `mse_ecot.ipynb`: Compute mean squared error for ECOT primitives.

 ## Contributing

 Contributions are welcome! Please open issues or submit pull requests.
