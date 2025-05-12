# MALLET: Multimodal Autonomy Language and Long-Context Evaluation Toolkit
Created by [Jacob Phillips](https://jacobdphillips.com/)

<img src="assets/mallet_system_diagram.png" alt="MALLET System Diagram"/>

MALLET is an open-source (Apache 2.0) toolkit for controlling real-world robots with cloud-hosted VLMs, as well as a suite of tools for evaluating the capabilities of VLMs in long-context multimodal settings. MALLET is built on top of [Paul Zhou's](https://github.com/zhouzypaul) [AutoEval](https://github.com/zhouzypaul/auto_eval) and [mse-check](https://github.com/zhouzypaul/mse-check), which allows us to submit action commands to real-world robots and evaluate offline policies.

MALLET provides a toolkit for researchers to conduct GPU-heavy real-world robotics research *without* having to purchase robots or GPUs! We build several capabilities for researchers to build and evaluate multimodal agents that can operate in the real world.

MALLET makes two large contributions to the robotics community:
1. MALLET presents a framework on top of [Embodied Chain of Thought (ECoT)](https://github.com/MichalZawalski/embodied-CoT) for translating between natural language and continuous, 7-DoF robot actions, which enables VLMs to directly control real-world robots.
2. MALLET enables robot researchers to use cloud-based, autoscaling GPU compute frameworks like [Modal](https://modal.com/) to serve VLM or VLA-based policies instead of exposing their own computers.

We use MALLET to evaluate the performance of VLMs on controlling real-world robots in `AutoEval`. We also use MALLET with `mse-check` to evaluate the performance of VLMs in long-context multimodal settings, ablating prompts, history selection, and inference time-cost tradeoffs.

## Overview

The MALLET repository is organized into three main directories:
- `mallet`: the pip-installable toolkit for controlling real-world robots with VLMs, including cloud-based policy servers, ECoT translation, and AutoEval integration.
- `modal_servers`: a directory of pre-built Modal wrappers for CPU-based VLM servers and GPU-based VLA servers.
- `mse-check`: a fork of [`mse-check`](https://github.com/zhouzypaul/mse-check) with greatly expanded capabilities for serving, evaluating, visualizing, and ablating multimodal policies with real-world data.

## Installation

MALLET requires Python 3.10 or higher. We recommend using `uv` for fast and reliable dependency management.

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and its submodules:
```bash
git clone --recursive https://github.com/jacobphillips99/mallet.git
cd mallet
```

3. Create and activate a virtual environment with Python 3.10:
```bash
uv venv .venv --python=python3.10
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

4. Install the package and its dependencies:
```bash
uv pip install -r requirements.txt
uv pip install -e .
```

## MALLET Toolkit
MALLET (Multimodal Autonomy Language and Long-Context Evaluation Toolkit) is a Python library that lets you control real-world robots using large Vision-Language Models (VLMs) and evaluate their performance. It provides a suite of tools for bridging natural language and continuous 7-DoF robot actions, built on the [Embodied Chain-of-Thought (ECoT)](https://github.com/MichalZawalski/embodied-CoT) paradigm. MALLET includes modules for language-to-action backtranslation, `AutoEval` compliant VLM and VLA servers, rate limiting and monitoring, and more.

### Language-to-Action Backtranslation
The ECoT project develops a [set of primitives for translating robot actions into natural language](https://github.com/MichalZawalski/embodied-CoT/blob/main/scripts/generate_embodied_data/primitive_movements.py). The ECoT project goes on to develop synthetic, grounded chain-of-thought reasoning traces for model training. Instead, we are interested in the reverse direction: given a natural language description of an action, how can we translate it into a sequence of robot actions? We [invert the ECoT primitive movements](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/utils/ecot_primitives/inverse_ecot_primitive_movements.py) to develop a language-to-action backtranslation framework for MALLET. We use VLMs to predict the direction, magnitude, and explanation for each degree of freedom in the robot's action space.

```bash
ECOT([0.7, -0.1, ..., 1.0]) --> "Move left, down, open gripper"

MALLET("Left 90% in order to align with the object, Down 10% to grasp object, Open Gripper 100% to prepare for grasp") --> [0.7, -0.1, ..., 1.0]
```

In order to translate a direction and magnitude into a robot action, we first calculate the [action bounds](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/utils/ecot_primitives/action_bounds.json) for the given robot. We select the action bounds based on the collected dataset, setting the bounds to the 10th and 90th percentiles of the data. We then scale the direction and magnitude to the appropriate range, and convert the normalized action into a robot action. This makes it easy for VLMs to predict actions that are sensible for the robot's action space. Additionally, we provide a set of prompts describing the robot, the methodology, the action space, the environment, and the output format. These components are extremely modular and enable simple integration with the rest of the MALLEt toolkit; all prompts and a special `PromptBuilder` class are availabe in [`mallet.models.translation.py`](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/models/translation.py).

### VLM and VLA Servers

#### VLM Class
We present AutoEval-compliant VLM and VLA FastAPI servers that can be used to serve and evaluate multimodal and robot policies. The [core VLM class](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/models/vlm.py) wraps `litellm` to provide a simple interface for calling any LLM. We build a [composable Pydantic-based framework](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/models/vlm.py#L106) for building multimodal requests with a focus on extensibility and modularity. The VLM class can generate asynchronous responses, build prompts, [store historical requests and responses](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/models/vlm.py#L106), and parse responses into a structured format. Here's a simple example of using the VLM class to build a multimodal request and return a robot action:

```python
import asyncio
from mallet.models import VLM, VLMInput, VLMRequest, parse_vlm_response
from mallet.utils.ecot_primitives.inverse_ecot_primitive_movements import text_to_move_vector

model = "gpt-4o-mini"

# create the VLM request
prompt_text = "Open the drawer"
image_bytes = ... # load image from disk or camera
vlm_input = VLMInput(prompt=prompt_text, images=[ImageInput(data=image_bytes)])
request = VLMRequest(vlm_input=vlm_input, model=model)

# instantiate the VLM class
vlm = VLM(model=model)
response = asyncio.run(vlm.generate(request))

# parse the response into a structured format
description, structured_answer = parse_vlm_response(response.text)

# convert the structured answer into a robot action
robot_action = text_to_move_vector(structured_answer)
```

#### Native FastAPI Servers
The [`mallet.servers.server.py`](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/servers/server.py) module contains a generic FastAPI server we call the VLMPolicyServer. This server is a drop-in replacement for the AutoEval `PolicyServer` and can be used to serve any multimodal policy by wrapping a VLM object with a FastAPI endpoint and serving it with Uvicorn. This allows us to serve multimodal policies locally or in the cloud. The VLMPolicyServer also enables logging, history, and serves as an interface between the AutoEval and mse-check frameworks and MALLET. For example, the VLMPolicyServer exposes the `/act` endpoint, which is hit by the AutoEval server with a payload (including the image and instruction) and returns the 7-DoF robot action.

```python
from mallet.servers.server import VLMPolicyServer

server = VLMPolicyServer(model="gpt-4o-mini", history_length=10, history_choice="all")
server.run(host="0.0.0.0", port=8000)
```

This spins up a FastAPI server on the localhost that can be hit with a payload of an image and instruction. Upon hitting the `/act` endpoint, the VLM object sends a remote API request to the model (in this case, `gpt-4o-mini`) and then translates the natural language into a continous robot action.

We also provide compatible forks of servers for OpenVLA (from [AutoEval](https://github.com/zhouzypaul/auto_eval/blob/main/auto_eval/policy_server/openvla_server.py) and [the original authors](https://github.com/openvla/openvla/blob/main/vla-scripts/deploy.py)) and [ECoT](https://colab.research.google.com/drive/1CzRKin3T9dl-4HYBVtuULrIskpVNHoAH?usp=sharing#scrollTo=owVajjweDopA).



### Rate Limiting and Monitoring
To safely scale and handle API usage, MALLET includes a real-time outbound request rate-limiter. Users can define per-provider and per-model limits (such as requests/min, tokens/min, and number of concurrent requests) to prevent unexpected rate-limiting by model providers. The [`rate_limits.yaml`](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/config/rate_limits.yaml) contains the configuration for the rate limiter, which is loaded into the singular [`RateLimit`](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/models/rate_limit.py)object. The RateLimit object is passed to each VLM object in order to limit the rate of outbound requests. The RateLimit object tracks usage statistics and even provides a CLI tool to view rate limit activity in real-time. See [rate_limit_cli.py](https://github.com/jacobphillips99/mallet/blob/main/src/mallet/models/rate_limit_cli.py) for the `curses` based UI tool showing each mode's request rate and token consumption live.

## Modal Servers


## mse-check

## Evaluation

## Acknowledgements and Citation
This project was developed by [Jacob Phillips](https://jacobdphillips.com) as a part of the [Andreessen Horowitz American Dynamism Engineering Fellows program](https://a16z.com/the-american-dynamism-engineering-fellows-program/). Special thanks to the American Dynamism team for their support and feedback on the project.

If using the MALLET toolkit in your work, please cite it to acknowledge the authors. Suggested format:

```bibtex
@software{MALLET,
    title = {MALLET: Multimodal Autonomy Language and Long-Context Evaluation Toolkit},
    author = {Jacob Phillips},
    url = {https://github.com/jacobphillips99/mallet},
    version = {insert version number},
    date = {insert date of usage},
    year = {2025},
}
