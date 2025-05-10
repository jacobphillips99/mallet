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
