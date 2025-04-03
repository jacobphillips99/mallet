#!/usr/bin/env python3
"""Example client for VLM AutoEval Robot Benchmark."""

import argparse
import base64
import json
import os
import requests
from pathlib import Path
import sys


def main():
    """Run the example client."""
    parser = argparse.ArgumentParser(description="Example client for VLM AutoEval Robot Benchmark")
    parser.add_argument("--server", type=str, default="http://localhost:8000", 
                      help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--image", type=str, required=True,
                      help="Path to image file to send")
    parser.add_argument("--task", type=str, default="Pick up the red block and place it in the blue box.",
                      help="Task description")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview",
                      help="VLM model to use")
    
    args = parser.parse_args()
    
    # Check if image file exists
    image_path = Path(args.image)
    if not image_path.exists() or not image_path.is_file():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Load the image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Prepare the request
    data = {
        "observation": {
            "image": image_data,
            "proprio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Example proprioceptive state
        },
        "task_description": args.task,
        "model": args.model
    }
    
    # Send the request
    try:
        print(f"Sending request to {args.server}/command...")
        response = requests.post(f"{args.server}/command", json=data)
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        result = response.json()
        
        # Print the response
        print("\nServer Response:")
        print(f"Actions: {result['actions']}")
        print(f"Success: {result['success']}")
        
        if result.get('info', {}).get('vlm_response'):
            print("\nVLM Response:")
            print(result['info']['vlm_response'])
        
        # Print model info
        if result.get('info', {}).get('model'):
            print(f"\nModel: {result['info']['model']}")
            print(f"Provider: {result['info']['provider']}")
            print(f"Response time: {result['info']['response_ms']} ms")
        
        # Print token usage if available
        if result.get('info', {}).get('usage'):
            print("\nToken Usage:")
            for key, value in result['info']['usage'].items():
                print(f"  {key}: {value}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 