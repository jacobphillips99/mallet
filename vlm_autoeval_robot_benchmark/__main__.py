"""Command-line interface for VLM AutoEval Robot Benchmark."""

import argparse
import logging
import os

from .server import run_server

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="VLM AutoEval Robot Benchmark Server"
    )
    
    # Server configuration
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing configuration files"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    print(f"Starting VLM AutoEval Robot Benchmark server at http://{args.host}:{args.port}")
    run_server(host=args.host, port=args.port, log_level=args.log_level)

if __name__ == "__main__":
    main() 