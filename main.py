#!/usr/bin/env python3
"""
DingDong RAG Entry Point
Main entry point for the DingDong RAG CLI application.
"""

import sys
from pathlib import Path

# Load environment variables from .env file first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env file will not be loaded.")
    print("Install with: uv add python-dotenv")

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main CLI
from dingdong_rag.cli.main import main

if __name__ == "__main__":
    main()