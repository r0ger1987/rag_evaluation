#!/usr/bin/env python3
"""
Main entry point for SmartRAG evaluation with Ragas
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from evaluation.smartrag_evaluator import main

if __name__ == "__main__":
    main()