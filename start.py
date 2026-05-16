#!/usr/bin/env python3
"""Quick start script for DS2API Browser.

Accounts and keys are loaded from .env file automatically by main.py.
See .env.example for the expected format.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from main import main

if __name__ == "__main__":
    main()
