#!/usr/bin/env python3
"""
Command Line Interface for LLM Libration package.

This module serves as the main CLI entry point, importing from the refactored
CLI modules for better organization and maintainability.
"""

from .cli import main

if __name__ == '__main__':
    main()
