#!/usr/bin/env python3
"""
Godelion Main Entry Point

Compatibility alias. Use `python run.py` instead.
"""
import sys
import warnings

warnings.warn(
    "DGM_outer.py is deprecated. Use `python run.py` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and run the main function from run.py
from run import main

if __name__ == "__main__":
    sys.exit(main())
