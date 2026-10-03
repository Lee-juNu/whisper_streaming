"""Isolated, reproducible ASR comparison harness.

Core modules (metrics, dataset, guards, mock, scoring, CLI) use only the Python 3
standard library. Optional engine packages are imported lazily inside
``asrbench.adapters.FasterWhisperEngine`` and nowhere else.
"""
from pathlib import Path

__version__ = "0.1.0"
ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
TEST_HOST = "127.0.0.1"
TEST_PORT = 18101
