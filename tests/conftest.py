import os

# keep TensorFlow quiet and CPU-only so the suite runs anywhere
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
sys.path.insert(0, os.path.dirname(__file__))   # tests/synthetic.py
