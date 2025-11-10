"""
helpers.py
- 一些小工具函数（占位）
"""
import os
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        import pandas as pd
        return pd.read_csv(path)
    return None
