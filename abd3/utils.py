"""
ABD3 Utilities.
"""

import torch
import pandas as pd
import os


def print_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")


def update_and_save_csv(save_dict, csv_path):
    """Save results to CSV, appending if file exists."""
    df = pd.DataFrame(save_dict)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
