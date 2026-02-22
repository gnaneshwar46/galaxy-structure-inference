"""
data_cleaning.py

Responsible for:
- Loading raw data
- Basic sanity filtering
- Missing value handling
- Log-scaling
- Saving cleaned dataset

This module must not:
- Perform model training
- Perform train/test splitting
- Contain hard-coded thresholds
"""
import os
import logging
import yaml
import pandas as pd
import numpy as np

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"        
)

# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset."""
    logging.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)

def basic_filtering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply basic physical cuts.
    Example: remove negative masses or extreme redshift.
    """
    data_cfg = config.get("data", {})

    if "redshift_max" in data_cfg:
        df = df[df["redshift"] <= data_cfg["redshift_max"]]

    if "sersic_min" in data_cfg:
        df = df[df["sersic_n"] >= data_cfg["sersic_min"]]
    
    if "sersic_max" in data_cfg:
        df = df[df["sersic_n"] <= delattr["sersic_max"]]

    logging.info("Applied astrophysical fitlers.")
    return df

def handle_missing_values(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    strategy = config.get("preprocessing", {}).get("missing_strategy", "drop")

    if strategy == "drop":
        df = df.dropna()
        logging.info("Dropped missing values.")
    elif strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
        logging.info("Filled missing values with median.")
    else:
        raise ValueError("Invalid missing strategy.")

    return df


def log_transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply log10 transformation to specified columns.
    """
    if config.get("preprocessing", {}).get("log_transform", False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(
            lambda x: np.log10(x) if (x > 0).all() else x
        )
        logging.info("Applied log10 transform to numeric columns.")

    return df

def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """Save cleaned dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    df.to_csv(output_path, index = False)
    logging.info(f"Saved cleaned data to {output_path}")

# ------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------

def run_data_cleaning(config_path: str):
    """
    Full cleaning pipeline
    """
    config = load_config(config_path)

    df = load_data(config["paths"]["raw_data"])
    logging.info(f"Initial shape: {df.shape}")

    df = basic_filtering(df,config)
    df = handle_missing_values(df, config)
    df = log_transform(df, config)

    logging.info(f"Final shape: {df.shape}")

    save_cleaned_data(df, config["paths"]["clean_data"])

if __name__ == "__main__":
    run_data_cleaning("config.yaml")
