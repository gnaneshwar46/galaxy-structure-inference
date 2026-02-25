"""
split_data.py

Responsible for:
- Loading cleaned dataset
- Creating structural class label
- Performing stratified train/test split
- Saving reproducible datasets
"""

import os
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_structural_label(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Create binary structural label:
    0 = disk-dominated (n < threshold)
    1 = bulge-dominated (n >= threshold)
    """
    df["structural_class"] = (df["sersic_n"] >= threshold).astype(int)
    return df


def run_split(config_path: str):
    config = load_config(config_path)

    input_path = config["paths"]["clean_data"]
    df = pd.read_csv(input_path)

    logging.info(f"Loaded cleaned dataset: {df.shape}")

    threshold = config["classification"]["sersic_threshold"]
    df = create_structural_label(df, threshold)

    logging.info("Structural class label created.")

    X = df.drop(columns=["structural_class", "sersic_n"])
    y = df["structural_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y
    )

    logging.info(f"Train shape: {X_train.shape}")
    logging.info(f"Test shape: {X_test.shape}")

    output_dir = "data/splits"
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df["structural_class"] = y_train

    test_df = X_test.copy()
    test_df["structural_class"] = y_test

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    logging.info("Saved stratified train/test splits.")


if __name__ == "__main__":
    run_split("config.yaml")