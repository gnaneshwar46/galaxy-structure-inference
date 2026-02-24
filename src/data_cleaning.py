"""
data_cleaning.py

Loads NSA FITS file and prepares cleaned dataset for
controlled structural classification study.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
from astropy.io import fits


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_fits_data(fits_path: str) -> pd.DataFrame:
    logging.info(f"Loading FITS file from {fits_path}")

    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[1].data

        # Convert FITS_rec to numpy structured array
        data = np.array(data)

        # Fix endian using dtype trick (NumPy 2.0 safe)
        data = data.astype(data.dtype.newbyteorder('='))

    df = pd.DataFrame({
        "redshift": data["Z"],
        "sersic_n": data["SERSIC_N"],
        "stellar_mass": data["SERSIC_MASS"],
        "effective_radius": data["SERSIC_TH50"]
    })

    logging.info(f"Loaded raw shape: {df.shape}")
    return df

def apply_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data_cfg = config["data"]

    initial = len(df)

    # Redshift cut
    df = df[df["redshift"] <= data_cfg["redshift_max"]]
    logging.info(f"After redshift cut: {len(df)} (removed {initial - len(df)})")
    initial = len(df)

    # Sérsic cuts
    df = df[df["sersic_n"] >= data_cfg["sersic_min"]]
    logging.info(f"After sersic_min cut: {len(df)} (removed {initial - len(df)})")
    initial = len(df)

    df = df[df["sersic_n"] <= data_cfg["sersic_max"]]
    logging.info(f"After sersic_max cut: {len(df)} (removed {initial - len(df)})")

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    logging.info(f"After dropping missing: {len(df)} (removed {before - len(df)})")
    return df


def log_transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if not config["preprocessing"]["log_transform"]:
        return df

    for col in ["stellar_mass", "effective_radius"]:
        before = len(df)
        df = df[df[col] > 0]
        removed = before - len(df)
        df[col] = np.log10(df[col])
        logging.info(f"Log-transformed {col} (removed {removed})")

    return df


def save_clean(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned dataset to {output_path}")


def run_data_cleaning(config_path: str):
    config = load_config(config_path)

    fits_path = config["paths"]["raw_data"]
    output_path = config["paths"]["clean_data"]

    df = load_fits_data(fits_path)
    df = apply_filters(df, config)
    df = handle_missing(df)
    df = log_transform(df, config)

    logging.info(f"Final cleaned shape: {df.shape}")

    save_clean(df, output_path)


if __name__ == "__main__":
    run_data_cleaning("config.yaml")