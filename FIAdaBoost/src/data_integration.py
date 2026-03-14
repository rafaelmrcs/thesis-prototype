import os
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

# Safety limit for laptop training
BUILDING_SAMPLE_SIZE = 10000

# Columns needed for Option B target
GHI_COL = "ALLSKY_SFC_SW_DWN"
SEI_COL = "solar_exposure_index"
TARGET_COL = "solar_energy_potential"


def integrate_datasets():
    print("Integrating Datasets (Option B: Solar Energy Potential)...")

    nasa_path = os.path.join(PROCESSED_DIR, "nasa_features.csv")
    osm_path = os.path.join(PROCESSED_DIR, "osm_features.geojson")

    if not os.path.exists(nasa_path) or not os.path.exists(osm_path):
        raise FileNotFoundError("Missing feature files. Run feature_engineering.py first.")

    nasa_df = pd.read_csv(nasa_path)
    osm_gdf = gpd.read_file(osm_path)

    # Validate required columns
    if GHI_COL not in nasa_df.columns:
        raise ValueError(f"NASA features missing required column: {GHI_COL}")
    if SEI_COL not in osm_gdf.columns:
        raise ValueError(f"OSM features missing required column: {SEI_COL}")

    # Sample buildings for feasibility
    if len(osm_gdf) > BUILDING_SAMPLE_SIZE:
        osm_sample = osm_gdf.sample(BUILDING_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        osm_sample = osm_gdf.reset_index(drop=True)

    # Cross Join: Repeat NASA daily data for every building
    nasa_expanded = nasa_df.loc[nasa_df.index.repeat(len(osm_sample))].reset_index(drop=True)
    osm_expanded = pd.concat([osm_sample] * len(nasa_df), ignore_index=True)

    # Concatenate (drop geometry)
    integrated_df = pd.concat([nasa_expanded, osm_expanded.drop(columns="geometry")], axis=1)

    # Make sure numeric
    integrated_df[GHI_COL] = pd.to_numeric(integrated_df[GHI_COL], errors="coerce")
    integrated_df[SEI_COL] = pd.to_numeric(integrated_df[SEI_COL], errors="coerce")

    # Option B Target: Solar Energy Potential
    integrated_df[TARGET_COL] = integrated_df[GHI_COL] * integrated_df[SEI_COL]

    # Clean any NaNs created during coercion
    integrated_df = integrated_df.replace([np.inf, -np.inf], np.nan)
    integrated_df = integrated_df.ffill().bfill()

    output_path = os.path.join(PROCESSED_DIR, "integrated_dataset.csv")
    integrated_df.to_csv(output_path, index=False)

    print(f"Integration Complete.")
    print(f"Saved: {output_path}")
    print(f"Rows: {len(integrated_df)}")
    print(f"Target created: {TARGET_COL} = {GHI_COL} * {SEI_COL}")


if __name__ == "__main__":
    integrate_datasets()