import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# =============================================================================
# CONFIG
# =============================================================================
PLACE_NAME = "Davao City, Philippines"
YEAR = "2024"

N_POINTS = 20000
BUILDING_CAP = 50000

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

KWH_TO_J = 3_600_000

OUTPUT_NAME = f"integrated_dataset_expanded_{N_POINTS}_{YEAR}.csv"

# =============================================================================
# STEP 2 — CLEAN SPATIAL DATA
# =============================================================================
def process_spatial_df(df: pd.DataFrame, year: str = YEAR) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 2 — Process Spatial Dataset")
    print("=" * 70)

    target_col = f"GHI_mean_{year}"

    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["lat", "lon", target_col]).reset_index(drop=True)
    df[target_col] = df[target_col].clip(0.5, 10.0)

    print(f"[Clean] {n_before:,} → {len(df):,} rows")
    print(f"[Range] {target_col}: {df[target_col].min():.3f} to {df[target_col].max():.3f} kWh/m²/day")

    return df


# =============================================================================
# STEP 4 — FEATURE ENGINEERING
# =============================================================================
def topo_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    bounds = gdf.geometry.bounds
    gdf["azimuth"] = np.degrees(
        np.arctan2(bounds.maxx - bounds.minx, bounds.maxy - bounds.miny)
    )

    gdf["orientation_score"] = (np.cos(np.radians(gdf["azimuth"] - 180)) + 1) / 2

    centroids = np.array([(p.x, p.y) for p in gdf.geometry.centroid])
    tree = cKDTree(centroids)
    neighbors = tree.query_ball_point(centroids, r=50)
    gdf["nearby_count"] = [len(n) - 1 for n in neighbors]

    max_count = max(int(gdf["nearby_count"].max()), 1)
    gdf["shading_factor"] = (0.3 * gdf["nearby_count"] / max_count).clip(0, 1)

    optimal_tilt = 7.2
    roof_tilt = 0.0
    gdf["tilt_factor"] = float(np.cos(np.radians(abs(roof_tilt - optimal_tilt))))

    gdf["solar_exposure_index"] = (
        gdf["orientation_score"]
        * gdf["rooftop_area_sq_m"]
        * (1 - gdf["shading_factor"])
        * gdf["tilt_factor"]
    )

    sei_max = gdf["solar_exposure_index"].max()
    gdf["SEI_norm"] = gdf["solar_exposure_index"] / sei_max if sei_max > 0 else 0.0

    print("[Features] Diagnostics:")
    for col in ["orientation_score", "shading_factor", "tilt_factor", "solar_exposure_index", "SEI_norm"]:
        print(f"  {col:<22} mean={gdf[col].mean():.4f} std={gdf[col].std():.4f}")

    return gdf


# =============================================================================
# STEP 5 — INTEGRATE
# =============================================================================
def integrate_datasets(spatial_df: pd.DataFrame, osm_gdf: gpd.GeoDataFrame, year: str = YEAR) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 5 — Integrate Spatial Points with Buildings")
    print("=" * 70)

    target_col = f"GHI_mean_{year}"

    osm_wgs = osm_gdf.to_crs("EPSG:4326").copy()

    centroids = osm_wgs.geometry.centroid
    osm_coords = np.column_stack([centroids.y, centroids.x])
    tree = cKDTree(osm_coords)

    query_coords = spatial_df[["lat", "lon"]].values
    _, idx = tree.query(query_coords, k=1)

    building_cols = [
        "rooftop_area_sq_m",
        "orientation_score",
        "shading_factor",
        "tilt_factor",
        "SEI_norm",
    ]

    matched = osm_wgs.iloc[idx][building_cols].reset_index(drop=True)
    integrated = pd.concat([spatial_df.reset_index(drop=True), matched], axis=1)

    integrated["GHI_mean_J"] = integrated[target_col] * KWH_TO_J

    critical = ["lat", "lon", target_col, "GHI_mean_J"] + building_cols
    integrated = integrated.dropna(subset=critical).reset_index(drop=True)

    out_path = os.path.join(PROCESSED_DIR, OUTPUT_NAME)
    integrated.to_csv(out_path, index=False)

    print(f"[Saved] {out_path}")
    print(f"[Rows] {len(integrated):,}")

    return integrated


# =============================================================================
# MAIN (NO DATA ACQUISITION)
# =============================================================================
def main():
    print("=" * 80)
    print("PIPELINE (NO DATA ACQUISITION)")
    print("=" * 80)

    # Load NASA dataset (already generated)
    raw_path = os.path.join(RAW_DIR, f"baseline_spatial_dataset_davao_city_2024_{N_POINTS}.csv")
    raw_spatial = pd.read_csv(raw_path)

    spatial_clean = process_spatial_df(raw_spatial)

    # Load OSM features (already generated)
    osm_path = os.path.join(PROCESSED_DIR, f"osm_features_expanded_{BUILDING_CAP}.geojson")
    osm_gdf = gpd.read_file(osm_path)

    osm_features = topo_features(osm_gdf)

    integrate_datasets(spatial_clean, osm_features)

    print("\nDONE")


if __name__ == "__main__":
    main()