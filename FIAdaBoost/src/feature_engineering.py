import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")


# =========================
# 2.2.1 TEMPORAL FEATURES
# =========================
def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Missing required column: 'date'")

    df = df.copy()
    df["month"] = df["date"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Philippines seasons (PAGASA): Dry (Dec–May) = 1, Rainy (Jun–Nov) = 0
    df["season"] = df["month"].apply(lambda m: 1 if m in [12, 1, 2, 3, 4, 5] else 0)

    return df


# =========================
# Leakage guard (important)
# =========================
LEAKAGE_COLS = {"sunshine_flag", "year_month", "sunshine_hours", "clear_sky_ratio"}

def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard safety: remove known leakage columns if they exist.
    These columns are derived from ALLSKY_SFC_SW_DWN (the target),
    so they should NOT be used for training when predicting that target.
    """
    df = df.copy()
    existing = [c for c in LEAKAGE_COLS if c in df.columns]
    if existing:
        print(f"[Leakage Guard] Dropping leakage columns: {existing}")
        df = df.drop(columns=existing, errors="ignore")
    return df


# =========================
# 2.2.3 TOPOGRAPHICAL FEATURES
# =========================
def topo_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    if "geometry" not in gdf.columns:
        raise ValueError("OSM GeoDataFrame must contain a 'geometry' column.")
    if "rooftop_area_sq_m" not in gdf.columns:
        raise ValueError("Missing 'rooftop_area_sq_m'. Ensure data_processing computed rooftop areas.")

    # Ensure CRS exists; OSM data is usually EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # Project to meters for buffer distance
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=32651)  # UTM Zone 51N covers much of PH

    # ================
    # 1) ORIENTATION SCORE (azimuth proxy from bounding box direction)
    # ================
    bounds = gdf.geometry.bounds
    gdf["azimuth"] = np.degrees(np.arctan2(bounds.maxx - bounds.minx, bounds.maxy - bounds.miny))
    gdf["orientation_score"] = np.cos(np.radians(gdf["azimuth"] - 180))

    # ================
    # 2) SHADING FACTOR (density heuristic within 50m)
    # ================
    centroids = np.array([(p.x, p.y) for p in gdf.geometry.centroid])
    tree = cKDTree(centroids)
    neighbors = tree.query_ball_point(centroids, r=50)
    gdf["nearby_count"] = [len(n) - 1 for n in neighbors]

    max_buildings = int(gdf["nearby_count"].max()) if len(gdf) else 0
    if max_buildings <= 0:
        max_buildings = 1

    gdf["shading_factor"] = 0.3 * (gdf["nearby_count"] / max_buildings)
    gdf["shading_factor"] = gdf["shading_factor"].clip(0, 1)  # keep physical bounds

    # ================
    # 3) TILT FACTOR (assumed roof tilt, compared to optimal tilt)
    # ================
    optimal_tilt = 7.2
    roof_tilt = 0.0
    gdf["tilt_factor"] = np.cos(np.radians(abs(roof_tilt - optimal_tilt)))

    # ================
    # 4) SOLAR EXPOSURE INDEX (SEI)
    # ================
    gdf["solar_exposure_index"] = (
        gdf["orientation_score"]
        * gdf["rooftop_area_sq_m"]
        * (1 - gdf["shading_factor"])
        * gdf["tilt_factor"]
    )

    return gdf


# =========================
# PIPELINE EXECUTION
# =========================
if __name__ == "__main__":
    nasa_path = os.path.join(PROCESSED_DIR, "nasa_clean.csv")
    osm_path = os.path.join(PROCESSED_DIR, "osm_clean.geojson")

    if not os.path.exists(nasa_path):
        raise FileNotFoundError(f"Missing {nasa_path}. Run data_processing.py first.")
    if not os.path.exists(osm_path):
        raise FileNotFoundError(f"Missing {osm_path}. Run data_processing.py first.")

    nasa_df = pd.read_csv(nasa_path)
    nasa_df["date"] = pd.to_datetime(nasa_df["date"], errors="coerce")
    nasa_df = nasa_df.dropna(subset=["date"]).reset_index(drop=True)

    # Temporal features
    nasa_df = temporal_features(nasa_df)

    # Leakage guard (drops any accidental leakage columns)
    nasa_df = drop_leakage_cols(nasa_df)

    # OSM topo features
    osm_gdf = gpd.read_file(osm_path)
    osm_gdf = topo_features(osm_gdf)

    nasa_df.to_csv(os.path.join(PROCESSED_DIR, "nasa_features.csv"), index=False)
    osm_gdf.to_file(os.path.join(PROCESSED_DIR, "osm_features.geojson"), driver="GeoJSON")

    print("Feature Engineering Complete (leakage-safe).")
