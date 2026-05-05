"""
feature_engineering.py
────────────────────────────────────────────────────────────────────────────
Implements §2.2 Feature Engineering from the thesis methodology.

WHAT THIS FILE DOES IN THE NEW DESIGN
──────────────────────────────────────
The primary dataset is 20,000 spatial coordinates (lat, lon, GHI_mean).
This file's job is to compute per-building topographical features from the
OSM building polygons so that data_integration.py can attach the nearest
building's features to each of the 20,000 spatial points.

ACTIVE FUNCTIONS (called by __main__ and data_integration.py):
  topo_features()     — computes orientation, shading, tilt, SEI per building
  normalize_sei()     — normalises solar_exposure_index to SEI_norm [0,1]

INACTIVE FUNCTIONS (kept for reference / optional daily-series analysis):
  temporal_features()          — adds month_sin, month_cos, season to a daily df
  drop_leakage_cols()          — removes GHI-derived features from a daily df
  aggregate_building_features()— reduces buildings to city-level means
  These are NOT used in the spatial 20,000-point pipeline.

OUTPUT
──────
  data/processed/osm_features.geojson
    One row per OSM building (up to 10,000).
    New columns: orientation_score, shading_factor, tilt_factor,
                 solar_exposure_index, SEI_norm
────────────────────────────────────────────────────────────────────────────
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")


# =============================================================================
# ACTIVE — 2.2.3  TOPOGRAPHICAL FEATURES
# Called by __main__ and imported by data_integration.py
# =============================================================================

def topo_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute per-building topographical features from OSM geometry (§2.2.3).

    Features produced:
      orientation_score  — how south-facing the rooftop is (§2.2.3.1)
      shading_factor     — estimated shading from nearby buildings (§2.2.3.3)
      tilt_factor        — how close roof tilt is to optimal (§2.2.3.4)
      solar_exposure_index (SEI) — composite score (§2.2.3)

    Input GeoDataFrame must have:
      geometry            — Polygon or MultiPolygon building footprint
      rooftop_area_sq_m   — pre-computed area in m² (from data_processing.py)
    """
    gdf = gdf.copy()

    if "geometry" not in gdf.columns:
        raise ValueError("GeoDataFrame must contain a 'geometry' column.")
    if "rooftop_area_sq_m" not in gdf.columns:
        raise ValueError(
            "Missing 'rooftop_area_sq_m'. Run data_processing.process_osm() first."
        )

    # Project to UTM Zone 51N (metres) for accurate distance calculations
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=32651)

    # ── §2.2.3.1  Orientation Score ───────────────────────────────────────
    # Azimuth derived from bounding-box dimensions.
    # cos(azimuth − 180°) → south-facing = +1, north-facing = −1.
    # Shifted to [0, 1] so no negative values enter the SEI product.
    bounds = gdf.geometry.bounds
    gdf["azimuth"] = np.degrees(
        np.arctan2(bounds.maxx - bounds.minx, bounds.maxy - bounds.miny)
    )
    gdf["orientation_score"] = (
        np.cos(np.radians(gdf["azimuth"] - 180)) + 1
    ) / 2

    # ── §2.2.3.3  Shading Factor ──────────────────────────────────────────
    # Heuristic: fraction of nearby buildings (within 50 m) relative to max.
    # Adapted from urban solar potential studies.
    centroids = np.array([(p.x, p.y) for p in gdf.geometry.centroid])
    tree      = cKDTree(centroids)
    neighbors = tree.query_ball_point(centroids, r=50)
    gdf["nearby_count"] = [len(n) - 1 for n in neighbors]   # exclude self

    max_count = max(int(gdf["nearby_count"].max()), 1)
    gdf["shading_factor"] = (
        0.3 * gdf["nearby_count"] / max_count
    ).clip(0, 1)

    # ── §2.2.3.4  Tilt Factor ─────────────────────────────────────────────
    # Optimal tilt ≈ latitude of Davao City (7.2°).
    # OSM has no per-building tilt data → flat roof (0°) assumed for all.
    # tilt_factor = cos(|0° − 7.2°|) ≈ 0.992  (constant for all buildings).
    optimal_tilt      = 7.2
    roof_tilt         = 0.0
    gdf["tilt_factor"] = float(
        np.cos(np.radians(abs(roof_tilt - optimal_tilt)))
    )

    # ── §2.2.3  Solar Exposure Index (SEI) ───────────────────────────────
    gdf["solar_exposure_index"] = (
        gdf["orientation_score"]
        * gdf["rooftop_area_sq_m"]
        * (1 - gdf["shading_factor"])
        * gdf["tilt_factor"]
    )

    print("[Topo Features] Diagnostics:")
    for col in ["orientation_score", "shading_factor",
                "tilt_factor", "solar_exposure_index"]:
        print(f"  {col:<25}  mean={gdf[col].mean():.4f}  "
              f"std={gdf[col].std():.4f}  "
              f"range=[{gdf[col].min():.4f}, {gdf[col].max():.4f}]")

    return gdf


def normalize_sei(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalise solar_exposure_index to [0, 1] → SEI_norm.

    SEI_norm is what gets attached to each spatial point and used as a
    model feature.  Normalisation keeps values on a bounded scale so they
    are comparable across buildings of very different sizes.
    """
    if "solar_exposure_index" not in gdf.columns:
        raise ValueError(
            "Run topo_features() before normalize_sei()."
        )
    gdf    = gdf.copy()
    sei_max = gdf["solar_exposure_index"].max()
    gdf["SEI_norm"] = (
        gdf["solar_exposure_index"] / sei_max if sei_max > 0 else 0.0
    )
    print(f"[SEI Norm] SEI_norm  mean={gdf['SEI_norm'].mean():.4f}  "
          f"range=[{gdf['SEI_norm'].min():.4f}, {gdf['SEI_norm'].max():.4f}]")
    return gdf


# =============================================================================
# INACTIVE — kept for reference / optional daily time-series pipeline
# These functions are NOT called in the spatial 20,000-point pipeline.
# =============================================================================

def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [INACTIVE in spatial pipeline]
    Adds month, month_sin, month_cos, season to a NASA daily DataFrame.
    Only relevant if running a daily time-series model (not the 20,000-point design).
    """
    if "date" not in df.columns:
        raise ValueError("Missing required column: 'date'")
    df = df.copy()
    df["month"]     = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["season"]    = df["month"].apply(
        lambda m: 1 if m in [12, 1, 2, 3, 4, 5] else 0
    )
    return df


LEAKAGE_COLS = {"sunshine_flag", "year_month", "sunshine_hours", "clear_sky_ratio"}

def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    [INACTIVE in spatial pipeline]
    Removes features derived from ALLSKY_SFC_SW_DWN (GHI).
    Only relevant in a daily time-series model where GHI is also a feature.
    """
    df      = df.copy()
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if to_drop:
        print(f"[Leakage Guard] Dropping: {to_drop}")
        df = df.drop(columns=to_drop, errors="ignore")
    return df


def aggregate_building_features(gdf: gpd.GeoDataFrame) -> dict:
    """
    [INACTIVE in spatial pipeline]
    Reduces per-building GDF to city-level mean statistics.
    Was used in the old daily cross-join design. No longer called.
    """
    required = ["rooftop_area_sq_m", "orientation_score",
                "shading_factor", "tilt_factor", "SEI_norm"]
    missing  = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return {
        "mean_rooftop_area_sq_m": float(gdf["rooftop_area_sq_m"].mean()),
        "mean_orientation_score": float(gdf["orientation_score"].mean()),
        "mean_shading_factor":    float(gdf["shading_factor"].mean()),
        "mean_tilt_factor":       float(gdf["tilt_factor"].mean()),
        "mean_SEI_norm":          float(gdf["SEI_norm"].mean()),
    }


# =============================================================================
# MAIN — spatial pipeline: OSM buildings only
# =============================================================================

if __name__ == "__main__":
    osm_path = os.path.join(PROCESSED_DIR, "osm_clean.geojson")

    if not os.path.exists(osm_path):
        raise FileNotFoundError(
            f"Missing: {osm_path}\n"
            "Run data_processing.process_osm() first."
        )

    print("=" * 55)
    print("Feature Engineering — OSM Building Topographical Features")
    print("=" * 55)

    # Load cleaned OSM buildings
    osm_gdf = gpd.read_file(osm_path)
    print(f"[Load] {len(osm_gdf):,} buildings loaded from osm_clean.geojson")

    # Compute topographical features
    osm_gdf = topo_features(osm_gdf)

    # Normalise SEI
    osm_gdf = normalize_sei(osm_gdf)

    # Save
    out_path = os.path.join(PROCESSED_DIR, "osm_features.geojson")
    osm_gdf.to_file(out_path, driver="GeoJSON")

    print(f"\n[Done] osm_features.geojson saved")
    print(f"  Buildings  : {len(osm_gdf):,}")
    print(f"  New columns: orientation_score, shading_factor, "
          f"tilt_factor, solar_exposure_index, SEI_norm")
    print(f"  Path       : {out_path}")
    print(f"\nNext → data_integration.py")
