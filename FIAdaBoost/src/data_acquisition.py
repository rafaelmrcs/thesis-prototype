import os
import time
import requests
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# --- Directories ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# =========================
# Helpers
# =========================
def get_city_boundary(place_name: str) -> gpd.GeoDataFrame:
    gdf = ox.geocode_to_gdf(place_name)
    if gdf.empty:
        raise ValueError(f"Could not geocode place boundary for: {place_name}")
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf

def sample_random_points_in_polygon(polygon, n_points=3000, seed=42) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon.bounds

    pts = []
    while len(pts) < n_points:
        lon = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        p = Point(lon, lat)
        if polygon.contains(p):
            pts.append(p)

    gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf[["lat", "lon", "geometry"]]

def nasa_power_request(lat: float, lon: float, start: str, end: str, parameters: str) -> dict:
    params = {
        "parameters": parameters,
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    r = requests.get(NASA_BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_series(js: dict, param: str) -> pd.Series:
    series = js["properties"]["parameter"][param]
    return pd.to_numeric(pd.Series(series), errors="coerce")

# =========================
# A) Baseline reproduction dataset (spatial)
# =========================
def fetch_nasa_baseline_spatial(place_name="Davao City, Philippines", year="2024",
                               n_points=3000, seed=42, sleep_s=0.2):
    print(f"[Baseline Spatial] Sampling {n_points} points in {place_name}...")

    boundary = get_city_boundary(place_name)
    poly = boundary.geometry.iloc[0]
    pts = sample_random_points_in_polygon(poly, n_points=n_points, seed=seed)

    out_path = os.path.join(
        RAW_DIR,
        f"baseline_spatial_dataset_{place_name.split(',')[0].lower().replace(' ', '_')}_{year}.csv"
    )
    tmp_path = out_path.replace(".csv", "_partial.csv")

    if os.path.exists(tmp_path):
        done_df = pd.read_csv(tmp_path)
        done_set = set(zip(done_df["lat"].round(6), done_df["lon"].round(6)))
        print(f"[Baseline Spatial] Resuming: {len(done_df)} rows done.")
    else:
        done_df = pd.DataFrame(columns=["lat", "lon", f"GHI_mean_{year}"])
        done_set = set()

    start = f"{year}0101"
    end = f"{year}1231"

    buffer_rows = []
    for _, r in pts.iterrows():
        lat = float(r["lat"])
        lon = float(r["lon"])
        key = (round(lat, 6), round(lon, 6))
        if key in done_set:
            continue

        try:
            js = nasa_power_request(lat, lon, start, end, parameters="ALLSKY_SFC_SW_DWN")
            ghi = extract_series(js, "ALLSKY_SFC_SW_DWN")
            ghi_mean = float(ghi.mean(skipna=True))
            buffer_rows.append({"lat": lat, "lon": lon, f"GHI_mean_{year}": ghi_mean})
        except Exception as e:
            print(f"[Baseline Spatial] Error at ({lat:.4f},{lon:.4f}): {e}")
            buffer_rows.append({"lat": lat, "lon": lon, f"GHI_mean_{year}": np.nan})

        if len(buffer_rows) >= 50:
            done_df = pd.concat([done_df, pd.DataFrame(buffer_rows)], ignore_index=True)
            done_df.to_csv(tmp_path, index=False)
            buffer_rows.clear()
            print(f"[Baseline Spatial] Saved progress: {len(done_df)} rows.")
        time.sleep(sleep_s)

    if buffer_rows:
        done_df = pd.concat([done_df, pd.DataFrame(buffer_rows)], ignore_index=True)

    done_df.to_csv(out_path, index=False)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"[Baseline Spatial] Done. Saved to {out_path}")

# =========================
# B) Time-series
# =========================
def fetch_nasa_timeseries(place_name="Davao City, Philippines", year="2024"):
    print(f"[Time Series] Fetching NASA POWER daily data for {place_name} (centroid)...")

    boundary = get_city_boundary(place_name)
    poly = boundary.geometry.iloc[0]
    centroid = poly.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)

    start = f"{year}0101"
    end = f"{year}1231"

    js = nasa_power_request(
        lat, lon, start, end,
        parameters="ALLSKY_SFC_SW_DWN,T2M,RH2M,ALLSKY_KT"
    )

    features = js["properties"]["parameter"]
    dates = list(features["ALLSKY_SFC_SW_DWN"].keys())

    df = pd.DataFrame({
        "date": dates,
        "ALLSKY_SFC_SW_DWN": list(features["ALLSKY_SFC_SW_DWN"].values()),
        "T2M": list(features["T2M"].values()),
        "RH2M": list(features["RH2M"].values()),
        "ALLSKY_KT": list(features["ALLSKY_KT"].values()),
        "lat": lat,
        "lon": lon,
    })

    out_path = os.path.join(RAW_DIR, "nasa_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"[Time Series] Saved to {out_path} (lat={lat:.5f}, lon={lon:.5f})")

# =========================
# C) OSM Buildings
# =========================
def fetch_osm_data():
    print("Fetching OpenStreetMap Data for Davao City...")
    place_name = "Davao City, Philippines"
    try:
        tags = {"building": True}
        gdf = ox.features_from_place(place_name, tags=tags)
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf[["geometry"]]
        output_path = os.path.join(RAW_DIR, "osm_buildings.geojson")
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"OSM Data saved to {output_path}")
    except Exception as e:
        print(f"Error fetching OSM data: {e}")

if __name__ == "__main__":
    # # 1) Replicate the Sales et al. methodology on OUR map (Davao)
    # fetch_nasa_baseline_spatial(place_name="Davao City, Philippines", n_points=1000)

    # 2) Your main time-series data
    fetch_nasa_timeseries()

    # 3) OSM buildings
    fetch_osm_data()