import pandas as pd
import geopandas as gpd
import numpy as np
import os

# --- Directories ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_nasa():
    """
    Matches Section 2.1.3.4: Cleaning and Interpolation.
    """
    path = os.path.join(ROOT_DIR, "data", "raw", "nasa_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    
    df = pd.read_csv(path)
    
    # Correct Date Parsing for NASA format YYYYMMDD
    df["date"] = df["date"].astype(str)
    df["date"] = pd.to_datetime(df["date"], format='%Y%m%d')
    
    # Replace invalid values (-999) with NaN and interpolate
    df.replace(-999, np.nan, inplace=True)
    df.interpolate(method='linear', inplace=True)
    
    return df

def process_osm():
    path = os.path.join(ROOT_DIR, "data", "raw", "osm_buildings.geojson")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    gdf = gpd.read_file(path)

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    gdf = gdf.to_crs("EPSG:32651")

    gdf["rooftop_area_sq_m"] = gdf.geometry.area

    if len(gdf) > 10000:
        gdf = gdf.sample(10000, random_state=42).reset_index(drop=True)

    return gdf

# def process_baseline_spatial(year="2024"):
#     path = os.path.join(ROOT_DIR, "data", "raw", f"baseline_spatial_dataset_philippines_{year}.csv")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing {path}")

#     df = pd.read_csv(path)
#     target = f"GHI_mean_{year}"

#     df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
#     df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
#     df[target] = pd.to_numeric(df[target], errors="coerce")

#     df = df.dropna(subset=["lat", "lon", target]).reset_index(drop=True)
#     return df


if __name__ == "__main__":
    print("Processing Data...")
    nasa_df = process_nasa()
    osm_gdf = process_osm()
    
    nasa_df.to_csv(os.path.join(PROCESSED_DIR, "nasa_clean.csv"), index=False)
    osm_gdf.to_file(os.path.join(PROCESSED_DIR, "osm_clean.geojson"), driver="GeoJSON")


    # baseline_df = process_baseline_spatial("2024")
    # baseline_df.to_csv(os.path.join(PROCESSED_DIR, "baseline_spatial_clean_2024.csv"), index=False)
    print("Data processing complete.")