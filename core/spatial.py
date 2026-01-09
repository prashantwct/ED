import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

VILLAGE_FILE = "centroids.csv"

def attach_nearest_village(df):
    """
    Attaches nearest village and distance (km) to each row.
    Safe: returns df unchanged if centroids.csv is missing.
    """
    try:
        villages = pd.read_csv(VILLAGE_FILE)
    except Exception:
        return df

    if not {"Latitude", "Longitude", "Village"}.issubset(villages.columns):
        return df

    # Convert to radians
    village_coords = np.radians(villages[["Latitude", "Longitude"]].values)
    sighting_coords = np.radians(df[["Latitude", "Longitude"]].values)

    tree = cKDTree(village_coords)
    dist, idx = tree.query(sighting_coords, k=1)

    df["Nearest Village"] = villages.iloc[idx]["Village"].values
    df["Distance to Village (km)"] = dist * 6371
    df["Near Village"] = df["Distance to Village (km)"] < 0.5

    return df
