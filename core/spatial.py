import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


VILLAGE_FILE = 'centroids.csv'


def attach_nearest_village(df):
try:
villages = pd.read_csv(VILLAGE_FILE)
except Exception:
return df


coords_v = np.radians(villages[['Latitude','Longitude']].values)
coords_s = np.radians(df[['Latitude','Longitude']].values)


tree = cKDTree(coords_v)
dist, idx = tree.query(coords_s, k=1)


df['Nearest Village'] = villages.iloc[idx]['Village'].values
df['Distance to Village (km)'] = dist * 6371
df['Near Village'] = df['Distance to Village (km)'] < 0.5


return df
