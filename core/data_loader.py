import pandas as pd


REQUIRED_COLUMNS = {
'Date', 'Latitude', 'Longitude', 'Division', 'Range', 'Beat'
}


def load_and_validate_csv(file):
try:
df = pd.read_csv(file)
except UnicodeDecodeError:
file.seek(0)
df = pd.read_csv(file, encoding='ISO-8859-1')


missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
raise ValueError(f"Missing required columns: {missing}")


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])


for c in ['Latitude', 'Longitude']:
df[c] = pd.to_numeric(df[c], errors='coerce')


for c in ['Division', 'Range', 'Beat']:
df[c] = df[c].astype(str).str.title().fillna('Unknown')


if 'Hour' not in df and 'Time' in df:
df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour


return df
