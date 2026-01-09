import pandas as pd


SEVERITY_WEIGHTS = {
'presence': 0.5,
'crop': 2.5,
'house': 5.0,
'injury': 20.0
}


def compute_severity(df):
return (
(df.get('Total Count', 0) > 0).astype(int) * SEVERITY_WEIGHTS['presence'] +
(df.get('Crop Damage', 0) > 0).astype(int) * SEVERITY_WEIGHTS['crop'] +
(df.get('House Damage', 0) > 0).astype(int) * SEVERITY_WEIGHTS['house'] +
(df.get('Injury', 0) > 0).astype(int) * SEVERITY_WEIGHTS['injury']
)


def compute_kpis(df):
return {
'entries': len(df),
'conflicts': ((df.get('Crop Damage',0)>0) | (df.get('House Damage',0)>0) | (df.get('Injury',0)>0)).sum(),
'severity': df['Severity Score'].sum(),
'night_pct': df.get('Is_Night', pd.Series()).mean() * 100
}
