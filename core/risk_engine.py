def beat_risk_scores(df):
grp = df.groupby('Beat')


risk = grp.apply(lambda x: (
x['Severity Score'].sum() * 0.5 +
(x.get('Is_Night',0).mean() * 100) * 0.3 +
(x.get('Near Village',0).mean() * 100) * 0.2
)).reset_index(name='Risk Score')


return risk.sort_values('Risk Score', ascending=False)
