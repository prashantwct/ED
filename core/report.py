def generate_html_report(df, start, end):
return f"""
<html>
<body style='font-family:Arial;'>
<h2>Elephant Monitoring Report</h2>
<p><b>Period:</b> {start} to {end}</p>
<p>Total Entries: {len(df)}</p>
<p>Total Severity: {df['Severity Score'].sum():.1f}</p>
</body>
</html>
"""
