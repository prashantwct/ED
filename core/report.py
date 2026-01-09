def generate_html_report(df, start_date, end_date):
    """
    Minimal safe HTML report generator.
    Guaranteed to not fail at startup.
    """
    total_entries = len(df)
    total_severity = df["Severity Score"].sum() if "Severity Score" in df else 0

    return f"""
    <html>
        <head>
            <title>Elephant Monitoring Report</title>
        </head>
        <body style="font-family: Arial, sans-serif;">
            <h2>🐘 Elephant Monitoring Report</h2>
            <p><b>Period:</b> {start_date} to {end_date}</p>
            <p><b>Total Entries:</b> {total_entries}</p>
            <p><b>Total Severity Score:</b> {total_severity:.1f}</p>
        </body>
    </html>
    """
