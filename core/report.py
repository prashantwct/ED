def _table_rows(df, columns):
    rows = "".join(
        "<tr>" + "".join(f"<td style='padding:4px 10px;border-bottom:1px solid #eee'>{r[c]}</td>" for c in columns) + "</tr>"
        for _, r in df.iterrows()
    )
    header = "".join(f"<th style='text-align:left;padding:4px 10px'>{c}</th>" for c in columns)
    return f"<table style='border-collapse:collapse'><tr>{header}</tr>{rows}</table>"


def generate_html_report(df, start_date, end_date, kpis, division_rates=None, top_beats=None):
    """
    Minimal, dependency-free HTML report reflecting the same KPIs shown
    on screen -- deliberately simple so it can never fail to render,
    but no longer just two numbers: it now carries the metrics that
    actually matter for a conflict review (human deaths, night share,
    conflict rate by division, top-risk beats).
    """
    total_entries = kpis.get("entries", len(df))
    conflicts = kpis.get("conflicts", 0)
    deaths = kpis.get("human_deaths", 0)
    death_incidents = kpis.get("human_death_incidents", 0)
    night_pct = kpis.get("night_pct", 0)
    severity = kpis.get("severity", 0)

    division_html = ""
    if division_rates is not None and not division_rates.empty:
        dr = division_rates.reset_index().rename(columns={"index": "Division"})
        division_html = "<h3>Conflict rate by division</h3>" + _table_rows(
            dr, ["Division", "Sightings", "Conflict Events", "Conflict Rate %"]
        )

    beats_html = ""
    if top_beats is not None and not top_beats.empty:
        beats_html = "<h3>Top priority beats</h3>" + _table_rows(
            top_beats.head(10), ["Beat", "Sightings", "Risk Score"]
        )

    return f"""
    <html>
        <head><title>Elephant Monitoring Report</title></head>
        <body style="font-family: Arial, sans-serif; color:#222; max-width:720px; margin:auto;">
            <h2>Elephant Monitoring Report</h2>
            <p><b>Period:</b> {start_date} to {end_date}</p>
            <ul>
                <li><b>Total entries:</b> {total_entries}</li>
                <li><b>Conflict-related entries:</b> {conflicts}</li>
                <li><b>Human deaths:</b> {deaths:.0f} across {death_incidents} incident(s)</li>
                <li><b>Share of conflict at night (7 PM-2 AM):</b> {night_pct:.1f}%</li>
                <li><b>Total severity score:</b> {severity:.1f}</li>
            </ul>
            {division_html}
            {beats_html}
        </body>
    </html>
    """
