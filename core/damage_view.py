"""The damage and compensation view: Streamlit layout over ``core.damage``.

A separate view, not more sections of the conflict dashboard, because it
answers different questions for different people. The conflict view
asks where elephants are causing trouble and which beats to resource.
This one asks what households have lost, who has not been helped to
claim, whether the damage reached the Gaj Rakshak register at all, and
how fast surveys follow the damage.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from core import damage
from core import damage_charts as dc
from core.ui import findings, section

FILES_KEY = "damage_files"
FETCH_KEY = "damage_fetch"
CHART_CONFIG = {"displayModeBar": False}


def _plot(fig, df: Optional[pd.DataFrame] = None, key: str = "", name: str = "") -> None:
    """A chart, and under it a Download menu for its SVG twin."""
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    if df is not None and key:
        _download(df, key, name)


def _download(df: pd.DataFrame, key: str, name: str) -> None:
    from core.damage_report import figure
    from core.exports import download_menu

    download_menu(name, lambda: figure(df, key))


@st.cache_data(show_spinner="Reading the damage surveys...")
def _load(files: Tuple[Tuple[bytes, str], ...]):
    return damage.load(files)


def _fetch(force: bool):
    from core.epicollect import EpicollectError, fetch_entries

    with st.status("Reading the Epicollect5 damage surveys...", expanded=False) as status:
        try:
            result = fetch_entries(force=force, progress=lambda n: status.update(label=n))
        except EpicollectError as exc:
            status.update(label="Could not read Epicollect5", state="error")
            st.error(str(exc))
            return None
        status.update(label=f"Epicollect5: {result.rows:,} survey(s), "
                      f"{result.requests} request(s)", state="complete")
    return result


def sources() -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Uploaded exports and/or the API sync, read into one frame."""
    st.markdown(
        "Household crop and house damage surveys from the Epicollect5 "
        "**herd-crop-damage** and **herd-house-damage** projects. Upload the CSV "
        "or ZIP export from each project's Data page, or fetch them directly."
    )
    upload_col, fetch_col = st.columns([3, 1])
    with upload_col:
        uploads = st.file_uploader(
            "Epicollect5 exports (CSV or ZIP, one or both projects)",
            type=["csv", "zip"], accept_multiple_files=True, key="damage_uploads",
        )
    with fetch_col:
        st.write("")
        fetch = st.button("Fetch from Epicollect5", width="stretch",
                          help="Needs each project's client app credentials in the "
                          "app's secrets while the projects are private.")
    if fetch:
        result = _fetch(force=FETCH_KEY in st.session_state)
        if result is not None:
            st.session_state[FETCH_KEY] = result

    files: List[Tuple[bytes, str]] = [(f.getvalue(), f.name) for f in uploads or []]
    # Streamlit forgets an upload once its page is not drawn, so a trip to
    # the conflict view would lose it. The files are kept for the session.
    if files:
        st.session_state[FILES_KEY] = tuple(files)
    elif st.session_state.get(FILES_KEY):
        kept = st.session_state[FILES_KEY]
        note, clear = st.columns([6, 1])
        note.caption("Using the export(s) uploaded earlier: "
                     + ", ".join(name for _, name in kept) + ".")
        if clear.button("Clear", key="damage_clear"):
            del st.session_state[FILES_KEY]
            st.rerun()
        files = list(kept)
    fetched = st.session_state.get(FETCH_KEY)
    tables_note: List[str] = []
    if fetched is not None:
        for slug, message in fetched.errors.items():
            st.warning(f"**{slug} was not fetched.** {message}")
        tables_note.append(f"Fetched {fetched.rows:,} survey(s) at {fetched.fetched_at}.")

    if not files and fetched is None:
        return None, []
    try:
        frame, notes = _load(tuple(files)) if files else (None, [])
        if fetched is not None and fetched.rows:
            tables = []
            for slug, entries in fetched.entries.items():
                table = damage.entries_frame(entries)
                table.attrs["source"] = slug
                tables.append(table)
            api_frame, api_notes = damage.tidy(tables)
            frame = api_frame if frame is None else _combine(frame, api_frame)
            notes = notes + api_notes
    except damage.DamageDataError as exc:
        st.error(str(exc))
        return None, []
    return frame, tables_note + notes


def _combine(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([a, b], ignore_index=True)
    has_id = both["Entry"].fillna("").astype(str) != ""
    return both[~(has_id & both.duplicated("Entry"))].reset_index(drop=True)


@st.cache_data(show_spinner="Drawing the HTML study...")
def _study_html(df: pd.DataFrame, notes: Tuple[str, ...], match: Optional[pd.Series],
                scope: str) -> bytes:
    from core.damage_report import build_html

    return build_html(df, notes, match, scope).encode("utf-8")


@st.cache_data(show_spinner="Drawing the PDF study...")
def _study_pdf(df: pd.DataFrame, notes: Tuple[str, ...], match: Optional[pd.Series],
               scope: str) -> bytes:
    from core.damage_report import build_pdf

    return build_pdf(df, notes, match, scope)


def _downloads(df: pd.DataFrame, notes: List[str], match: Optional[pd.Series],
               pick: str) -> None:
    """The whole study as a vector HTML file and an A4 PDF.

    Drawn on demand: building both takes a few seconds, which every
    rerun of the page should not pay.
    """
    scope = "" if pick == "All" else f"{pick} damage only"
    stem = "damage_study" + ("" if pick == "All" else f"_{pick.lower()}")
    with st.container(border=True):
        st.markdown("**Download the study** -- every chart and the map as vector "
                    "graphics, sharp at any zoom and in print. Same content in both.")
        if not st.session_state.get("damage_report_ready"):
            if st.button("Prepare HTML and PDF", key="damage_prepare"):
                st.session_state["damage_report_ready"] = True
                st.rerun()
            return
        left, right = st.columns(2)
        try:
            html = _study_html(df, tuple(notes), match, scope)
            pdf = _study_pdf(df, tuple(notes), match, scope)
        except Exception as exc:  # noqa: BLE001 - the view must survive a failed export
            st.error(f"The study could not be drawn: {type(exc).__name__}: {exc}")
            return
        left.download_button("Download HTML", html, f"{stem}.html", mime="text/html",
                             width="stretch", type="primary")
        right.download_button("Download PDF (A4)", pdf, f"{stem}.pdf",
                              mime="application/pdf", width="stretch", type="primary")


def render(df: pd.DataFrame, notes: List[str], register: Optional[pd.DataFrame],
           basemap: str) -> None:
    """The whole view, top to bottom."""
    if notes:
        with st.expander(f"{len(notes)} data note(s)", expanded=False):
            for note in notes:
                st.caption(note)

    kinds = sorted(df["Kind"].unique())
    pick = st.segmented_control("Damage type", ["All"] + kinds, default="All",
                                key="damage_kind") or "All"
    if pick != "All":
        df = df[df["Kind"] == pick]
    if df.empty:
        st.info("No surveys of that type.")
        return

    match = None
    if register is not None and not register.empty:
        with st.sidebar.expander("Gaj Rakshak cross-check", expanded=False):
            radius = st.slider("Match within (km)", 0.25, 5.0, 1.0, 0.25)
            days = st.slider("Match within (days of the damage)", 0, 14, 3)
        match = damage.match_register(df, register, radius_km=radius, days=days)

    s = damage.summary(df)

    # -- Headline numbers ------------------------------------------------------
    # Deltas here describe, they do not compare, so no arrows.
    note = dict(delta_color="off", delta_arrow="off", border=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Households surveyed", f"{s['reports']:,}",
              delta=f"{s['crop']} crop, {s['house']} house", **note)
    c2.metric("Estimated loss (Rs)", damage.lakh(s["loss"]),
              delta=f"{s['villages']} villages", **note)
    c3.metric("No claim made", f"{s['not_applied']:,}",
              delta=f"{s['not_applied'] / s['reports']:.0%} of cases", **note,
              help="Compensation recorded as 'Not applied'. Their estimated loss is "
              f"{damage.rupees(s['not_applied_loss'])}.")
    c4.metric("Compensation paid (Rs)", damage.lakh(s["paid"]),
              delta=f"{s['paid_cases']} case(s)", **note,
              help=f"{s['in_process']} more in process, {s['not_received']} recorded "
              "as not received.")
    c5.metric("Days, damage to survey", "-" if s["median_days"] != s["median_days"]
              else f"{s['median_days']:.0f}", delta="median", **note)

    section("assessment", "Assessment", "What the surveys say, in order of what needs doing")
    findings(damage.headlines(df, match))
    _downloads(df, notes, match, pick)

    # -- Compensation ----------------------------------------------------------
    section("report", "Compensation",
            "Where each household's claim stands. A case not applied for is "
            "the one field staff can still change.")
    pipeline = damage.compensation_pipeline(df)
    left, right = st.columns(2)
    with left:
        st.markdown("**Cases by claim status**")
        _plot(dc.compensation_bars(pipeline, "Cases"), df, "claims_cases",
              "Cases by claim status")
    with right:
        st.markdown("**Estimated loss by claim status**")
        _plot(dc.compensation_bars(pipeline, "Loss"), df, "claims_loss",
              "Estimated loss by claim status")

    queue = damage.follow_up(df)
    st.markdown(f"**Follow-up queue** -- {len(queue)} case(s) with no claim, largest loss first")
    if queue.empty:
        st.success("Every surveyed household has a claim made or recorded.")
    else:
        st.dataframe(
            queue, width="stretch", hide_index=True,
            column_config={
                "Entry": st.column_config.TextColumn(
                    "Epicollect5 entry", help="Open it in the project to see who to visit."),
                "Damage Date": st.column_config.DateColumn(format="DD MMM YYYY"),
                "Estimated Loss": st.column_config.NumberColumn("Loss (Rs)", format="%,.0f"),
                "Repeat Household": st.column_config.CheckboxColumn("Hit before"),
            },
        )
        st.download_button("Download follow-up queue (CSV)",
                           queue.to_csv(index=False).encode("utf-8"),
                           "damage_follow_up.csv", mime="text/csv")
        st.caption(
            "Names and phone numbers stay in Epicollect5: the entry ID opens the "
            "record for anyone with access to the project."
        )

    # -- Losses over time ------------------------------------------------------
    section("trend", "Losses over time", "Estimated loss by month of damage")
    _plot(dc.monthly_loss(damage.monthly(df)), df, "monthly", "Estimated loss by month")

    # -- Gaj Rakshak gap -------------------------------------------------------
    section("broadcast", "Gaj Rakshak gap",
            "Damage that never reached the register is invisible to the conflict "
            "dashboard and to the beat ranking.")
    r1, r2, r3 = st.columns(3)
    r1.metric("Remarks: not in Gaj Rakshak", f"{s['register_missing']:,}",
              delta=f"of {s['reports']} surveys", **note)
    r2.metric("Remarks: wrong location there", f"{s['register_wrong']:,}", border=True)
    if match is not None:
        r3.metric("No register report nearby", f"{int((~match).sum()):,}",
                  delta=f"{(~match).mean():.0%} of surveys", **note,
                  help="No Gaj Rakshak report within the distance and days set in "
                  "the sidebar.")
        st.caption(
            "The cross-check compares positions and dates with the register loaded "
            "in the conflict view. Remarks only count what surveyors wrote down."
        )
    else:
        r3.info("Load the Gaj Rakshak register in the conflict view to check every "
                "survey against it by place and date.")

    # -- Where -----------------------------------------------------------------
    section("village", "Villages",
            "Beat is the one the point falls in, or the nearest when it lies on "
            "farmland outside every beat.")
    table = damage.villages(df, match)
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Estimated Loss": st.column_config.NumberColumn("Loss (Rs)", format="%,.0f"),
            "Last Damage": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Repeat Households": st.column_config.NumberColumn("Hit twice+"),
        },
    )
    st.download_button("Download village summary (CSV)",
                       table.to_csv(index=False).encode("utf-8"),
                       "damage_by_village.csv", mime="text/csv")
    from core.map_engine import render_damage_map

    render_damage_map(df, dc.STATUS_COLORS, style_name=basemap)
    _download(df, "map", "Damage survey map")
    legend = " ".join(
        f'<span class="ci-legend__item"><span class="ci-legend__dot" '
        f'style="background:{color};"></span>{status}</span>'
        for status, color in dc.STATUS_COLORS.items()
        if status in set(df["Compensation"])
    )
    st.markdown(f'<div class="ci-legend">{legend}</div>', unsafe_allow_html=True)
    st.caption("Circle area is estimated loss; colour is the claim's status. "
               "Fields are solid, houses ringed.")

    # -- Who and when ----------------------------------------------------------
    section("clock", "When and by how many",
            "How many elephants were present, and when the damage was done.")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Elephants present**")
        _plot(dc.by_kind_bars(damage.herd_sizes(df), "Herd", damage.HERD_LABELS,
                              x_title="Elephants in the group"), df, "herd", "Elephants present")
    with h2:
        st.markdown("**Hour of damage**")
        _plot(dc.hour_profile(damage.hourly(df)), df, "hours", "Hour of damage")

    # -- Crop detail -----------------------------------------------------------
    crop = df[df["Kind"] == damage.CROP]
    if not crop.empty:
        section("target", "Crop damage", f"{len(crop)} fields")
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("**Crops damaged**")
            _plot(dc.ranked_bars(damage.crop_table(df), "Crop", color=dc.KIND_COLORS[damage.CROP]),
                  df, "crops", "Crops damaged")
            st.caption("A field with two crops counts under both.")
        with k2:
            st.markdown("**Stage of the crop**")
            stages = damage.exploded_counts(crop, "Crop Stage", damage.CROP_STAGES)
            _plot(dc.ranked_bars(stages, "Crop Stage",
                                 color=dc.KIND_COLORS[damage.CROP]),
                  df, "stages", "Stage of the crop")
        k3, k4 = st.columns(2)
        with k3:
            st.markdown("**Fencing at the damaged field**")
            _plot(dc.ranked_bars(damage.exploded_counts(crop, "Fencing"), "Fencing",
                                 color="#6B8578"),
                  df, "fencing", "Fencing at the damaged field")
            st.caption(
                "Only damaged fields are surveyed, so this is what was in place where "
                "damage happened -- not evidence of which fence works."
            )
        with k4:
            check = damage.area_check(df)
            if check:
                st.markdown("**Damaged area: claimed against measured**")
                _plot(dc.area_comparison(check), df, "area",
                      "Damaged area claimed against measured")
                st.caption(
                    f"Over {check['cases']} fields with both figures, the calculated "
                    f"area is {check['ratio']:.0%} of what owners reported, in the "
                    "units the form records -- the gap a claim is checked against."
                )

    # -- House detail ----------------------------------------------------------
    house = df[df["Kind"] == damage.HOUSE]
    if not house.empty:
        section("village", "House damage", f"{len(house)} houses")
        q1, q2, q3 = st.columns(3)
        with q1:
            st.markdown("**Type of house**")
            _plot(dc.ranked_bars(damage.exploded_counts(house, "House Type"), "House Type",
                                 color=dc.KIND_COLORS[damage.HOUSE]),
                  df, "house_type", "Type of house")
        with q2:
            st.markdown("**Part of the house broken**")
            _plot(dc.ranked_bars(damage.exploded_counts(house, "Rooms Damaged"),
                                 "Rooms Damaged", color=dc.KIND_COLORS[damage.HOUSE]),
                  df, "rooms", "Part of the house broken")
            st.caption("Kitchens and bakhari (grain stores) are where food is kept.")
        with q3:
            lit = house["Light On"].dropna()
            st.markdown("**Light on at the time**")
            if len(lit):
                st.metric("Houses with a light on", f"{int(lit.sum())} of {len(lit)}",
                          delta=f"{lit.mean():.0%}", **note)
                st.caption("Whether a light was on in or outside the house when "
                           "the elephant came.")

    # -- Survey timeliness -----------------------------------------------------
    section("clock", "Survey timeliness",
            "Days from the damage to the survey visit. Late visits lose evidence and "
            "delay the claim.")
    _plot(dc.by_kind_bars(damage.survey_lag(df), "Delay", damage.LAG_LABELS,
                          x_title="Days from damage to survey"),
          df, "lag", "Days from damage to survey")

    st.warning(
        "Owner names, phone numbers, photos, surveyor emails and remark text are "
        "dropped as the export is read; downloads carry only the Epicollect5 entry ID."
    )
