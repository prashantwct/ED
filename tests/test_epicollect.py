"""The Epicollect5 damage-report source, against a local stand-in for the API.

The stand-in serves a project definition and paged entries the way the
export API documents them, honours ``filter_by=uploaded_at``, and counts
requests, so these check the things that would go wrong quietly: fields
mapped from the wrong question, a page never read, a re-sync that pulls
the whole project again, or a point placed in the wrong beat.
"""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from core import epicollect
from core.data_loader import load_and_validate_csv

# Inside the Baroundha beat polygon (Barondha range, Satna division).
INSIDE = {"latitude": 25.00213, "longitude": 80.62602, "accuracy": 5}
NOWHERE = {"latitude": 21.0, "longitude": 76.0, "accuracy": 5}

CROP_FORM = {
    "ref": "crop_form",
    "inputs": [
        {"ref": "a", "type": "date", "question": "Date of damage"},
        {"ref": "b", "type": "time", "question": "Time of damage"},
        {"ref": "c", "type": "location", "question": "Location of the field"},
        {"ref": "d", "type": "text", "question": "Village name"},
        {"ref": "e", "type": "integer", "question": "Number of elephants in the herd"},
        {"ref": "f", "type": "integer", "question": "Number of houses nearby"},
        {"ref": "g", "type": "radio", "question": "Was anyone injured?"},
    ],
}
HOUSE_FORM = {
    "ref": "house_form",
    "inputs": [
        {"ref": "a", "type": "location", "question": "Where"},
        {"ref": "b", "type": "dropdown", "question": "Division"},
        {"ref": "c", "type": "text", "question": "Range"},
        {"ref": "d", "type": "text", "question": "Beat"},
        {"ref": "e", "type": "integer", "question": "Number of people killed"},
    ],
}


def _crop_entry(i, uploaded, location=INSIDE):
    return {
        "ec5_uuid": f"crop-{i}",
        "created_at": "2025-11-30T10:00:00.000Z",
        "uploaded_at": uploaded,
        "title": f"crop {i}",
        "1_Date_of_damage": f"2025-11-{(i % 27) + 1:02d}T00:00:00.000",
        "2_Time_of_damage": "1970-01-01T21:40:00.000",
        "3_Location_of_the_fi": location,
        "4_Village_name": "Kirar",
        "5_Number_of_elephan": 4,
        "6_Number_of_houses_n": 12,
        "7_Was_anyone_injured": "No",
    }


class Api(BaseHTTPRequestHandler):
    projects = {}
    log = []
    # slug -> the bearer token that opens it; absent means public.
    private = {}
    # client_id -> (secret, token it is issued). A client app opens one project.
    clients = {}
    token_requests = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        Api.log.append((url.path, query))
        parts = url.path.strip("/").split("/")
        slug = parts[3] if len(parts) > 3 else ""
        needed = Api.private.get(slug)
        if needed and self.headers.get("Authorization") != f"Bearer {needed}":
            # What the live API answers for a private project.
            return self._json({"errors": [{"code": "ec5_77", "title": "Access denied."}]}, 404)
        if parts[:3] == ["api", "export", "project"] and parts[3] in Api.projects:
            name, form, _ = Api.projects[parts[3]]
            return self._json({"data": {"project": {"name": name, "forms": [form]}}})
        if parts[:3] == ["api", "export", "entries"] and parts[3] in Api.projects:
            entries = Api.projects[parts[3]][2]
            since = query.get("filter_from") if query.get("filter_by") == "uploaded_at" else None
            if since:
                entries = [e for e in entries if e["uploaded_at"] >= since]
            per_page = int(query.get("per_page", 50))
            page = int(query.get("page", 1))
            last = max(1, -(-len(entries) // per_page))
            chunk = entries[(page - 1) * per_page: page * per_page]
            return self._json({
                "data": {"id": parts[3], "type": "entries", "entries": chunk},
                "meta": {"total": len(entries), "per_page": per_page,
                         "current_page": page, "last_page": last},
            })
        self._json({"errors": [{"code": "ec5_11", "title": "Project does not exist"}]}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Api.token_requests += 1
        if urlparse(self.path).path != "/api/oauth/token" \
                or body.get("grant_type") != "client_credentials":
            return self._json({"errors": [{"code": "ec5_1"}]}, 400)
        secret, token = Api.clients.get(str(body.get("client_id")), (None, None))
        if secret is None or body.get("client_secret") != secret:
            return self._json({"errors": [{"code": "ec5_257", "title": "Bad client"}]}, 400)
        self._json({"token_type": "Bearer", "expires_in": 7200, "access_token": token})

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def api(monkeypatch):
    epicollect.reset()
    monkeypatch.setattr(epicollect, "PER_PAGE", 3)
    Api.log = []
    Api.private, Api.clients, Api.token_requests = {}, {}, 0
    for name in ("EPICOLLECT_CLIENT_ID", "EPICOLLECT_CLIENT_SECRET",
                 *epicollect.credential_names("herd-crop-damage"),
                 *epicollect.credential_names("herd-house-damage")):
        monkeypatch.delenv(name, raising=False)
    Api.projects = {
        "herd-crop-damage": ("Herd Crop Damage", CROP_FORM, [
            _crop_entry(i, f"2025-12-01T00:00:0{i}.000Z") for i in range(5)
        ] + [_crop_entry(9, "2025-12-02T00:00:00.000Z", location=NOWHERE)]),
        "herd-house-damage": ("Herd House Damage", HOUSE_FORM, [{
            "ec5_uuid": "house-1", "created_at": "2025-10-05T08:00:00.000Z",
            "uploaded_at": "2025-10-05T08:00:00.000Z", "title": "h",
            "1_Where": INSIDE, "2_Division": "Anuppur", "3_Range": "Kotma",
            "4_Beat": "Beat 7", "5_Number_of_people_ki": 1,
        }]),
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    epicollect.reset()


def _fetch(api, **kwargs):
    result = epicollect.fetch_damage_reports(base_url=api, interval=0, **kwargs)
    frame, warnings = load_and_validate_csv(io.BytesIO(result.data))
    return result, frame


def test_reads_every_page_of_both_projects(api):
    result, frame = _fetch(api)
    assert result.per_project == {"herd-crop-damage": 6, "herd-house-damage": 1}
    pages = [q["page"] for path, q in Api.log if path.endswith("herd-crop-damage")
             and "entries" in path]
    assert pages == ["1", "2"]  # six entries at three a page
    assert len(frame) == 7


def test_maps_fields_from_the_form_not_from_guessed_keys(api):
    _, frame = _fetch(api)
    crop = frame[frame["Source"] == "Epicollect5: Herd Crop Damage"].iloc[0]
    assert crop["Crop Damage"] == 1 and crop["House Damage"] == 0
    assert crop["Total Count"] == 4  # "number of elephants", not "houses nearby"
    assert crop["Hour"] == 21
    assert crop["Injury"] == 0  # "No"
    assert crop["Village (reported)"] == "Kirar"
    assert crop["Date"] == pd.Timestamp("2025-11-01")

    house = frame[frame["Source"] == "Epicollect5: Herd House Damage"].iloc[0]
    assert house["House Damage"] == 1
    assert house["Death"] == 1
    # The form's own division wins over the map.
    assert (house["Division"], house["Range"], house["Beat"]) == ("Anuppur", "Kotma", "Beat 7")
    # No date question: the entry's creation date stands in, with no hour.
    assert house["Date"] == pd.Timestamp("2025-10-05")
    assert pd.isna(house["Hour"])


def test_places_reports_without_a_beat_in_the_boundaries(api):
    _, frame = _fetch(api)
    crop = frame[frame["Source"].str.contains("Crop")]
    placed = crop[crop["Latitude"].round(3) == 25.002].iloc[0]
    assert (placed["Division"], placed["Range"], placed["Beat"]) == (
        "Satna", "Barondha", "Baroundha"
    )
    lost = crop[crop["Latitude"] == 21.0].iloc[0]
    assert lost["Beat"] == epicollect.OUTSIDE_BEATS
    assert lost["Division"] == epicollect.UNPLACED


def test_resync_asks_only_for_what_was_uploaded_since(api):
    _fetch(api)
    Api.log = []
    Api.projects["herd-crop-damage"][2].append(
        _crop_entry(20, "2025-12-03T00:00:00.000Z")
    )
    result, frame = _fetch(api, force=True)

    entry_calls = [q for path, q in Api.log if "/entries/" in path]
    assert all(q.get("filter_by") == "uploaded_at" for q in entry_calls)
    assert not [path for path, _ in Api.log if "/project/" in path]  # form kept
    assert result.per_project["herd-crop-damage"] == 7  # nothing lost or doubled


def test_recent_sync_is_reused_without_asking(api):
    _fetch(api)
    Api.log = []
    _fetch(api)
    assert Api.log == []


def test_requests_of_one_kind_are_spaced(api):
    waits = []
    client = epicollect._Client(api, interval=24.0, sleep=waits.append)
    client.get("project", "/api/export/project/herd-crop-damage")
    client.get("project", "/api/export/project/herd-house-damage")
    assert waits and 23 < waits[0] <= 24


def test_missing_project_explains_private_access(api):
    with pytest.raises(epicollect.EpicollectError, match="EPICOLLECT_NO_SUCH_PROJECT_CLIENT_ID"):
        epicollect.fetch_damage_reports(
            {"no-such-project": "Crop Damage"}, base_url=api, interval=0
        )


def test_app_runs_on_epicollect_reports_alone(api, monkeypatch):
    """The landing page's Epicollect5 button through to a ranked dashboard."""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    real = epicollect.fetch_damage_reports

    def against_stand_in(**kwargs):
        kwargs.update(base_url=api, interval=0)
        return real(**kwargs)

    monkeypatch.setattr(epicollect, "fetch_damage_reports", against_stand_in)
    app = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()

    button = next(b for b in at.button if "Epicollect5" in b.label)
    at = button.click().run()

    assert not at.exception, at.exception
    assert any("7 valid rows" in s.value for s in at.success)
    assert any("Beat priorities" in m.value for m in at.markdown)


# --- Private projects -------------------------------------------------------
def test_credential_names_follow_the_slug():
    assert epicollect.credential_names("herd-crop-damage") == (
        "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID",
        "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_SECRET",
    )


def test_private_project_without_credentials_does_not_block_the_other(api):
    Api.private["herd-crop-damage"] = "crop-token"
    result, frame = _fetch(api)

    assert result.per_project == {"herd-house-damage": 1}
    message = result.errors["herd-crop-damage"]
    assert "Access denied." in message
    assert "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID" in message
    assert Api.token_requests == 0  # nothing configured, nothing asked
    assert len(frame) == 1


def test_each_private_project_uses_its_own_client_app(api, monkeypatch):
    Api.private = {"herd-crop-damage": "crop-token", "herd-house-damage": "house-token"}
    Api.clients = {"11": ("crop-secret", "crop-token"), "22": ("house-secret", "house-token")}
    crop_id, crop_secret = epicollect.credential_names("herd-crop-damage")
    house_id, house_secret = epicollect.credential_names("herd-house-damage")
    monkeypatch.setenv(crop_id, "11")
    monkeypatch.setenv(crop_secret, "crop-secret")
    monkeypatch.setenv(house_id, "22")
    monkeypatch.setenv(house_secret, "house-secret")

    result, frame = _fetch(api)
    assert result.errors == {}
    assert result.per_project == {"herd-crop-damage": 6, "herd-house-damage": 1}
    assert Api.token_requests == 2

    # The tokens last two hours and the API issues ten an hour: reused.
    _fetch(api, force=True)
    assert Api.token_requests == 2


def test_shared_credentials_are_the_fallback(api, monkeypatch):
    Api.private["herd-crop-damage"] = "crop-token"
    Api.clients = {"11": ("crop-secret", "crop-token")}
    monkeypatch.setenv("EPICOLLECT_CLIENT_ID", "11")
    monkeypatch.setenv("EPICOLLECT_CLIENT_SECRET", "crop-secret")

    result, _ = _fetch(api)
    assert result.errors == {}
    assert result.per_project["herd-crop-damage"] == 6


def test_rejected_credentials_say_which_project(api, monkeypatch):
    Api.private["herd-crop-damage"] = "crop-token"
    crop_id, crop_secret = epicollect.credential_names("herd-crop-damage")
    monkeypatch.setenv(crop_id, "11")
    monkeypatch.setenv(crop_secret, "wrong")

    result, _ = _fetch(api)
    assert "refused" in result.errors["herd-crop-damage"]
    assert crop_id in result.errors["herd-crop-damage"]
    assert result.per_project == {"herd-house-damage": 1}


def test_everything_private_raises_with_each_reason(api):
    Api.private = {"herd-crop-damage": "a", "herd-house-damage": "b"}
    with pytest.raises(epicollect.EpicollectError) as caught:
        epicollect.fetch_damage_reports(base_url=api, interval=0)
    assert "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID" in str(caught.value)
    assert "EPICOLLECT_HERD_HOUSE_DAMAGE_CLIENT_ID" in str(caught.value)


def test_app_shows_the_private_project_and_loads_the_rest(api, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    Api.private["herd-crop-damage"] = "crop-token"
    real = epicollect.fetch_damage_reports

    def against_stand_in(**kwargs):
        kwargs.update(base_url=api, interval=0)
        return real(**kwargs)

    monkeypatch.setattr(epicollect, "fetch_damage_reports", against_stand_in)
    app = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    at = next(b for b in at.button if "Epicollect5" in b.label).click().run()

    assert not at.exception, at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "herd-crop-damage was not loaded" in warnings
    assert "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID" in warnings
    assert any("1 valid rows" in s.value for s in at.success)
