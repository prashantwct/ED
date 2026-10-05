"""Syncing the Epicollect5 damage projects, against a local stand-in for the API.

The stand-in serves a project definition and paged entries the way the
export API documents them, honours ``filter_by=uploaded_at``, answers
private projects with the live API's 404, and counts requests and
tokens. These check what would go wrong quietly: a page never read, a
re-sync that pulls the whole project again, a token per pull when the
API issues ten an hour, or one private project hiding the other.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from core import damage, epicollect
from damage_fixtures import NOWHERE, crop_entry, house_entry

FORM = {"ref": "form_1", "inputs": []}


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
        "herd-crop-damage": ("Herd Crop Damage", FORM, [
            crop_entry(i, f"2026-09-01T00:00:0{i}.000Z") for i in range(5)
        ] + [crop_entry(9, "2026-09-02T00:00:00.000Z", location=NOWHERE)]),
        "herd-house-damage": ("Herd House Damage", FORM, [house_entry(1)]),
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    epicollect.reset()


def _fetch(api, **kwargs):
    return epicollect.fetch_entries(base_url=api, interval=0, **kwargs)


def _counts(result):
    return {slug: len(entries) for slug, entries in result.entries.items()}


def test_reads_every_page_of_both_projects(api):
    result = _fetch(api)
    assert _counts(result) == {"herd-crop-damage": 6, "herd-house-damage": 1}
    pages = [q["page"] for path, q in Api.log if path.endswith("herd-crop-damage")
             and "entries" in path]
    assert pages == ["1", "2"]  # six entries at three a page


def test_api_entries_read_like_the_csv_export(api):
    """The fetch and the upload end in the same de-identified frame."""
    result = _fetch(api)
    tables = []
    for slug, entries in result.entries.items():
        table = damage.entries_frame(entries)
        table.attrs["source"] = slug
        tables.append(table)
    frame, _ = damage.tidy(tables)

    assert sorted(frame["Kind"].value_counts().items()) == [("Crop", 6), ("House", 1)]
    crop = frame[frame["Entry"] == "crop-0"].iloc[0]
    assert crop["Estimated Loss"] == 5000 and crop["Hour"] == 22
    assert crop["Beat"] == "Kaseru"
    assert frame[frame["Entry"] == "crop-9"].iloc[0]["Beat"] == "Unknown"
    flat = frame.to_csv()
    assert "Owner 0" not in flat and "9000000000" not in flat


def test_resync_asks_only_for_what_was_uploaded_since(api):
    _fetch(api)
    Api.log = []
    Api.projects["herd-crop-damage"][2].append(
        crop_entry(20, "2026-09-03T00:00:00.000Z")
    )
    result = _fetch(api, force=True)

    entry_calls = [q for path, q in Api.log if "/entries/" in path]
    assert all(q.get("filter_by") == "uploaded_at" for q in entry_calls)
    assert not [path for path, _ in Api.log if "/project/" in path]  # form kept
    assert _counts(result)["herd-crop-damage"] == 7  # nothing lost or doubled


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
        epicollect.fetch_entries(["no-such-project"], base_url=api, interval=0)


# --- Private projects -------------------------------------------------------
def test_credential_names_follow_the_slug():
    assert epicollect.credential_names("herd-crop-damage") == (
        "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID",
        "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_SECRET",
    )


def test_private_project_without_credentials_does_not_block_the_other(api):
    Api.private["herd-crop-damage"] = "crop-token"
    result = _fetch(api)

    assert _counts(result) == {"herd-house-damage": 1}
    message = result.errors["herd-crop-damage"]
    assert "Access denied." in message
    assert "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID" in message
    assert Api.token_requests == 0  # nothing configured, nothing asked


def test_each_private_project_uses_its_own_client_app(api, monkeypatch):
    Api.private = {"herd-crop-damage": "crop-token", "herd-house-damage": "house-token"}
    Api.clients = {"11": ("crop-secret", "crop-token"), "22": ("house-secret", "house-token")}
    crop_id, crop_secret = epicollect.credential_names("herd-crop-damage")
    house_id, house_secret = epicollect.credential_names("herd-house-damage")
    monkeypatch.setenv(crop_id, "11")
    monkeypatch.setenv(crop_secret, "crop-secret")
    monkeypatch.setenv(house_id, "22")
    monkeypatch.setenv(house_secret, "house-secret")

    result = _fetch(api)
    assert result.errors == {}
    assert _counts(result) == {"herd-crop-damage": 6, "herd-house-damage": 1}
    assert Api.token_requests == 2

    # The tokens last two hours and the API issues ten an hour: reused.
    _fetch(api, force=True)
    assert Api.token_requests == 2


def test_shared_credentials_are_the_fallback(api, monkeypatch):
    Api.private["herd-crop-damage"] = "crop-token"
    Api.clients = {"11": ("crop-secret", "crop-token")}
    monkeypatch.setenv("EPICOLLECT_CLIENT_ID", "11")
    monkeypatch.setenv("EPICOLLECT_CLIENT_SECRET", "crop-secret")

    result = _fetch(api)
    assert result.errors == {}
    assert _counts(result)["herd-crop-damage"] == 6


def test_rejected_credentials_say_which_project(api, monkeypatch):
    Api.private["herd-crop-damage"] = "crop-token"
    crop_id, crop_secret = epicollect.credential_names("herd-crop-damage")
    monkeypatch.setenv(crop_id, "11")
    monkeypatch.setenv(crop_secret, "wrong")

    result = _fetch(api)
    assert "refused" in result.errors["herd-crop-damage"]
    assert crop_id in result.errors["herd-crop-damage"]
    assert _counts(result) == {"herd-house-damage": 1}


def test_everything_private_raises_with_each_reason(api):
    Api.private = {"herd-crop-damage": "a", "herd-house-damage": "b"}
    with pytest.raises(epicollect.EpicollectError) as caught:
        epicollect.fetch_entries(base_url=api, interval=0)
    assert "EPICOLLECT_HERD_CROP_DAMAGE_CLIENT_ID" in str(caught.value)
    assert "EPICOLLECT_HERD_HOUSE_DAMAGE_CLIENT_ID" in str(caught.value)
