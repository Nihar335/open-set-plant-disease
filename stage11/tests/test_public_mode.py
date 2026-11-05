"""Public-hosting behaviour of the server. No model is loaded: the engine is replaced by a stub."""
from __future__ import annotations

import io
from collections import defaultdict

import pytest
from PIL import Image

from stage11 import server


class _StubEngine:
    def diagnose(self, data, plant, state, place):
        return {"verdict": "unknown"}


def _form():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(buf, format="PNG")
    buf.seek(0)
    return {"image": (buf, "leaf.png"), "plant": "tomato", "state": "Karnataka"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_engine", _StubEngine())
    monkeypatch.setattr(server, "_hits", defaultdict(server.deque))
    return server.app.test_client()


def _post(client, ip="1.1.1.1"):
    return client.post("/api/diagnose", data=_form(), content_type="multipart/form-data",
                       headers={"X-Forwarded-For": ip})


def test_local_mode_has_no_limits_and_is_not_marked_hosted(client, monkeypatch):
    monkeypatch.setattr(server, "PUBLIC", False)
    monkeypatch.setattr(server, "PER_VISITOR", (1, 600))
    assert client.get("/api/meta").json["hosted"] is False
    assert [_post(client).status_code for _ in range(3)] == [200, 200, 200]


def test_public_mode_is_marked_hosted(client, monkeypatch):
    monkeypatch.setattr(server, "PUBLIC", True)
    assert client.get("/api/meta").json["hosted"] is True


def test_visitor_limit_applies_per_visitor(client, monkeypatch):
    monkeypatch.setattr(server, "PUBLIC", True)
    monkeypatch.setattr(server, "PER_VISITOR", (2, 600))
    assert [_post(client).status_code for _ in range(3)] == [200, 200, 429]
    assert _post(client, ip="2.2.2.2").status_code == 200


def test_overall_limit_applies_across_visitors(client, monkeypatch):
    monkeypatch.setattr(server, "PUBLIC", True)
    monkeypatch.setattr(server, "ALL_VISITORS", (2, 3600))
    assert [_post(client, ip=f"3.3.3.{i}").status_code for i in range(3)] == [200, 200, 429]


def test_busy_server_answers_503_and_a_slot_is_released_afterwards(client, monkeypatch):
    monkeypatch.setattr(server, "PUBLIC", True)
    monkeypatch.setattr(server, "MAX_WAIT_S", 0)
    slots = server.threading.BoundedSemaphore(1)
    monkeypatch.setattr(server, "_slots", slots)
    assert slots.acquire(blocking=False)
    assert _post(client).status_code == 503
    slots.release()
    assert _post(client).status_code == 200
    assert slots.acquire(blocking=False)  # the finished request gave its slot back


def test_page_has_the_hosted_notice_and_privacy_targets(client):
    html = client.get("/").get_data(as_text=True)
    for el in ('id="demoNotice"', 'id="privacyNote"', 'id="privacyHow"', "m.hosted"):
        assert el in html
