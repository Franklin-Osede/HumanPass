"""HTTP surface with a fake pipeline (no models, no DB)."""

import json
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline import AlignedFace, PipelineRejection, ReasonCode
from conftest import TOKEN, make_settings

URL = "/internal/v1/enrollments"
AUTH = {"X-Service-Token": TOKEN, "Content-Type": "application/octet-stream"}
MAX = 5 * 1024 * 1024


class FakePipeline:
    def __init__(self, reject: ReasonCode | None = None):
        self.reject = reject
        self.calls: list[bytes] = []

    def run(self, data):
        self.calls.append(bytes(data))
        if self.reject:
            raise PipelineRejection(self.reject)
        return AlignedFace(crop=np.zeros((112, 112, 3), np.uint8), detection_score=0.99)


@pytest.fixture
def fake():
    return FakePipeline()


@pytest.fixture
def client(fake):
    with TestClient(create_app(make_settings(), pipeline=fake)) as c:
        yield c


def assert_no_biometric_leak(response):
    text = response.text.lower()
    assert "embedding" not in text and "similarity" not in text

    def walk(value):
        if isinstance(value, list):
            assert not any(isinstance(v, float) for v in value), "float array in response"
            for v in value:
                walk(v)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)

    walk(json.loads(response.text))


# --- token ---
def test_healthz_needs_no_token(client):
    r = client.get("/healthz")
    assert r.status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {"X-Service-Token": TOKEN[:-1]}])
def test_token_required(client, fake, headers):
    r = client.post(f"{URL}?subjectId=sub_1", content=b"x",
                    headers={"Content-Type": "application/octet-stream", **headers})
    assert r.status_code == 401
    assert fake.calls == []


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_disabled(client, path):
    assert client.get(path, headers=AUTH).status_code == 404


# --- subjectId ---
@pytest.mark.parametrize("query", ["", "?subjectId=", "?subjectId=a%20b", "?subjectId=sub.1",
                                   "?subjectId=" + "a" * 65, "?subjectId=%C3%A9", "?subjectId=a%0A"])
def test_invalid_subject_id_is_400(client, fake, query):
    r = client.post(URL + query, content=b"x", headers=AUTH)
    assert r.status_code == 400
    assert r.json() == {"reasonCode": "INVALID_SUBJECT_ID"}
    assert fake.calls == []


def test_subject_id_of_64_chars_is_accepted(client):
    r = client.post(f"{URL}?subjectId={'a' * 64}", content=b"x", headers=AUTH)
    assert r.status_code == 501


# --- body ---
def test_multipart_is_rejected(client, fake):
    r = client.post(f"{URL}?subjectId=sub_1", files={"image": ("a.jpg", b"x")},
                    headers={"X-Service-Token": TOKEN})
    assert r.status_code == 415
    assert fake.calls == []


def test_exact_bytes_reach_pipeline(client, fake):
    payload = bytes(range(256)) * 10
    client.post(f"{URL}?subjectId=sub_1", content=payload, headers=AUTH)
    assert fake.calls == [payload]


def test_body_at_limit_is_accepted(client, fake):
    r = client.post(f"{URL}?subjectId=sub_1", content=b"\0" * MAX, headers=AUTH)
    assert r.status_code == 501
    assert len(fake.calls[0]) == MAX


def test_body_over_limit_is_413(client, fake):
    r = client.post(f"{URL}?subjectId=sub_1", content=b"\0" * (MAX + 1), headers=AUTH)
    assert r.status_code == 413
    assert fake.calls == []


def test_chunked_body_over_limit_is_413(client, fake):
    def chunks():  # no Content-Length: the streaming limit must catch it
        for _ in range(6):
            yield b"\0" * (1024 * 1024)

    r = client.post(f"{URL}?subjectId=sub_1", content=chunks(), headers=AUTH)
    assert r.status_code == 413
    assert fake.calls == []


# --- responses ---
@pytest.mark.parametrize("code", list(ReasonCode))
def test_rejection_maps_to_422(code):
    with TestClient(create_app(make_settings(), pipeline=FakePipeline(reject=code))) as c:
        r = c.post(f"{URL}?subjectId=sub_1", content=b"x", headers=AUTH)
    assert r.status_code == 422
    body = r.json()
    assert set(body) == {"reasonCode", "processingId"}
    assert body["reasonCode"] == code.value
    uuid.UUID(body["processingId"])
    assert_no_biometric_leak(r)


def test_valid_sample_is_not_implemented_yet(client):
    r = client.post(f"{URL}?subjectId=sub_1", content=b"x", headers=AUTH)
    assert r.status_code == 501
    assert r.json()["reasonCode"] == "NOT_IMPLEMENTED"
    assert_no_biometric_leak(r)


def test_processing_id_is_unique_per_call():
    with TestClient(create_app(make_settings(), pipeline=FakePipeline(reject=ReasonCode.NO_FACE))) as c:
        ids = {c.post(f"{URL}?subjectId=sub_1", content=b"x", headers=AUTH).json()["processingId"] for _ in range(5)}
    assert len(ids) == 5


def test_subject_id_and_body_not_logged(client, caplog):
    caplog.set_level("DEBUG")
    client.post(f"{URL}?subjectId=sub_secret_marker", content=b"IMAGEBYTESMARKER", headers=AUTH)
    # httpx is the test client itself (it logs its own request URL), not the engine.
    server_logs = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("httpx"))
    assert "processing_id=" in server_logs
    assert "sub_secret_marker" not in server_logs
    assert "IMAGEBYTESMARKER" not in server_logs
