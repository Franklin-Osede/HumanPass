import os

import cv2
import numpy as np
import pytest

from app.pipeline import ALIGNED_SIZE, Pipeline, PipelineRejection, ReasonCode, detect_single_face
from conftest import encode, requires_models


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces
        self.input_size = None

    def setInputSize(self, size):
        self.input_size = size

    def detect(self, img):
        return 1, self.faces


def face_row(score=0.95, w=150.0, h=160.0) -> np.ndarray:
    row = np.zeros(15, np.float32)
    row[2], row[3], row[14] = w, h, score
    return row


def reason(fn, *args) -> ReasonCode:
    with pytest.raises(PipelineRejection) as exc:
        fn(*args)
    return exc.value.reason_code


IMG = np.zeros((480, 640, 3), np.uint8)


def test_no_face_when_detector_returns_none():
    assert reason(detect_single_face, IMG, FakeDetector(None)) == ReasonCode.NO_FACE


def test_no_face_when_detector_returns_empty():
    assert reason(detect_single_face, IMG, FakeDetector(np.zeros((0, 15), np.float32))) == ReasonCode.NO_FACE


def test_multiple_faces():
    faces = np.stack([face_row(), face_row()])
    assert reason(detect_single_face, IMG, FakeDetector(faces)) == ReasonCode.MULTIPLE_FACES


def test_single_face_and_input_size():
    detector = FakeDetector(np.stack([face_row(score=0.93, w=150, h=160)]))
    detection = detect_single_face(IMG, detector)
    assert detector.input_size == (640, 480)
    assert (detection.score, detection.width, detection.height) == pytest.approx((0.93, 150, 160))


# --- real models (skipped when the ONNX files are absent) ---


@requires_models
@pytest.mark.parametrize(
    "img",
    [np.full((480, 640, 3), 128, np.uint8), np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)],
    ids=["blank", "noise"],
)
def test_real_models_no_face(settings, img):
    assert reason(Pipeline(settings).run, encode(img)) == ReasonCode.NO_FACE


SAMPLE_FACE = os.environ.get("HUMANPASS_SAMPLE_FACE")
requires_sample = pytest.mark.skipif(not SAMPLE_FACE, reason="HUMANPASS_SAMPLE_FACE not set")


@requires_models
@requires_sample
def test_sample_face_passes_and_aligns(settings):
    with open(SAMPLE_FACE, "rb") as f:
        face = Pipeline(settings).run(f.read())
    assert face.crop.shape == (ALIGNED_SIZE, ALIGNED_SIZE, 3)
    assert face.detection_score >= settings.min_detection_score


@requires_models
@requires_sample
def test_sample_face_twice_is_multiple_faces(settings):
    img = cv2.imread(SAMPLE_FACE, cv2.IMREAD_COLOR)
    assert reason(Pipeline(settings).run, encode(np.hstack([img, img]))) == ReasonCode.MULTIPLE_FACES
