"""Biometric pipeline: decode → detect exactly one face → align → quality gates.

Images exist only in memory. Nothing here writes to disk or logs image content, and
references are dropped (`del`) as soon as a stage no longer needs them.
"""

import threading
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from .config import Settings

ALIGNED_SIZE = 112


class ReasonCode(StrEnum):
    INVALID_IMAGE = "INVALID_IMAGE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    TOO_DARK = "TOO_DARK"
    TOO_BRIGHT = "TOO_BRIGHT"
    BLURRY = "BLURRY"


class PipelineRejection(Exception):
    """The sample is unusable. Carries only a reason code, never image data."""

    def __init__(self, reason_code: ReasonCode):
        super().__init__(reason_code.value)
        self.reason_code = reason_code


@dataclass(frozen=True)
class Detection:
    score: float
    width: float
    height: float
    row: np.ndarray  # raw YuNet row: x, y, w, h, 5 landmarks (x, y), score


@dataclass
class AlignedFace:
    crop: np.ndarray  # ALIGNED_SIZE × ALIGNED_SIZE BGR
    detection_score: float


def decode_image(data: bytes | bytearray, settings: Settings) -> np.ndarray:
    """Decode to BGR, reject oversized images, downscale the longest side to max_image_side."""
    buf = np.frombuffer(data, dtype=np.uint8)
    try:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except cv2.error:
        img = None
    del buf
    if img is None:
        raise PipelineRejection(ReasonCode.INVALID_IMAGE)

    height, width = img.shape[:2]
    if height * width > settings.max_image_pixels:
        del img
        raise PipelineRejection(ReasonCode.IMAGE_TOO_LARGE)

    longest = max(height, width)
    if longest > settings.max_image_side:
        scale = settings.max_image_side / longest
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def detect_single_face(img: np.ndarray, detector) -> Detection:
    height, width = img.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(img)
    count = 0 if faces is None else len(faces)
    if count == 0:
        raise PipelineRejection(ReasonCode.NO_FACE)
    if count > 1:
        raise PipelineRejection(ReasonCode.MULTIPLE_FACES)
    row = faces[0]
    return Detection(score=float(row[14]), width=float(row[2]), height=float(row[3]), row=row)


def check_quality(detection: Detection, crop: np.ndarray, settings: Settings) -> None:
    """Quality gates, in spec order. Raises PipelineRejection on the first failure."""
    if detection.score < settings.min_detection_score:
        raise PipelineRejection(ReasonCode.LOW_CONFIDENCE)
    if min(detection.width, detection.height) < settings.min_face_size_px:
        raise PipelineRejection(ReasonCode.FACE_TOO_SMALL)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_gray = float(gray.mean())
    if mean_gray < settings.min_mean_gray:
        raise PipelineRejection(ReasonCode.TOO_DARK)
    if mean_gray > settings.max_mean_gray:
        raise PipelineRejection(ReasonCode.TOO_BRIGHT)
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < settings.min_laplacian_variance:
        raise PipelineRejection(ReasonCode.BLURRY)


class _ThreadModels(threading.local):
    """OpenCV model objects are stateful: one instance per thread.

    threading.local re-runs __init__ (with the same arguments) in each new thread.
    """

    def __init__(self, settings: Settings):
        self.detector = cv2.FaceDetectorYN.create(
            str(settings.yunet_model_path), "", (320, 320), settings.detector_score_threshold
        )
        # Loaded here only for alignCrop.
        self.recognizer = cv2.FaceRecognizerSF.create(str(settings.sface_model_path), "")


class Pipeline:
    def __init__(self, settings: Settings):
        for path in (settings.yunet_model_path, settings.sface_model_path):
            if not path.is_file():
                raise FileNotFoundError(f"model not found: {path}")
        self._settings = settings
        self._models = _ThreadModels(settings)  # load once now so a bad model fails at startup

    def run(self, data: bytes | bytearray) -> AlignedFace:
        """CPU-bound; call from a worker thread."""
        img = decode_image(data, self._settings)
        models = self._models
        detection = detect_single_face(img, models.detector)
        crop = models.recognizer.alignCrop(img, detection.row)
        del img
        check_quality(detection, crop, self._settings)
        return AlignedFace(crop=crop, detection_score=detection.score)
