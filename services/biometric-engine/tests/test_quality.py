"""Quality gates on synthetic crops. Each rejection test passes every earlier gate and fails
exactly one, so it fails if that gate is removed (the sample would pass or hit a later gate)."""

import cv2
import numpy as np
import pytest

from app.pipeline import Detection, PipelineRejection, ReasonCode, check_quality


def det(score=0.95, width=150.0, height=150.0) -> Detection:
    return Detection(score=score, width=width, height=height, row=np.zeros(15, np.float32))


def textured(mean=128, amplitude=60, seed=0) -> np.ndarray:
    """Sharp crop: uniform noise around `mean` (very high Laplacian variance)."""
    rng = np.random.default_rng(seed)
    return rng.integers(mean - amplitude, mean + amplitude + 1, (112, 112, 3)).astype(np.uint8)


def flat(value) -> np.ndarray:
    return np.full((112, 112, 3), value, np.uint8)


def checkerboard(low, high) -> np.ndarray:
    """Mean exactly (low + high) / 2, and sharp."""
    board = np.indices((112, 112)).sum(axis=0) % 2
    gray = np.where(board == 0, low, high).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def reason(detection, crop, settings) -> ReasonCode:
    with pytest.raises(PipelineRejection) as exc:
        check_quality(detection, crop, settings)
    return exc.value.reason_code


def test_good_sample_passes(settings):
    check_quality(det(), textured(), settings)


# --- LOW_CONFIDENCE ---
def test_low_confidence(settings):
    assert reason(det(score=0.89), textured(), settings) == ReasonCode.LOW_CONFIDENCE


def test_score_at_threshold_passes(settings):
    check_quality(det(score=0.9), textured(), settings)


# --- FACE_TOO_SMALL ---
@pytest.mark.parametrize("width, height", [(111, 200), (200, 111)])
def test_face_too_small_uses_min_side(settings, width, height):
    assert reason(det(width=width, height=height), textured(), settings) == ReasonCode.FACE_TOO_SMALL


def test_face_at_min_size_passes(settings):
    check_quality(det(width=112, height=112), textured(), settings)


# --- TOO_DARK / TOO_BRIGHT (flat crops are also blurry, so removing the gate yields BLURRY) ---
def test_too_dark(settings):
    assert reason(det(), flat(39), settings) == ReasonCode.TOO_DARK


def test_too_bright(settings):
    assert reason(det(), flat(221), settings) == ReasonCode.TOO_BRIGHT


def test_brightness_bounds_are_inclusive(settings):
    check_quality(det(), checkerboard(20, 60), settings)  # mean 40
    check_quality(det(), checkerboard(200, 240), settings)  # mean 220


def test_dark_just_below_bound(settings):
    assert reason(det(), checkerboard(20, 58), settings) == ReasonCode.TOO_DARK  # mean 39


# --- BLURRY ---
def test_flat_crop_is_blurry(settings):
    assert reason(det(), flat(128), settings) == ReasonCode.BLURRY


def test_blurred_texture_is_blurry(settings):
    crop = cv2.GaussianBlur(textured(), (0, 0), sigmaX=4)
    assert reason(det(), crop, settings) == ReasonCode.BLURRY


# --- order ---
def test_order_confidence_before_size_before_brightness(settings):
    assert reason(det(score=0.5, width=50, height=50), flat(10), settings) == ReasonCode.LOW_CONFIDENCE
    assert reason(det(width=50, height=50), flat(10), settings) == ReasonCode.FACE_TOO_SMALL


def test_thresholds_come_from_settings():
    from conftest import make_settings

    lenient = make_settings(min_detection_score=0.1, min_face_size_px=10, min_mean_gray=0,
                            max_mean_gray=255, min_laplacian_variance=0)
    check_quality(det(score=0.2, width=20, height=20), flat(5), lenient)
