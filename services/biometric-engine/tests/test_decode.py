import numpy as np
import pytest

from app.pipeline import PipelineRejection, ReasonCode, decode_image
from conftest import encode, make_settings


def rejection(data, settings) -> ReasonCode:
    with pytest.raises(PipelineRejection) as exc:
        decode_image(data, settings)
    return exc.value.reason_code


@pytest.mark.parametrize(
    "data",
    [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, b"\xff\xd8\xff" + b"\x00" * 64],
    ids=["empty", "garbage", "png-magic-only", "jpeg-magic-only"],
)
def test_undecodable_is_invalid_image(data, settings):
    assert rejection(data, settings) == ReasonCode.INVALID_IMAGE


def test_truncated_png_is_invalid_image(settings):
    data = encode(np.full((50, 50, 3), 128, np.uint8))
    assert rejection(data[: len(data) // 3], settings) == ReasonCode.INVALID_IMAGE


def test_valid_png_and_jpeg_decode_to_bgr(settings):
    img = np.full((20, 30, 3), 100, np.uint8)
    for ext in (".png", ".jpg"):
        out = decode_image(encode(img, ext), settings)
        assert out.shape == (20, 30, 3)


def test_grayscale_decodes_to_three_channels(settings):
    out = decode_image(encode(np.full((20, 20), 100, np.uint8)), settings)
    assert out.shape == (20, 20, 3)


def test_pixel_limit_is_exclusive():
    data = encode(np.zeros((20, 20, 3), np.uint8))  # 400 px
    assert decode_image(data, make_settings(max_image_pixels=400)).shape == (20, 20, 3)
    assert rejection(data, make_settings(max_image_pixels=399)) == ReasonCode.IMAGE_TOO_LARGE


def test_default_limit_rejects_just_over_40_megapixels(settings):
    data = encode(np.zeros((5001, 8000), np.uint8))  # 40,008,000 px
    assert rejection(data, settings) == ReasonCode.IMAGE_TOO_LARGE


@pytest.mark.parametrize(
    "shape, expected",
    [((1000, 2000), (640, 1280)), ((2000, 1000), (1280, 640)), ((1280, 1280), (1280, 1280)), ((300, 400), (300, 400))],
)
def test_downscale_longest_side_to_1280(settings, shape, expected):
    out = decode_image(encode(np.zeros((*shape, 3), np.uint8)), settings)
    assert out.shape[:2] == expected
