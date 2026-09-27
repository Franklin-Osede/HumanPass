import cv2
import numpy as np
import pytest

from app.config import Settings

TOKEN = "t" * 64


def make_settings(**overrides) -> Settings:
    return Settings(service_token=TOKEN, **overrides)


def encode(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def models_present() -> bool:
    s = make_settings()
    return s.yunet_model_path.is_file() and s.sface_model_path.is_file()


requires_models = pytest.mark.skipif(not models_present(), reason="ONNX models not present")


@pytest.fixture
def settings() -> Settings:
    return make_settings()
