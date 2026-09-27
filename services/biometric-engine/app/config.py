"""Engine settings, read from environment variables (field name, case-insensitive).

PROVISIONAL: every biometric threshold below comes from an OpenCV reference or was chosen
by hand. None has been calibrated on a representative dataset (see docs/model-card.md).
"""

import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    # Shared secret expected in X-Service-Token (env: SERVICE_TOKEN).
    service_token: SecretStr = Field(min_length=32)

    models_dir: Path = Path("/models")
    yunet_model_file: str = "face_detection_yunet_2023mar.onnx"
    sface_model_file: str = "face_recognition_sface_2021dec.onnx"

    # Bounded pool for CPU work; each thread holds its own OpenCV model instances.
    worker_threads: int = Field(default_factory=lambda: min(4, os.cpu_count() or 1), ge=1)

    max_body_bytes: int = Field(default=5 * 1024 * 1024, gt=0)

    # --- PROVISIONAL thresholds ---
    max_image_pixels: int = Field(default=40_000_000, gt=0)  # > → IMAGE_TOO_LARGE
    max_image_side: int = Field(default=1280, gt=0)  # longest side after downscaling
    detector_score_threshold: float = 0.7  # YuNet candidate threshold
    min_detection_score: float = 0.9  # < → LOW_CONFIDENCE
    min_face_size_px: int = 112  # min(w, h) < → FACE_TOO_SMALL
    min_mean_gray: float = 40.0  # < → TOO_DARK
    max_mean_gray: float = 220.0  # > → TOO_BRIGHT
    min_laplacian_variance: float = 50.0  # < → BLURRY

    @property
    def yunet_model_path(self) -> Path:
        return self.models_dir / self.yunet_model_file

    @property
    def sface_model_path(self) -> Path:
        return self.models_dir / self.sface_model_file
