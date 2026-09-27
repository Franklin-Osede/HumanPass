# syntax=docker/dockerfile:1

# ---- models: download the ONNX models and verify their sha256 (never committed) ----
FROM python:3.12-slim AS models
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /models
ARG MODEL_BASE_URL=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
COPY models.sha256 ./
# sha256sum -c exits non-zero on any mismatch, which fails the build.
RUN curl -fsSL --retry 3 -o face_detection_yunet_2023mar.onnx \
      "$MODEL_BASE_URL/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
 && curl -fsSL --retry 3 -o face_recognition_sface_2021dec.onnx \
      "$MODEL_BASE_URL/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
 && sha256sum -c models.sha256

# ---- runtime ----
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELS_DIR=/models
RUN groupadd --system --gid 10001 engine \
 && useradd --system --uid 10001 --gid engine --no-create-home --shell /usr/sbin/nologin engine
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY --from=models /models/*.onnx /models/
COPY app ./app
USER engine
EXPOSE 8000
# No access log: the request line carries the candidate subjectId.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--no-server-header"]

# ---- test ----
FROM runtime AS test
USER root
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY pyproject.toml ./
COPY tests ./tests
USER engine
CMD ["python", "-m", "pytest", "-p", "no:cacheprovider"]
