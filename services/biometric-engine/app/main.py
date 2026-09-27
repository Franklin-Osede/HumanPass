"""biometric-engine HTTP surface. Internal only: no published port, no docs/openapi routes."""

import asyncio
import hmac
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .pipeline import Pipeline, PipelineRejection

log = logging.getLogger("humanpass.engine")

SUBJECT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
TOKEN_EXEMPT_PATHS = frozenset({"/healthz"})


class ServiceTokenMiddleware:
    """Default deny: every path except /healthz requires a valid X-Service-Token."""

    def __init__(self, app, token: str):
        self.app = app
        self._token = token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] not in TOKEN_EXEMPT_PATHS:
            given = dict(scope["headers"]).get(b"x-service-token", b"")
            if not hmac.compare_digest(given, self._token):
                await JSONResponse({"reasonCode": "UNAUTHORIZED"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


class BodyTooLarge(Exception):
    pass


async def read_body_limited(request: Request, limit: int) -> bytearray:
    """Read the raw body into memory, aborting as soon as it exceeds `limit` bytes.

    Raw octet-stream (not multipart/UploadFile), so Starlette never spools it to disk.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise BodyTooLarge
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            del body
            raise BodyTooLarge
    return body


def _error(status: int, reason_code: str, processing_id: uuid.UUID | None = None) -> JSONResponse:
    content = {"reasonCode": reason_code}
    if processing_id is not None:
        content["processingId"] = str(processing_id)
    return JSONResponse(content, status_code=status)


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = settings or Settings()
    pipeline = pipeline or Pipeline(settings)
    executor = ThreadPoolExecutor(max_workers=settings.worker_threads, thread_name_prefix="pipeline")

    @asynccontextmanager
    async def lifespan(_app):
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(ServiceTokenMiddleware, token=settings.service_token.get_secret_value())

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/internal/v1/enrollments")
    async def enroll(request: Request):
        processing_id = uuid.uuid4()
        started = time.perf_counter()

        subject_id = request.query_params.get("subjectId")
        if subject_id is None or not SUBJECT_ID_RE.fullmatch(subject_id):
            return _error(400, "INVALID_SUBJECT_ID")
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != "application/octet-stream":
            return _error(415, "UNSUPPORTED_MEDIA_TYPE")
        try:
            body = await read_body_limited(request, settings.max_body_bytes)
        except BodyTooLarge:
            return _error(413, "PAYLOAD_TOO_LARGE")

        try:
            face = await asyncio.get_running_loop().run_in_executor(executor, pipeline.run, body)
        except PipelineRejection as rejection:
            latency_ms = round((time.perf_counter() - started) * 1000)
            log.info("enrollment processing_id=%s outcome=REJECTED reason=%s latency_ms=%d",
                     processing_id, rejection.reason_code, latency_ms)
            return _error(422, rejection.reason_code.value, processing_id)
        finally:
            del body
        del face

        # Milestone 3 scaffold: embedding (M4) and decision/persistence (M5) are not implemented yet.
        log.info("enrollment processing_id=%s outcome=NOT_IMPLEMENTED", processing_id)
        return _error(501, "NOT_IMPLEMENTED", processing_id)

    return app
