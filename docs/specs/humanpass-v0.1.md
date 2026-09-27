# HumanPass v0.1 — Specification and implementation prompt

> Save this document as `docs/specs/humanpass-v0.1.md` in Milestone 1.
> Re-read it at the start of every milestone. It is the source of truth.

## Goal
Build a monorepo named `humanpass`. When it's done:
- `docker compose up` starts everything from scratch.
- `POST /api/v1/enroll` (multipart, field `image`) returns `{ enrollmentId, model: {name, version}, duplicate }`.

Positioning: "HumanPass helps platforms enforce one-human-one-account without requiring government ID." It verifies uniqueness within a domain, not identity. Never claim "identity verified".

OUT OF SCOPE (do not build): liveness, /verify, Angular/UI, passkeys, credentials, blockchain, Redis, Kubernetes, Terraform, observability stack.

## Architecture (frozen)
Client → identity-api (NestJS 11, TypeScript) → biometric-engine (Python 3.12, FastAPI, OpenCV) → PostgreSQL 16 + pgvector (image `pgvector/pgvector:pg16`).

- The embedding NEVER leaves biometric-engine. It is not returned to NestJS, to the client, or written to logs.
- Uploaded images and any image derived from them (crops, aligned faces, debug dumps) are NEVER written to any filesystem, object storage or log.

## Models (commercial-friendly licenses)
- Detection: YuNet `face_detection_yunet_2023mar.onnx` (MIT)
  sha256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Embedding: SFace `face_recognition_sface_2021dec.onnx` (Apache-2.0), 128-d
  sha256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
- Download both at Docker build time from `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/...` and verify with `sha256sum -c`. Never commit them.
- Runtime: `opencv-python-headless==4.10.0.84` with `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF` (`alignCrop` + `feature`).
- Store with every template: `model_name=opencv-sface`, `model_version=2021dec`, `embedding_dimension=128`.
- Do NOT use InsightFace (its pretrained models are non-commercial).

## Identifiers (semantics are part of the spec)
| Identifier | Meaning | Created when |
|---|---|---|
| `enrollment_id` (ULID) | One enrollment **attempt**; returned to the client | Every request |
| `subject_id` (`sub_<32 hex>`) | One **enrolled subject** (a unique human in this domain) | A candidate is generated per request and sent to the engine, but it only becomes a subject if the outcome is ENROLLED |
| `processing_id` (UUID) | One biometric execution; used in security_events and logs | Every engine call |

For DUPLICATE and REJECTED, no new subject exists: `identity.enrollments.subject_id` is NULL and the engine persists no profile. Document this in the privacy model.

## Database (db/init/00-roles.sh, runs on an empty volume)
- Create two LOGIN roles, with passwords taken from env:
  - `identity_api` → schema `identity` only
  - `biometric_engine` → schema `biometric` only
- `REVOKE ALL ON SCHEMA public FROM PUBLIC`.
- Revoke cross-schema access explicitly in both directions.
- Run `CREATE EXTENSION vector SCHEMA biometric`, so the identity role cannot even reference the vector type.
- Set up `ALTER DEFAULT PRIVILEGES`:
  - identity: SELECT/INSERT/UPDATE on tables
  - biometric: SELECT/INSERT/UPDATE/DELETE on tables (DELETE is for erasure) **plus USAGE on sequences**
- Set each role's `search_path` to its own schema.
- Migrations run as the superuser in one-shot compose services (`biometric-migrate`, `identity-migrate`). No service uses the superuser at runtime. Use a forward-only SQL migrator with a `<schema>.schema_migrations` table, revoked from the runtime role.
- Postgres publishes NO ports.

Tables:
- `biometric.biometric_profiles`: `id uuid pk`, `subject_id text unique not null`, `embedding biometric.vector(128)`, `model_name`, `model_version`, `embedding_dimension int check (=128)`, `enrolled_at`, `revoked_at null`.
  - NO HNSW/IVFFlat index (see ADR-005). Add only a partial btree on `(model_name, model_version) where revoked_at is null`.
- `biometric.security_events`: `processing_id uuid`, `decision` (ENROLLED|DUPLICATE|REJECTED), `reason_code`, `similarity real`, `model_name`, `model_version`, `latency_ms`, `created_at`.
  - It must have NO `subject_id` and NO embedding.
- `identity.enrollments`: `id text pk` (ULID), `subject_id text null`, `status` (ENROLLED|DUPLICATE|REJECTED), `reason_code`, `model_name`, `model_version`, `created_at`.
  - Add `CHECK ((status = 'ENROLLED') = (subject_id IS NOT NULL))`.
  - Add a unique index on `subject_id WHERE subject_id IS NOT NULL`.

## biometric-engine
- Internal endpoint: `POST /internal/v1/enrollments?subjectId=...`.
  - The body is **raw `application/octet-stream`, NOT multipart**, because Starlette spools multipart uploads over 1 MB to a temp file on disk.
  - Read the body with a streaming size limit of 5 MB and return 413 over the limit.
  - Validate `subjectId` against `^[A-Za-z0-9_-]{1,64}$`.
- Require header `X-Service-Token`, compared with `hmac.compare_digest`. `/healthz` is exempt.
- Publish no ports. Disable the docs/openapi routes.

Pipeline:
1. Decode with `cv2.imdecode`: invalid → `INVALID_IMAGE`; more than 40 MP → `IMAGE_TOO_LARGE`; downscale the longest side to 1280.
2. Run YuNet with score threshold 0.7: 0 faces → `NO_FACE`, more than 1 → `MULTIPLE_FACES`.
3. `alignCrop` to 112×112.
4. Quality gates, in this order:
   - detection score < 0.9 → `LOW_CONFIDENCE`
   - min(w, h) < 112 px → `FACE_TOO_SMALL`
   - mean gray of the aligned crop < 40 → `TOO_DARK`, > 220 → `TOO_BRIGHT`
   - Laplacian variance of the aligned crop < 50 → `BLURRY`
5. Compute the SFace feature and L2-normalise it (reject zero or non-finite vectors).
- OpenCV model objects are stateful, so use one instance per thread (`threading.local`).
- Run CPU work in a threadpool. `del` buffers as soon as possible.
- All thresholds come from env via pydantic-settings and are marked PROVISIONAL.

`DecisionPolicy`:
- Separate from the pipeline: it takes the best cosine similarity (or None) and returns `DUPLICATE` / `NOT_DUPLICATE`.
- Threshold 0.363 (OpenCV's SFace reference), `>=` means duplicate.

Two execution paths (do not mix them):

```
QUALITY REJECTION (no lock, no transaction around the pipeline)
  pipeline → PipelineRejection → security_event(REJECTED, reason_code) → 422

VALID BIOMETRIC SAMPLE
  pipeline (outside the lock) → BEGIN
    → pg_advisory_xact_lock(hashtext('enroll:'||model_name||':'||model_version))
    → exact NN search → DecisionPolicy → INSERT profile only if NOT_DUPLICATE
    → INSERT security_event(ENROLLED|DUPLICATE, similarity)
  → COMMIT → 200
```

- Exact NN query: `SELECT 1 - (embedding <=> $v::vector) ... WHERE same model_name AND model_version AND revoked_at IS NULL ORDER BY embedding <=> $v LIMIT 1`.
- Duplicates are never persisted.
- Send the vector as a pgvector text literal `[a,b,...]`.
- Use psycopg 3 + psycopg-pool with `options=-c search_path=biometric`.

Responses:
- 200 `{processingId, decision: ENROLLED|DUPLICATE, duplicate, model}`. Never include the embedding or the similarity.
- 422 `{reasonCode, processingId}`.

## identity-api (NestJS 11)
- `POST /api/v1/enroll`:
  - `FileInterceptor('image')` with `memoryStorage()` and limits `{fileSize: 5MB, files: 1, fields: 0}`.
  - Validate JPEG/PNG by **magic bytes**, not the declared content type.
  - Generate a ULID (write a small generator; it must encode 1469918176385 as `01ARYZ6S41`) and a candidate `subject_id`.
  - Forward the raw bytes to the engine with fetch and `AbortSignal.timeout`.
  - Persist the enrollment row, with `subject_id` only when ENROLLED.
  - Zero the upload buffer with `buffer.fill(0)` in a `finally` block.

Status mapping:

| Result | HTTP | Body |
|---|---|---|
| Enrolled | 201 | `{enrollmentId, model, duplicate:false}` |
| Duplicate | 409 | `{enrollmentId, duplicate:true}` (never who matched) |
| Rejected | 422 | `{enrollmentId, reasonCode}` |
| No file | 400 | `IMAGE_REQUIRED` |
| Too large | 413 | |
| Wrong type | 415 | `UNSUPPORTED_IMAGE_TYPE` |
| Engine down or unexpected status | 503 | `BIOMETRIC_ENGINE_UNAVAILABLE` |

- `/healthz` checks the DB.
- Use `pg` directly (no ORM) with `search_path=identity`.

## docker-compose
- Services: `postgres`, `biometric-migrate`, `identity-migrate`, `biometric-engine`, `identity-api`, plus test-profile services `biometric-tests` and `identity-tests`.
- Networks:
  - `backend` is `internal: true`.
  - identity-api is also on `edge` and is the ONLY service with a published port (`127.0.0.1:3000:3000`).
- Engine and API run with `read_only: true`, `tmpfs: [/tmp]`, `no-new-privileges` and non-root users.
- Ordering: `depends_on` with `service_healthy` / `service_completed_successfully`.
- Secrets come from `.env`. `scripts/init-env.sh` generates random hex values (they are embedded in URLs, so hex only).
- Dockerfiles are multi-stage:
  - engine: `models` → `runtime` → `test`
  - api: `build` → `runtime` → `test`

## Tests
Engine (pytest):
- Unit tests for the policy: at threshold → dup, just below → not dup, None → not dup.
- L2 normalisation, including the zero and NaN cases.
- Every quality gate, using synthetic crops.
- Decode rejections.
- Real-model tests (skip if the models are absent): blank and noise images → `NO_FACE`.
- Optional test with env `HUMANPASS_SAMPLE_FACE`: produces a unit embedding, and the mirrored image still matches.
- API tests with fakes:
  - token required
  - response contains no float arrays, and no "embedding" or "similarity"
  - 422 mapping, 413, 400 for an invalid subjectId
  - security events never receive `subject_id`
  - a quality rejection never calls the store's enroll/lock path
- Integration tests with `TEST_DATABASE_URL` as the **runtime role**. Use a unique model_name per test and clean up afterwards.
  - identical → dup
  - different embedding with similarity 0.55 → dup
  - similarity 0.20 → not dup
  - another model_version → excluded
  - revoked → excluded
  - a duplicate is not persisted
  - the same subject twice → conflict
  - the security_event is committed atomically with the decision
  - **10 concurrent enrollments of the same vector → exactly 1 NOT_DUPLICATE, 9 DUPLICATE, 1 row.** Confirm this test FAILS when the advisory lock is removed, and report that you did.
  - the runtime role cannot read `identity.*`

API (jest + supertest):
- Status codes: 201, 201 for PNG, 409, 422, 400, 415, 413, 503.
- The exact image bytes are forwarded to the engine.
- `subject_id` is persisted only for ENROLLED.
- No "embedding", "similarity" or "vector" appears in any response.
- The client's status mapping is tested against a local HTTP server.
- The ULID spec vector.

## Scripts and Makefile
- `make env | up | down | reset | logs | test | verify | enroll-demo SELFIE=.. GROUP=..`
- `scripts/enroll-demo.sh`: selfie → 201, same selfie → 409, group photo → 422 MULTIPLE_FACES; prints PASS/FAIL.
- `scripts/verify-isolation.sh`:
  - each role is denied on the other schema (via `docker compose exec postgres psql -U <role>`)
  - **invariant: no biometric image artifacts on any filesystem.** Scan every writable path in both service containers (at least `/tmp`) for files whose magic bytes are JPEG/PNG/WebP/BMP, or whose names end in image extensions. Fail if any is found. Other temp files are allowed.
- Don't commit face images. The demo uses the user's own photos.

## Docs
- `docs/specs/humanpass-v0.1.md`: this document, verbatim.
- `docs/adr/001-server-side-embeddings.md`: the browser is untrusted, and a client could forge vectors.
- `docs/adr/002-no-raw-image-storage.md`: enforcement table. Be honest that Python/Node cannot guarantee RAM zeroisation. Consequence: changing the model requires re-enrollment.
- `docs/adr/003-biometric-deduplication.md`:
  - safeguards: policy separation, duplicates not persisted, no oracle, race-safe, exact search
  - production requirements: legal basis, necessity/proportionality, DPIA, non-biometric alternative, appeal path
  - the risk that false duplicates grow with N: false duplicate rate → legitimate users blocked → appeals/abandonment → lost conversion
  - the Open architectural decision note, verbatim:
    > v0.1 does not define whether biometric deduplication operates within a customer/tenant boundary or across the HumanPass network. The initial commercial model is expected to favor tenant-scoped deduplication. Cross-tenant verification, if introduced, should prefer portable credentials over cross-customer biometric searches and requires separate privacy, legal and threat-model analysis.
- `docs/adr/004-biometric-template-storage.md`: no column encryption, because pgvector must search the vectors. Include the compensating-controls table. Rejected: column encryption, cancelable biometrics (for now), homomorphic encryption.
- `docs/adr/005-exact-search.md`: HNSW can miss the true nearest neighbour (a false negative, i.e. a missed duplicate), and filtered search makes it worse. Benchmark recall before adopting it.
- `docs/threat-model.md`: STRIDE table.
  - Explicitly ❌ in v0.1: printed photo, screen, replay, deepfake/virtual camera.
  - ✅: forged embedding, Sybil, race condition (T-07), identification oracle, template exfiltration, image artifacts on disk.
  - ⚠️: rate limiting, mTLS, decompression bomb, false duplicates, and **T-10 distributed inconsistency**, which must include:
    - the concrete state: engine committed the biometric profile, the network or identity-api failed, the identity enrollment is missing, and the profile is orphaned (it still blocks that face as a duplicate);
    - future strategies: idempotency key on the engine call, reconciliation job, transactional outbox, saga/workflow, collapsing persistence ownership into one service.
- `docs/privacy-model.md`: data inventory, the identifier semantics table above, lifecycle, revocation/erasure, DPIA considerations (reference the AEPD's March 2025 decision on LaLiga stadium biometrics).
- `docs/model-card.md`: models, licenses, hashes, pipeline, a provisional thresholds table with env vars, limitations (accuracy, not calibrated, bias not measured, no liveness, scale).
- `README.md`: positioning, architecture diagram, quick start, API table, layout, docs index, definition-of-done checklist, roadmap.

## Milestones
1. `chore: monorepo skeleton, docker-compose, env example, spec` (includes `docs/specs/humanpass-v0.1.md`)
2. `feat(db): roles, schemas, pgvector, migrations`
3. `feat(engine): decode + YuNet detection + quality checks` (+ tests)
4. `feat(engine): SFace embedding + L2 normalization` (+ tests)
5. `feat(engine): exact pgvector search + DecisionPolicy + advisory lock` (+ integration and concurrency tests)
6. `feat(api): POST /api/v1/enroll + biometric client` (+ tests)
7. `docs: ADR-001..005, threat model, privacy model, model card`
8. `chore: Makefile, enroll-demo, verify-isolation, README`

## Definition of done
- [ ] `docker compose up` works from scratch with no manual steps
- [ ] same selfie → 201 then 409
- [ ] each role is denied on the other schema
- [ ] no biometric image artifact on any filesystem
- [ ] no response or log contains the embedding
- [ ] 10 concurrent enrollments of the same embedding → exactly 1 persisted (and the test fails without the lock)
- [ ] both test suites green
- [ ] all docs written

## IMPLEMENTATION MODE

Do NOT attempt to implement all eight milestones in one pass.

Start with Milestone 1 only.

For each milestone:
1. Re-read `docs/specs/humanpass-v0.1.md` and inspect the existing repository.
2. State the files you intend to create or modify.
3. Implement the milestone.
4. Run its relevant tests/checks.
5. Fix failures.
6. Summarize:
   - files changed
   - commands executed
   - tests passed/failed
   - known limitations
7. STOP.

Wait for explicit approval before proceeding to the next milestone.

Never modify files belonging to a future milestone unless required to make the current milestone executable.

Don't add anything beyond this scope. If a decision in this spec seems wrong, stop and explain why instead of changing it silently.
