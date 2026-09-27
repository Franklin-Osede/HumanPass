-- Biometric templates. The embedding never leaves this schema / the biometric-engine.
CREATE TABLE biometric.biometric_profiles (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id          text NOT NULL UNIQUE,
  embedding           biometric.vector(128) NOT NULL,
  model_name          text NOT NULL,
  model_version       text NOT NULL,
  embedding_dimension integer NOT NULL CHECK (embedding_dimension = 128),
  enrolled_at         timestamptz NOT NULL DEFAULT now(),
  revoked_at          timestamptz NULL
);

-- Deliberately NO HNSW/IVFFlat index: approximate search can miss the true nearest
-- neighbour, i.e. a missed duplicate (ADR-005). Exact scan over the active rows of one model.
CREATE INDEX biometric_profiles_active_model_idx
  ON biometric.biometric_profiles (model_name, model_version)
  WHERE revoked_at IS NULL;

-- One row per biometric execution. Must never hold a subject_id or an embedding.
CREATE TABLE biometric.security_events (
  processing_id uuid PRIMARY KEY,
  decision      text NOT NULL CHECK (decision IN ('ENROLLED', 'DUPLICATE', 'REJECTED')),
  reason_code   text NULL,
  similarity    real NULL,
  model_name    text NOT NULL,
  model_version text NOT NULL,
  latency_ms    integer NOT NULL CHECK (latency_ms >= 0),
  created_at    timestamptz NOT NULL DEFAULT now()
);
