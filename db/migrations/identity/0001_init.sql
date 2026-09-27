-- One row per enrollment attempt. subject_id is set only when a new subject was ENROLLED;
-- DUPLICATE and REJECTED attempts create no subject.
-- model_name/model_version are nullable: a rejection response from the engine carries no model.
CREATE TABLE identity.enrollments (
  id            text PRIMARY KEY,  -- ULID
  subject_id    text NULL,
  status        text NOT NULL CHECK (status IN ('ENROLLED', 'DUPLICATE', 'REJECTED')),
  reason_code   text NULL,
  model_name    text NULL,
  model_version text NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT enrollments_subject_iff_enrolled CHECK ((status = 'ENROLLED') = (subject_id IS NOT NULL))
);

CREATE UNIQUE INDEX enrollments_subject_id_uq
  ON identity.enrollments (subject_id)
  WHERE subject_id IS NOT NULL;
