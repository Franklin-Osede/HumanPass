#!/usr/bin/env bash
# Runs once, as the superuser, on an empty data volume (docker-entrypoint-initdb.d).
# Creates the two runtime LOGIN roles, their schemas and the privilege boundary between them.
# Tables are created later by the migrate services (db/migrate.sh), also as the superuser;
# the ALTER DEFAULT PRIVILEGES below grant the runtime roles access to those tables.
set -euo pipefail

: "${IDENTITY_API_DB_PASSWORD:?IDENTITY_API_DB_PASSWORD is required}"
: "${BIOMETRIC_ENGINE_DB_PASSWORD:?BIOMETRIC_ENGINE_DB_PASSWORD is required}"

psql -X -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v identity_pw="$IDENTITY_API_DB_PASSWORD" \
  -v biometric_pw="$BIOMETRIC_ENGINE_DB_PASSWORD" <<'SQL'
BEGIN;

REVOKE ALL ON SCHEMA public FROM PUBLIC;

CREATE ROLE identity_api LOGIN PASSWORD :'identity_pw';
CREATE ROLE biometric_engine LOGIN PASSWORD :'biometric_pw';

-- Schemas are owned by the superuser, so runtime roles cannot ALTER/DROP objects or CREATE new ones.
CREATE SCHEMA identity;
CREATE SCHEMA biometric;
REVOKE ALL ON SCHEMA identity FROM PUBLIC;
REVOKE ALL ON SCHEMA biometric FROM PUBLIC;

GRANT USAGE ON SCHEMA identity TO identity_api;
GRANT USAGE ON SCHEMA biometric TO biometric_engine;

-- Cross-schema access, revoked explicitly in both directions.
REVOKE ALL ON SCHEMA biometric FROM identity_api;
REVOKE ALL ON ALL TABLES IN SCHEMA biometric FROM identity_api;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA biometric FROM identity_api;
REVOKE ALL ON SCHEMA identity FROM biometric_engine;
REVOKE ALL ON ALL TABLES IN SCHEMA identity FROM biometric_engine;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA identity FROM biometric_engine;

-- The vector type lives only in the biometric schema: identity_api cannot even reference it.
CREATE EXTENSION vector SCHEMA biometric;

-- Privileges on tables/sequences the superuser creates later (i.e. by migrations).
ALTER DEFAULT PRIVILEGES FOR ROLE CURRENT_USER IN SCHEMA identity
  GRANT SELECT, INSERT, UPDATE ON TABLES TO identity_api;
ALTER DEFAULT PRIVILEGES FOR ROLE CURRENT_USER IN SCHEMA biometric
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO biometric_engine;  -- DELETE is for erasure
ALTER DEFAULT PRIVILEGES FOR ROLE CURRENT_USER IN SCHEMA biometric
  GRANT USAGE ON SEQUENCES TO biometric_engine;

ALTER ROLE identity_api SET search_path = identity;
ALTER ROLE biometric_engine SET search_path = biometric;

COMMIT;
SQL
