#!/usr/bin/env bash
# Forward-only SQL migrator.
#   usage: migrate.sh <schema> <migrations-dir>
# Applies <dir>/*.sql in lexical order, each in its own transaction together with its
# <schema>.schema_migrations row. Already-applied files are skipped; an applied file whose
# content changed is a hard error (migrations are immutable, never edited or rolled back).
# Runs as the superuser (PG* env vars); schema_migrations is revoked from the runtime roles.
set -euo pipefail

schema="${1:?usage: migrate.sh <schema> <migrations-dir>}"
dir="${2:?usage: migrate.sh <schema> <migrations-dir>}"

if [[ ! "$schema" =~ ^[a-z_]+$ ]]; then
  echo "migrate: invalid schema name '$schema'" >&2
  exit 2
fi

sql() { psql -X -q -v ON_ERROR_STOP=1 "$@"; }

sql <<SQL
CREATE TABLE IF NOT EXISTS ${schema}.schema_migrations (
  version    text PRIMARY KEY,
  checksum   text NOT NULL,
  applied_at timestamptz NOT NULL DEFAULT now()
);
REVOKE ALL ON ${schema}.schema_migrations FROM PUBLIC, identity_api, biometric_engine;
SQL

shopt -s nullglob
files=("$dir"/*.sql)
if ((${#files[@]} == 0)); then
  echo "migrate[$schema]: no migrations in $dir" >&2
  exit 1
fi

for file in "${files[@]}"; do
  version="$(basename "$file" .sql)"
  checksum="$(sha256sum "$file" | cut -d' ' -f1)"

  applied="$(sql -tA -v version="$version" <<SQL
SELECT checksum FROM ${schema}.schema_migrations WHERE version = :'version';
SQL
)"

  if [[ -n "$applied" ]]; then
    if [[ "$applied" != "$checksum" ]]; then
      echo "migrate[$schema]: $version was modified after being applied (checksum mismatch)" >&2
      exit 1
    fi
    echo "migrate[$schema]: $version already applied"
    continue
  fi

  sql --single-transaction -v version="$version" -v checksum="$checksum" <<SQL
\i ${file}
INSERT INTO ${schema}.schema_migrations (version, checksum) VALUES (:'version', :'checksum');
SQL
  echo "migrate[$schema]: applied $version"
done
