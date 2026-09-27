#!/usr/bin/env bash
# Creates .env from .env.example, filling every empty KEY= with a random 32-byte hex value.
# Hex only: the values are embedded in postgres connection URLs.
# Refuses to overwrite an existing .env unless --force is given.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
example="$root/.env.example"
target="$root/.env"

if [[ -e "$target" && "${1:-}" != "--force" ]]; then
  echo "init-env: $target already exists (use --force to regenerate)" >&2
  exit 0
fi

rand_hex() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    od -An -N32 -tx1 /dev/urandom | tr -d ' \n'
  fi
}

tmp="$(mktemp "$root/.env.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
chmod 600 "$tmp"

while IFS= read -r line || [[ -n "$line" ]]; do
  if [[ "$line" =~ ^([A-Z0-9_]+)=$ ]]; then
    printf '%s=%s\n' "${BASH_REMATCH[1]}" "$(rand_hex)" >>"$tmp"
  else
    printf '%s\n' "$line" >>"$tmp"
  fi
done <"$example"

mv "$tmp" "$target"
trap - EXIT
echo "init-env: wrote $target"
