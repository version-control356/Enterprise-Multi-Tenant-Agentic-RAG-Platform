#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 backup <directory> | restore <dump-file>" >&2
    exit 2
}

compose=(docker compose)

if [[ $# -lt 2 ]]; then
    usage
fi

case "$1" in
    backup)
        destination=$2
        mkdir -p "$destination"
        "${compose[@]}" exec -T postgres pg_dump -Fc \
            -U "${POSTGRES_USER:?POSTGRES_USER must be set}" \
            -d "${POSTGRES_DB:?POSTGRES_DB must be set}" \
            > "$destination/postgres.dump"
        echo "PostgreSQL backup written to $destination/postgres.dump"
        ;;
    restore)
        dump_file=$2
        [[ -f "$dump_file" ]] || { echo "Backup not found: $dump_file" >&2; exit 1; }
        "${compose[@]}" exec -T postgres pg_restore \
            --clean --if-exists --no-owner \
            -U "${POSTGRES_USER:?POSTGRES_USER must be set}" \
            -d "${POSTGRES_DB:?POSTGRES_DB must be set}" \
            < "$dump_file"
        echo "PostgreSQL backup restored from $dump_file"
        ;;
    *)
        usage
        ;;
esac
