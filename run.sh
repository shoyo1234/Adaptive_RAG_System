#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s\n\nRuns the dataset checks, index build, tests, evaluations, timing run, and exports requirements.txt.\n' "${BASH_SOURCE[0]}"
    exit 0
fi

if [[ $# -gt 0 ]]; then
    printf 'Unexpected argument: %s\nUsage: %s\n' "$1" "${BASH_SOURCE[0]}" >&2
    exit 2
fi

if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    printf 'Python was not found. Install Python 3 and the project dependencies first.\n' >&2
    exit 1
fi

if ! "$PYTHON" -c 'import lzma' >/dev/null 2>&1; then
    printf 'The selected Python interpreter (%s) is missing its standard-library lzma module.\n' "$PYTHON" >&2
    printf 'Recreate .venv using a Python installation with lzma support, then reinstall dependencies:\n' >&2
    printf '  mv .venv .venv-broken\n  /usr/bin/python3 -m venv .venv\n' >&2
    printf '  .venv/bin/python -m pip install -r requirements.txt\n' >&2
    exit 1
fi

cd "$PROJECT_DIR"

run_step() {
    printf '\n>>>'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

run_step "$PYTHON" -m adaptive_rag.data
run_step "$PYTHON" -m adaptive_rag.indexes
run_step "$PYTHON" -m pytest tests -v
run_step "$PYTHON" -m evaluation.run_eval --split dev --mode all --tag three_way
run_step "$PYTHON" -m evaluation.precompute_kcurve --split dev
run_step "$PYTHON" -m evaluation.precompute_kcurve --split test
run_step "$PYTHON" -m evaluation.timing --split test --repeats 3
printf '\n>>> %q -m pip freeze > requirements.txt\n' "$PYTHON"
"$PYTHON" -m pip freeze > "$PROJECT_DIR/requirements.txt"

printf '\nAll pipeline commands completed successfully.\n'
