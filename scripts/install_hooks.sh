#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
GIT_DIR="$(git rev-parse --git-dir)"
HOOKS_DIR="$GIT_DIR/hooks"
TARGET="$HOOKS_DIR/pre-commit"
SOURCE="$REPO_ROOT/.githooks/pre-commit"

mkdir -p "$HOOKS_DIR"
ln -sf "$SOURCE" "$TARGET"
chmod +x "$SOURCE"

printf 'Installed Git pre-commit hook in %s.\n' "$HOOKS_DIR"
