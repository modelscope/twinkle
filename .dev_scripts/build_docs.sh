#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
make -C "$REPO_DIR/docs" html SOURCEDIR=source_en BUILDDIR=build/en
make -C "$REPO_DIR/docs" html SOURCEDIR=source_zh BUILDDIR=build/zh
