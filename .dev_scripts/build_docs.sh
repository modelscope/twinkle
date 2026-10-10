#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Sphinx leaves HTML for removed sources behind; clean both language outputs.
rm -rf "$REPO_DIR/docs/build/en" "$REPO_DIR/docs/build/zh"
make -C "$REPO_DIR/docs" html SOURCEDIR=source_en BUILDDIR=build/en
make -C "$REPO_DIR/docs" html SOURCEDIR=source_zh BUILDDIR=build/zh
