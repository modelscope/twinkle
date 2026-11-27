# Make sure to modify __release_datetime__ to release time when making official release.
# The version itself is not written here: every distribution in this repository reads the single
# VERSION file at the repository root (see pyproject.toml and twinkle_cs/README.md).
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("twinkle-kit")
except PackageNotFoundError:
    # Source checkout without an installed distribution (e.g. tests running from `src/`).
    __version__ = (Path(__file__).resolve().parents[2] / "VERSION").read_text().strip()

# default release datetime for branches under active development is set
# to be a time far-far-away-into-the-future
__release_datetime__ = '2099-10-13 08:56:12'
