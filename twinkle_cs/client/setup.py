"""Version shim for a sub-distribution of the twinkle repository.

Every distribution takes its version from the repository-root ``VERSION`` file; none carries a
version of its own. setuptools refuses to read files outside the project directory from
``pyproject.toml``, so this shim copies ``VERSION`` next to the package before handing it to
``setup()``. ``MANIFEST.in`` ships that copy inside the sdist, which keeps the wheel rebuildable
from the sdist alone. The copy is git-ignored.
"""
from pathlib import Path
from setuptools import setup
from shutil import copyfile

_here = Path(__file__).resolve().parent
_version_file = _here / "VERSION"
if not _version_file.exists():
    copyfile(_here.parents[1] / "VERSION", _version_file)
setup(version=_version_file.read_text().strip())
