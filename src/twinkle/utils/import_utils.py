# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.metadata
from functools import lru_cache
from packaging.requirements import Requirement


@lru_cache
def requires(package: str):
    req = Requirement(package)
    pkg_name = req.name
    try:
        installed_version = importlib.metadata.version(pkg_name)
        if req.specifier:
            if not req.specifier.contains(installed_version):
                raise ImportError(f"Package '{pkg_name}' version {installed_version} "
                                  f'does not satisfy {req.specifier}')
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(f"Required package '{pkg_name}' is not installed")


@lru_cache
def exists(package: str):
    try:
        requires(package)
        return True
    except ImportError:
        return False
