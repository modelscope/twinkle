# Copyright (c) ModelScope Contributors. All rights reserved.
import fnmatch
import glob
import inspect
import os
import shutil
from functools import lru_cache


def deep_getattr(obj, attr: str, default=None):
    attrs = attr.split('.')
    for a in attrs:
        if obj is None:
            break
        if isinstance(obj, dict):
            obj = obj.get(a, default)
        else:
            obj = getattr(obj, a, default)
    return obj


@lru_cache(maxsize=None)
def signature_info(fn):
    signature = inspect.signature(fn)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    return accepts_kwargs, frozenset(signature.parameters)


def has_signature_parameter(fn, name: str) -> bool:
    return name in signature_info(fn)[1]


def call_with_supported_kwargs(fn, *args, **kwargs):
    accepts_kwargs, parameters = signature_info(fn)
    if not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return fn(*args, **kwargs)


def copy_files_by_pattern(source_dir, dest_dir, patterns, exclude_patterns=None):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if isinstance(patterns, str):
        patterns = [patterns]

    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = [exclude_patterns]

    def should_exclude_file(file_path, file_name):
        for exclude_pattern in exclude_patterns:
            if fnmatch.fnmatch(file_name, exclude_pattern):
                return True
            rel_file_path = os.path.relpath(file_path, source_dir)
            if fnmatch.fnmatch(rel_file_path, exclude_pattern):
                return True
        return False

    for pattern in patterns:
        pattern_parts = pattern.split(os.path.sep)
        if len(pattern_parts) > 1:
            subdir_pattern = os.path.sep.join(pattern_parts[:-1])
            file_pattern = pattern_parts[-1]

            for root, dirs, files in os.walk(source_dir):
                rel_path = os.path.relpath(root, source_dir)
                if rel_path == '.' or (rel_path != '.' and not fnmatch.fnmatch(rel_path, subdir_pattern)):
                    continue

                for file in files:
                    if fnmatch.fnmatch(file, file_pattern):
                        file_path = os.path.join(root, file)

                        if should_exclude_file(file_path, file):
                            continue

                        target_dir = os.path.join(dest_dir, rel_path)
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir)
                        dest_file = os.path.join(target_dir, file)

                        if not os.path.exists(dest_file):
                            shutil.copy2(file_path, dest_file)
        else:
            search_path = os.path.join(source_dir, pattern)
            matched_files = glob.glob(search_path)

            for file_path in matched_files:
                if os.path.isfile(file_path):
                    file_name = os.path.basename(file_path)

                    if should_exclude_file(file_path, file_name):
                        continue

                    destination = os.path.join(dest_dir, file_name)
                    if not os.path.exists(destination):
                        shutil.copy2(file_path, destination)


def get_runtime_meta() -> str:
    import platform
    import socket
    import sys
    hostname = 'unknown'
    ip = 'unknown'
    try:
        hostname = socket.gethostname()
    except Exception:  # noqa: BLE001
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # UDP connect does not actually send packets; resolves outbound iface IP.
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        try:
            ip = socket.gethostbyname(hostname)
        except Exception:  # noqa: BLE001
            pass
    rank = os.environ.get('RANK', '?')
    world_size = os.environ.get('WORLD_SIZE', '?')
    local_rank = os.environ.get('LOCAL_RANK', '?')
    lines = [
        f'- **Host**: `{hostname}` (`{ip}`)',
        f'- **Python**: `{platform.python_version()}` @ `{sys.executable}`',
        f'- **Rank**: `{rank}/{world_size}` (local_rank=`{local_rank}`)',
    ]
    return '\n'.join(lines)
