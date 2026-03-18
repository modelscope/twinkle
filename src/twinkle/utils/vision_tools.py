# Copyright (c) ModelScope Contributors. All rights reserved.
import base64
import os
import re
import requests
from io import BytesIO
from requests.adapters import HTTPAdapter
from typing import TYPE_CHECKING, TypeVar, Union
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    from PIL import Image

_T = TypeVar('_T')


def load_mm_file(path: Union[str, bytes, _T]) -> Union[BytesIO, _T]:
    res = path
    if isinstance(path, str):
        path = path.strip()
        if path.startswith('http'):
            retries = Retry(total=3, backoff_factor=1, allowed_methods=['GET'])
            with requests.Session() as session:
                session.mount('http://', HTTPAdapter(max_retries=retries))
                session.mount('https://', HTTPAdapter(max_retries=retries))
                response = session.get(path, timeout=10)
                response.raise_for_status()
                content = response.content
                res = BytesIO(content)
        else:
            data = path
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    res = BytesIO(f.read())
            else:
                if data.startswith('data:'):
                    match_ = re.match(r'data:(.+?);base64,(.+)', data)
                    assert match_ is not None
                    data = match_.group(2)
                data = base64.b64decode(data)
                res = BytesIO(data)
    elif isinstance(path, bytes):
        res = BytesIO(path)
    return res


def load_image(image: Union[str, bytes, 'Image.Image']) -> 'Image.Image':
    image = load_mm_file(image)
    if isinstance(image, BytesIO):
        from PIL import Image
        image = Image.open(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image
