# Copyright (c) ModelScope Contributors. All rights reserved.
import base64
import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Dict, Optional

from .base import Notifier


class DingNotifier(Notifier):
    """Send notifications to a DingTalk custom robot webhook.

    Args:
        ding_url: The full webhook URL, e.g.
            ``https://oapi.dingtalk.com/robot/send?access_token=xxx``.
        secret: Optional signing secret. If provided, ``timestamp``/``sign``
            query parameters are appended to each request as required by
            DingTalk's signed-robot mode.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        ding_url: str,
        secret: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        super().__init__()
        if not ding_url:
            raise ValueError('ding_url must be a non-empty DingTalk webhook URL')
        self.ding_url = ding_url
        self.secret = secret
        self.timeout = timeout

    def to_dict(self) -> Dict[str, str]:
        d = {
            'class': 'DingNotifier',
            'ding_url': self.ding_url,
            'timeout': str(self.timeout),
        }
        if self.secret:
            d['secret'] = self.secret
        return d

    @classmethod
    def _from_dict_impl(cls, data: dict):
        url = data.get('ding_url', '')
        if not url:
            return None
        return cls(
            ding_url=url,
            secret=data.get('secret') or None,
            timeout=float(data.get('timeout', '5.0')),
        )

    def _sign(self) -> dict:
        """Build ``timestamp``/``sign`` query params for signed webhooks."""
        if not self.secret:
            return {}
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f'{timestamp}\n{self.secret}'
        hmac_code = hmac.new(
            self.secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return {'timestamp': timestamp, 'sign': sign}

    def _build_url(self) -> str:
        extra = self._sign()
        if not extra:
            return self.ding_url
        sep = '&' if '?' in self.ding_url else '?'
        query = '&'.join(f'{k}={v}' for k, v in extra.items())
        return f'{self.ding_url}{sep}{query}'

    def __call__(self, message: str) -> dict:
        """Send ``message`` as a DingTalk markdown notification.

        The first ``#``/``##``/``###`` heading line is used as ``title``
        (which surfaces in the chat preview); falls back to the first
        non-empty line, truncated to 64 chars.

        Returns the parsed JSON response from DingTalk. Raises on HTTP
        failure or on a non-zero ``errcode`` in the response body.
        """
        import requests

        text = str(message)
        title = _extract_title(text)
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': title,
                'text': text
            },
        }
        resp = requests.post(
            self._build_url(),
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get('errcode', 0) != 0:
            raise RuntimeError(f'DingTalk notify failed: errcode={result.get("errcode")}, '
                               f'errmsg={result.get("errmsg")}')
        return result


def _extract_title(text: str, max_len: int = 64) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Strip leading markdown heading markers if any.
        cleaned = stripped.lstrip('#').strip() or stripped
        return cleaned[:max_len]
    return 'Twinkle'
