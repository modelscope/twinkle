# notifier tests (plan T6-01..T6-08).
import base64
import hashlib
import hmac
import json
import urllib.parse

import pytest

from twinkle.notifier import DingNotifier
from twinkle.notifier.base import Notifier


class TestDingNotifierConstruction:
    """T6-01: the empty-URL guard."""

    def test_empty_url_rejected(self):
        with pytest.raises(ValueError, match='non-empty'):
            DingNotifier(ding_url='')

    def test_plain_url_accepted(self):
        n = DingNotifier(ding_url='https://oapi.dingtalk.com/robot/send?access_token=x')
        assert n.secret is None


class TestSerialization:
    """T6-02: to_dict/from_dict round-trip with and without a secret."""

    @pytest.mark.parametrize('secret', [None, 'SEC'])
    def test_round_trip(self, secret):
        n = DingNotifier(ding_url='https://x/y', secret=secret, timeout=7.5)
        data = n.to_dict()
        rebuilt = Notifier.from_dict(data)
        assert isinstance(rebuilt, DingNotifier)
        assert rebuilt.ding_url == 'https://x/y'
        assert rebuilt.timeout == 7.5
        assert rebuilt.secret == secret


class TestSigning:
    """T6-03/T6-04/T6-05: HMAC-SHA256 signature against an independent computation."""

    def test_no_secret_no_signature(self):
        assert DingNotifier(ding_url='https://x/y')._sign() == {}

    def test_sign_matches_independent_hmac(self, monkeypatch):
        timestamp = 1700000000000
        secret = 'mysecret'
        notifier = DingNotifier(ding_url='https://x/y', secret=secret)
        monkeypatch.setattr('twinkle.notifier.ding_notifier.time.time', lambda: timestamp / 1000)
        out = notifier._sign()
        # Independent oracle: the documented DingTalk recipe.
        string_to_sign = f'{timestamp}\n{secret}'
        digest = hmac.new(secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).digest()
        expected = urllib.parse.quote_plus(base64.b64encode(digest))
        assert out == {'timestamp': str(timestamp), 'sign': expected}
        # T6-05: the base64 signature is URL-escaped (no raw + / characters).
        assert '+' not in out['sign'] and '/' not in out['sign']

    def test_build_url_appends_signature(self, monkeypatch):
        monkeypatch.setattr('twinkle.notifier.ding_notifier.time.time', lambda: 1700000000.0)
        notifier = DingNotifier(ding_url='https://x/y', secret='s')
        url = notifier._build_url()
        assert 'timestamp=' in url and 'sign=' in url


class TestSend:
    """T6-06/T6-07: markdown payload and the failure path."""

    def _notifier(self):
        return DingNotifier(ding_url='https://x/y')

    def test_send_posts_markdown_payload(self, monkeypatch):
        posted = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {'errcode': 0}

        def fake_post(url, data=None, headers=None, timeout=None):
            posted.update(url=url, data=data, headers=headers, timeout=timeout)
            return _Resp()

        monkeypatch.setattr('requests.post', fake_post)
        result = self._notifier()('hello')
        payload = json.loads(posted['data'])
        assert payload['msgtype'] == 'markdown'
        assert payload['markdown']['text'] == 'hello'
        assert result == {'errcode': 0}

    def test_send_errcode_raises_runtime_error(self, monkeypatch):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {'errcode': 310000, 'errmsg': 'bad token'}

        monkeypatch.setattr('requests.post', lambda *a, **k: _Resp())
        with pytest.raises(RuntimeError, match='errcode=310000'):
            self._notifier()('hello')


class TestNotifierDispatch:
    """T6-08: the class dispatch on to_dict."""

    def test_unknown_class_returns_none(self):
        assert Notifier.from_dict({'class': 'NoSuchNotifier'}) is None
