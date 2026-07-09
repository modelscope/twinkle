# Notifier

The Notifier component provides a pluggable notification system for sending alerts during training. When exceptions occur or training events need attention, notifiers deliver messages to external channels (e.g., DingTalk webhooks).

## Base Interface

```python
from twinkle.notifier import Notifier

class Notifier:
    def __call__(self, message: str):
        """Send a notification message."""
        ...

    def to_dict(self) -> dict:
        """Serialize for checkpoint/restore."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> Notifier:
        """Restore from serialized form."""
        ...
```

## DingNotifier

Sends notifications to DingTalk (钉钉) custom robot webhooks.

```python
from twinkle.notifier import DingNotifier

notifier = DingNotifier(
    ding_url='https://oapi.dingtalk.com/robot/send?access_token=xxx',
    secret='SECxxxxxxx',  # Optional: for signed robots
    timeout=5.0,
)

# Send a message
notifier("### Training Complete\n\n- Steps: 1000\n- Loss: 0.25")
```

**Parameters:**
- `ding_url`: Full DingTalk webhook URL with access token
- `secret`: Optional signing secret for signed-robot mode
- `timeout`: HTTP request timeout in seconds (default: 5.0)

Messages are sent as DingTalk **Markdown** format. The first heading line is extracted as the chat preview title.

## Exception Notifications

Twinkle provides automatic exception notification with deduplication:

```python
from twinkle.notifier.base import notify_exception

# Automatically sends formatted exception info
# Only one rank sends per unique exception (prevents flooding)
try:
    model.forward_backward(batch)
except Exception as e:
    notify_exception(notifier, context='forward_backward', exc=e, name='sft_train')
```

The notification includes:
- Exception type and message
- Full traceback
- Runtime metadata (rank, PID, hostname)
- Deduplication: only one notification per unique exception across all ranks

## Custom Notifier

Create custom notifiers by subclassing `Notifier`:

```python
from twinkle.notifier import Notifier

class SlackNotifier(Notifier):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def __call__(self, message: str):
        import requests
        requests.post(self.webhook_url, json={'text': message})

    def to_dict(self):
        return {'class': 'SlackNotifier', 'webhook_url': self.webhook_url}

    @classmethod
    def _from_dict_impl(cls, data):
        return cls(webhook_url=data['webhook_url'])
```

> Notifiers are registered automatically via `__init_subclass__`, so `Notifier.from_dict()` can restore any subclass by name.
