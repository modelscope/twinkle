import os
from typing import Dict, Optional

from twinkle.utils import get_logger, get_runtime_meta

logger = get_logger()


class Notifier:
    """Base class for notification sinks.

    Subclasses implement ``to_dict`` / ``_from_dict_impl`` for serialization.
    The registry auto-populates via ``__init_subclass__`` so ``from_dict``
    can dispatch by class name at restore time.
    """

    _registry: Dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Notifier._registry[cls.__name__] = cls

    def __call__(self, message: str):
        raise NotImplementedError

    def to_dict(self) -> Dict[str, str]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> Optional['Notifier']:
        class_name = data.get('class')
        if not class_name:
            return None
        target = cls._registry.get(class_name)
        if target is None:
            return None
        return target._from_dict_impl(data)

    @classmethod
    def _from_dict_impl(cls, data: dict) -> Optional['Notifier']:
        raise NotImplementedError


_runtime_meta_cache = None


def _try_claim_notify_slot(exc: BaseException, context: str, name: Optional[str] = None) -> bool:
    """Build an exception fingerprint and delegate to the generic single-winner claim."""
    from twinkle.utils.parallel import try_claim_once
    tb = exc.__traceback__
    last_frame = ''
    while tb is not None:
        last_frame = f'{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}'
        tb = tb.tb_next
    key = f'{name or "_"}|{type(exc).__name__}|{last_frame}|{context}'
    payload = (f'rank={os.environ.get("RANK", "?")} '
               f'pid={os.getpid()} ctx={context}\n')
    return try_claim_once(key, payload=payload, namespace='twinkle_notify')


def notify_exception(notifier: Notifier, context: str, exc: BaseException, name: Optional[str] = None) -> None:
    if notifier is None:
        return
    if getattr(exc, '_twinkle_notified', False):
        return
    if not _try_claim_notify_slot(exc, context, name):
        try:
            setattr(exc, '_twinkle_notified', True)
        except Exception:  # noqa: BLE001
            pass
        return

    global _runtime_meta_cache
    if _runtime_meta_cache is None:
        _runtime_meta_cache = get_runtime_meta()
    try:
        import traceback
        tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        title = f'[Twinkle] `{name or "unnamed"}` — Exception in `{context}`'
        msg = (f'### {title}\n\n'
               f'- **Type**: `{type(exc).__name__}`\n'
               f'- **Message**: {exc}\n'
               f'{_runtime_meta_cache}\n\n'
               f'```\n{tb_str}```\n')
        notifier(msg)
    except Exception as e:  # noqa
        logger.exception(f'Failed to send twinkle exception notification: {e}')
    finally:
        try:
            setattr(exc, '_twinkle_notified', True)
        except Exception:  # noqa
            pass
