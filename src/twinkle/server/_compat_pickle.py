"""Compatibility patches for running the twinkle server on this stack
(pydantic 2.11/2.12 + ray serve + starlette + tinker SDK).

Three pickle/validation-time hazards are defused:

1. starlette State.__getattr__: during cloudpickle deserialization an attribute
   lookup on a State whose ``_state`` slot is not yet filled re-enters
   ``__getattr__`` forever. Read the slot through ``__dict__`` and raise
   AttributeError instead.
2. pydantic MockValSer.__getattr__: same pattern -- ``_attempt_rebuild`` /
   ``_val_or_ser`` slots may be unfilled mid-unpickle; guard every access with
   ``object.__getattribute__`` and try the rebuild before failing.
3. tinker SDK request types are stdlib dataclasses whose ``__module__`` was
   rewritten to ``'tinker'``, so pydantic evaluates their string annotations
   against the ``tinker`` package namespace (missing typing names, numpy,
   LossFnInputs, ...). Inject the missing names and attach a pydantic core
   schema to ``TensorData`` (its ``np.ndarray`` field has no core schema).
"""
import sys
import typing


def _apply_tinker_pydantic_compat() -> None:
    """Make tinker SDK request types validatable by pydantic.

    Must run before the first FastAPI body validation (i.e. at import time).
    """
    try:
        import numpy as np
        import numpy.typing as npt
        import tinker.types.tensor_data  # noqa: F401  (eager import)
        import tinker.types as tt

        tinker_ns = sys.modules['tinker']
        for name in ('Optional', 'List', 'Dict', 'Union', 'Mapping', 'Any', 'Literal',
                     'Annotated', 'Sequence', 'Tuple', 'Callable'):
            if not hasattr(tinker_ns, name):
                setattr(tinker_ns, name, getattr(typing, name))
        if not hasattr(tinker_ns, 'np'):
            tinker_ns.np = np
        if not hasattr(tinker_ns, 'npt'):
            tinker_ns.npt = npt
        for name in dir(tt):
            if not hasattr(tinker_ns, name):
                setattr(tinker_ns, name, getattr(tt, name))

        def _td_validate(v, info=None):
            if isinstance(v, tt.TensorData):
                return v
            return tt.TensorData(
                data=v.get('data'), dtype=v['dtype'], shape=v.get('shape'),
                sparse_crow_indices=v.get('sparse_crow_indices'),
                sparse_col_indices=v.get('sparse_col_indices'))

        def _td_get_pydantic_core_schema(source_type, handler):
            from pydantic_core import core_schema
            return core_schema.with_info_plain_validator_function(_td_validate)

        if not hasattr(tt.TensorData, '_twinkle_pydantic_patched'):
            tt.TensorData.__get_pydantic_core_schema__ = _td_get_pydantic_core_schema
            tt.TensorData._twinkle_pydantic_patched = True
    except Exception:
        pass


def _apply_server_pickle_compat() -> None:
    # --- starlette State ---
    try:
        from starlette.datastructures import State

        if not getattr(State.__getattr__, '_twinkle_safe', False):
            def _safe_state_getattr(self, key):
                try:
                    return self.__dict__['_state'][key]
                except KeyError:
                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{key}'")

            _safe_state_getattr._twinkle_safe = True
            State.__getattr__ = _safe_state_getattr
    except Exception:
        pass

    # --- pydantic MockValSer ---
    try:
        import pydantic._internal._mock_val_ser as _mvs

        if not getattr(_mvs.MockValSer.__getattr__, '_twinkle_safe', False):
            def _safe_mock_getattr(self, item):
                try:
                    attempt = object.__getattribute__(self, '_attempt_rebuild')
                except AttributeError:
                    raise AttributeError(f'MockValSer not fully initialized (item={item!r})')
                if attempt:
                    try:
                        val_ser = attempt()
                    except RecursionError:
                        raise AttributeError(f'MockValSer rebuild loop (item={item!r})')
                    if val_ser is not None:
                        return getattr(val_ser, item)
                raise AttributeError(f'MockValSer unavailable (item={item!r})')

            _safe_mock_getattr._twinkle_safe = True
            _mvs.MockValSer.__getattr__ = _safe_mock_getattr
    except Exception:
        pass


_apply_tinker_pydantic_compat()
_apply_server_pickle_compat()
