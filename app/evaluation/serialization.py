"""Generic dataclass <-> JSON-safe conversion.

Extracted from three byte-identical copies that previously lived in
``app.evaluation.benchmark`` (Phase 16F), ``app.evaluation.reasoning_benchmark``
(Phase 20A), and ``app.evaluation.judge_benchmark`` (Phase 20B) — each phase's
own docstring explains *why* the copy existed ("no shared base class exists to
extend without modifying an earlier phase"); this module is that shared base,
added without modifying any of the three call sites' public behavior.

Note: ``app.evaluation.experiment_tracking`` (Phase 21F) has its own, slightly
different ``_to_jsonable`` (it accepts any ``collections.abc.Mapping`` and
stringifies keys, not just plain ``dict``) and does not use this module — the
two are not behaviorally identical, so unifying them would be a behavior
change, not a pure simplification.
"""

from __future__ import annotations

import types
import typing
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into something ``json.dumps`` accepts:
    dataclasses -> dicts of their fields, enums -> their ``.value``,
    tuples -> lists, dicts -> dicts (values converted), everything else
    passed through unchanged (str/int/float/bool/None).
    """
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def from_jsonable(data: Any, target_type: Any) -> Any:
    """The inverse of ``to_jsonable``, guided by ``target_type``'s resolved
    type hints. Generic across any dataclass tree because it walks
    ``dataclasses.fields``/``typing.get_type_hints`` rather than hand-coding
    each type — adding a field to any report dataclass needs no change here.
    """
    if data is None:
        return None

    origin = typing.get_origin(target_type)

    if origin in (typing.Union, types.UnionType):
        non_none_args = [arg for arg in typing.get_args(target_type) if arg is not type(None)]
        return from_jsonable(data, non_none_args[0])

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(data)

    if is_dataclass(target_type):
        hints = typing.get_type_hints(target_type)
        kwargs = {
            f.name: from_jsonable(data[f.name], hints[f.name]) for f in fields(target_type)
        }
        return target_type(**kwargs)

    if origin is tuple:
        (item_type, *_rest) = typing.get_args(target_type)
        return tuple(from_jsonable(item, item_type) for item in data)

    if origin is dict:
        _key_type, value_type = typing.get_args(target_type)
        return {key: from_jsonable(item, value_type) for key, item in data.items()}

    return data
