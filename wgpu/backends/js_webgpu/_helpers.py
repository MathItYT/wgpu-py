from __future__ import annotations

import asyncio
import dataclasses
import re
from collections.abc import Mapping
from typing import Any, Callable

from ..._async import GPUPromise


def snake_to_camel(name: str) -> str:
    return re.sub(r"_([a-zA-Z0-9])", lambda m: m.group(1).upper(), name)


def _unwrap(value: Any) -> Any:
    internal = getattr(value, "_internal", None)
    return internal if internal is not None else value


def to_js_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    value = _unwrap(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return value
    if dataclasses.is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        # Pyodide 0.29+ converts Python dictionaries to JavaScript Maps by
        # default. WebGPU descriptors are WebIDL dictionaries and therefore
        # require ordinary JavaScript Objects (not Maps).
        from js import Object
        from pyodide.ffi import to_js

        normalized = {
            snake_to_camel(str(k)): to_js_value(v)
            for k, v in value.items()
            if v is not None
        }
        return to_js(normalized, dict_converter=Object.fromEntries)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_js_value(v) for v in value]
    return value


def descriptor(**kwargs: Any) -> dict[str, Any]:
    return {snake_to_camel(k): to_js_value(v) for k, v in kwargs.items() if v is not None}


def js_get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name)
    except (AttributeError, TypeError):
        return default
    return default if value is None else value


def js_to_python(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.to_py()
    except (AttributeError, TypeError):
        return value


def js_record_to_dict(value: Any) -> dict[str, Any]:
    value = js_to_python(value)
    if isinstance(value, dict):
        return {str(k): js_to_python(v) for k, v in value.items()}
    return {}


def make_promise(title: str, awaitable: Any, *, handler: Callable[[Any], Any] | None = None) -> GPUPromise:
    promise = GPUPromise(title, handler)

    async def resolve() -> None:
        try:
            result = await awaitable
        except Exception as exc:
            promise._wgpu_set_error(exc)
        else:
            promise._wgpu_set_input(result)

    asyncio.ensure_future(resolve())
    return promise


def require_browser_webgpu() -> Any:
    try:
        from js import navigator
    except ImportError as exc:
        raise RuntimeError("The js_webgpu backend requires Pyodide.") from exc
    gpu = getattr(navigator, "gpu", None)
    if gpu is None:
        raise RuntimeError("WebGPU is not available in this browser. Use a browser with WebGPU enabled.")
    return gpu


def camel_method(name: str) -> str:
    return snake_to_camel(name)
