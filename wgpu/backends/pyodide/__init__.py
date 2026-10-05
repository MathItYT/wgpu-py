"""Pyodide WebGPU backend compatibility module.

The implementation lives in :mod:`wgpu.backends.js_webgpu`. This module
provides the backend name expected by rendercanvas' Pyodide integration.
"""

from ..js_webgpu import *  # noqa: F401,F403
