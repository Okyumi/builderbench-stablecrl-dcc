"""Compatibility shims for MuJoCo 3.7.0 against newer warp-lang.

MuJoCo 3.7.0 still uses Warp APIs that NVIDIA moved or deleted in 1.16:

- ``GraphMode`` lived at ``warp._src.jax_experimental.ffi``. The failed import
  made ``mjxw.types.GraphMode`` an alias of ``int``, so ``put_model`` raised
  ``AttributeError: type object 'int' has no attribute 'WARP'``.
- ``warp.types.warp_type_to_np_dtype`` moved to ``warp._src.types``. MJX
  still reads the public module and then raises
  ``AttributeError: module 'warp.types' has no attribute 'warp_type_to_np_dtype'``.
"""
from __future__ import annotations


def patch_mjx_warp_graph_mode() -> None:
    import mujoco.mjx.warp as mjxw

    graph_mode = getattr(mjxw.types, "GraphMode", None)
    if graph_mode is not None and hasattr(graph_mode, "WARP"):
        return
    try:
        from warp import JaxCallableGraphMode
    except ImportError as exc:
        raise ImportError(
            "warp-lang is required for the MJX warp backend"
        ) from exc
    mjxw.types.GraphMode = JaxCallableGraphMode


def patch_warp_type_maps() -> None:
    import warp.types as warp_types

    if hasattr(warp_types, "warp_type_to_np_dtype"):
        return
    try:
        import warp._src.types as warp_src_types
    except ImportError as exc:
        raise ImportError(
            "warp-lang is required for the MJX warp backend"
        ) from exc
    for name in ("warp_type_to_np_dtype", "np_dtype_to_warp_type"):
        if hasattr(warp_src_types, name) and not hasattr(warp_types, name):
            setattr(warp_types, name, getattr(warp_src_types, name))


def apply_mjx_warp_compat() -> None:
    patch_mjx_warp_graph_mode()
    patch_warp_type_maps()


apply_mjx_warp_compat()
