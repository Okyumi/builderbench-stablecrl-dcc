# MJX Warp / warp-lang version pin

Date: 2026-08-22

## Root cause

The scratch environment was created with `pip install -e '.[all]'`. That
installs `mujoco==3.7.0` and `mujoco-mjx==3.7.0` as pinned, but originally
did **not** pin `warp-lang`. Pip resolved `warp-lang==1.16.0`.

`mujoco-mjx==3.7.0` declares `warp-lang==1.12.0` on its `warp` extra. The
1.16 interpreter rejects 3.7 Warp kernels (`WarpCodegenKeyError` in
`_sensor_pos` / `_frame_axis`) after earlier public-API breaks
(`GraphMode`, `warp_type_to_np_dtype`). Compatibility shims cannot fix
kernel codegen.

## Fix

`pyproject.toml` now pins `warp-lang == 1.12.0`. Reinstall that exact
wheel in `$SCRATCH/.venvs/builderbench-stablecrl-dcc`. The GraphMode /
dtype shims remain as no-ops when 1.12 already exposes those symbols.

## Validation

`tests/test_mjx_warp_compat.py` checks versions and the 3.7 Warp symbols.
Full `put_model(impl='warp')` still needs a CUDA device.

## Limitations

Do not upgrade `warp-lang` independently of `mujoco` / `mujoco-mjx`.
`mujoco-mjx==3.12` is the first release that pairs with `warp-lang==1.16`.
