# End-of-run matplotlib is optional for job success

Date: 2026-08-22

## Failure

Slurm array `16192450` is the pre-pin Warp 1.16 batch; every cell died in
about a minute on `warp.types.warp_type_to_np_dtype`. Ignore that summary
mail.

The replacement array `16196368` trained to the 200M-step boundary, then
exited 1 because `stable_crl.py` imported matplotlib only to write
`metrics.png`.

## Fix

Install `matplotlib==3.10.8` into the scratch env (running tasks import it
at process end, so this can still save in-flight jobs). Guard the plot in
`stable_crl.py` and `stable_crl_dcc.py` so a missing plotting stack cannot
fail an otherwise finished run. W&B logs and `eval_log.jsonl` are already
written before that import.

## Validation

`16196368` eval logs reached `env_steps ~= 2.01e8` with finite losses before
the matplotlib crash. The Warp 1.12 pin is not implicated in that failure.
