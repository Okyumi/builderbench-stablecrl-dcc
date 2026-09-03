# W&B progress-video implementation

Sequence A now records training progress directly in Weights & Biases.

- Every task-level run logs its measured pre-training forward-transfer
  success at W&B step 0.
- Numerical evaluation metrics and media are committed together at the same
  W&B step, avoiding separate media rows with ambiguous step alignment.
- The production configuration requests 10 videos from 50 evaluation points,
  giving videos at evaluations 5, 10, ..., 50.
- The cadence is calculated by the Torch launcher from the requested video
  count and number of evaluations. Short smoke runs therefore still produce
  useful progress videos.
- Each video contains one deterministic evaluation episode and is logged as
  `eval/video` in the task-specific W&B run.
- `summarize_diverse_continual.py --upload-wandb` uploads the completed
  cross-run forward-transfer/AUC table and aggregate gains after all jobs
  finish.

The three compared algorithms use the same recording cadence. Video rollouts
are evaluation-only and do not enter replay or change the training budget.
