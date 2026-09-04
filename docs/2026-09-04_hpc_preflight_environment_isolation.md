# HPC preflight environment isolation

The diverse continual launcher configures forward-only evaluation before it
delegates to the shared Torch launcher. The shared launcher's repository test
preflight inherited those settings. Consequently, the legacy launcher test
observed `--no-eval-next-task` instead of testing its own default
`--eval-next-task` command.

`DRAFT.sh` now removes the experiment-specific evaluation, task-data, and run
namespace variables only while running the test preflight. Nested launcher
tests therefore exercise their own defaults. The actual experiment process
still receives the Sequence A forward-only settings after the preflight.
