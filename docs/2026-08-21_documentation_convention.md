# Implementation documentation convention

Date: 2026-08-21

Every implementation change in this repository must add or update a Markdown
record under `docs/`. New records use this filename format:

```text
YYYY-MM-DD_<short_topic>.md
```

The record should state the scope, important design decisions, configuration
or compatibility changes, validation performed, and known limitations. If a
later change materially revises an earlier implementation, create a new dated
record and link back to the older one rather than erasing the experiment
history.
