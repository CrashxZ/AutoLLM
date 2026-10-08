# MIND-CAV v2 contributor instructions

Run commands from the repository root. Keep controller, validator and endpoint
changes in separately identified evaluation versions. Never rewrite historical
raw data or silently replace seeds. Use fresh output directories for campaigns.
Do not commit `.env`, recordings, generated results, caches or simulator binaries.

Backend tests: `python -m pytest -q server/tests` (CARLA Python API required for
lane-control tests, running simulator not required). Dashboard: `cd client/web &&
npm ci && npm run build`; run `npm run lint` and report existing failures honestly.
Python formatting: black (88 columns) and isort. Keep coordination logic modular;
no learned component may bypass deterministic admission. Do not start long training
or simulator campaigns without an explicit configuration and compute budget.
