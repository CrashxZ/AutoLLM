# Release scope and changes

This source tree was assembled from the MIND-CAV-v2 development workspace and
selected AutoLLM follow-up tools. It is a separate checkout on branch `v2`; neither
original workspace nor historical campaigns were modified.

Included fixes preserve newer speed commands after lane-change termination and
preserve configured automatic-lane behavior after rejected prechecks. The opt-in
joint conflict-recovery candidate is enabled in the corrected follow-up preset.
The old predictor/executor speed setting is retained; this release does not claim
that its finite-horizon model exactly matches physical execution.

The default boot ranker is explicitly deterministic (`MIND_CAV_RANKER_VARIANT`).
The runtime ranker endpoint can select the bundled learned model. This is a release
configuration change, not a new controller evaluation. Other source edits improve
local paths and add portable entry points; the historical schedule is unmodified.

Raw data, screenshots, videos, paper sources, reviewer correspondence, scratch
investigations, caches, `.env` files and external simulator binaries are excluded.
The retained model provenance may contain original relative output paths. They
identify original artifacts and are not locations to overwrite during reproduction.

The core CARLA study uses scripted typed proposals and fixed goals. Including
model-training utilities does not establish that an LLM/VLM was deployed in those
comparisons. The learned cost ranker and MAPPO actor serve different purposes.
Single-MEC simulation, idealized inputs and finite-horizon checking do not establish
real V2X delivery, coordinated physical atomicity, formal safety or multi-MEC handoff.

No raw-data publication URL or new software license was invented for this release.
See source provenance and validation notes for exactly what was copied and checked.

A release packaging repair registers the built dashboard static mount after API
and WebSocket routes; otherwise a production build hides `/health` and other API
endpoints. A regression test checks route ordering without starting CARLA.
