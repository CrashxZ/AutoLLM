# MIND-CAV v2

Multi-Intelligence Negotiation and Decision System for Connected Autonomous Vehicles.
A clean source release of the corrected, validator-constrained coordination system.

Vehicles submit typed proposals. The MEC checks the submitted plan, generates and
ranks joint alternatives when needed, and returns ACK, PLAN, or NACK. Only validated
plans are committed. Transactions retain proposal, decision, and execution records.
Admission is relative to a finite-horizon prediction model; it is not a formal
collision-avoidance guarantee. The reference configuration ranks deterministically;
the supplied learned ranker is an optional ablation. Neither requires an OpenAI key.

## Repository layout

| Directory | Contents |
|---|---|
| `server/coordination/` | Typed transactions, conflict prediction, validator, candidate generation/ranking, audit |
| `server/` | FastAPI/CARLA backend, lane-change executor, streaming, focused tests |
| `client/web/` | Existing React dashboard, preserved |
| `marl/` | Required baseline environments, MAPPO actor and training support |
| `scripts/` | Campaign runner, replay analysis, model training, recording and latency benchmark |
| `models/` | Small NumPy MAPPO actor and learned candidate ranker; provenance and checksums |
| `configs/` | Corrected follow-up settings |
| `experiments/historical/` | Unmodified historical schedule for seed/layout reference |
| `docs/` | Installation, reproducibility, scope, validation and source provenance |

Start with [installation](docs/INSTALLATION.md), then [reproduction](docs/REPRODUCIBILITY.md).
See [release scope](docs/RELEASE_SCOPE.md) before interpreting historical results.
Dashboard v2 support is tracked in the [UI migration TODO](docs/UI_TODO.md).
Raw campaigns, videos, datasets, manuscripts, credentials and build caches are not
part of this Git branch. No public raw-data download URL is supplied by this release.
