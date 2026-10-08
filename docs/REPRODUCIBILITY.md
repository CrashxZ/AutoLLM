# Reproducing the corrected system

## Versions and evidence

`configs/corrected_followup.json` describes the corrected executor and joint
conflict-recovery configuration. Historical schedules remain byte-identical in
`experiments/historical/`; their old labels do not make new runs confirmatory.
The release runner copies layouts/seeds and labels all new episodes as follow-up
runs. These seeds have been examined before; they are not a new held-out test.

The historical raw 756-episode campaign, full 252-episode eight-vehicle follow-up,
video gallery and datasets are excluded from Git. Without those raw records you
can regenerate new episodes and reanalyze them, but cannot independently verify
or reproduce historical numerical claims. No claim of bitwise deterministic CARLA
execution is made. Use a new output directory for each campaign.

## Verify the included models

```bash
sha256sum -c models/SHA256SUMS
```

`models/provenance/` retains the original adapted-MAPPO training manifest, actor
metadata, acceptance and validation summaries. This actor was trained in the
CARLA-aligned lightweight environment, not directly in CARLA. Its inference and
adaptation logic is in `marl/carla_aligned_mappo.py` and the shared runner. The
learned candidate ranker is supervised cost regression, not an RL policy.

## Plan and run

With CARLA and the backend running under the example configuration:

```bash
# Inspect without connecting or running the simulator.
python scripts/run_campaign.py --preset eight --output data/experiments/eight_001 --dry-run
# Four episodes for a single paired block; useful as a local smoke test.
python scripts/run_campaign.py --preset smoke --max-blocks 1 --output data/experiments/smoke_001
# Nine paired blocks, covering every scenario and fleet size: 36 episodes.
python scripts/run_campaign.py --preset smoke --output data/experiments/demo_001
# 63 paired blocks at eight vehicles: 252 episodes, 63 per method.
python scripts/run_campaign.py --preset eight --output data/experiments/eight_001
# Full 189-block matrix: 756 episodes.
python scripts/run_campaign.py --preset full --output data/experiments/full_001
```

The four methods are FCFS-GAP, adapted MAPPO, deterministic MIND-CAV and learned
MIND-CAV. Each block shares seed, layout, map, initial dynamics and executor.
The example configuration disables Traffic Manager automatic lane changes and
pairwise collision avoidance, matching the evaluated executor setup. Camera
sensors and disk frames are disabled for comparisons. Each episode retains
metadata, events and trajectory; record and initial-state pairing checks must pass.
No outcome-dependent seed replacement or exclusion is performed. Wall timeout is
210 s, simulated deadline is 60 s, with 0.05-s simulator ticks and 0.5-s coordinator
steps. Inspect the saved manifest and schedule before drawing comparisons.

## Descriptive reanalysis and plots

```bash
python scripts/analyze_campaign.py --runs data/experiments/full_001 --output data/derived/full_001
python scripts/plot_scenario_results.py --analysis data/derived/full_001 --output results/full_001_figures
```

The figure script expects the full 756-episode matrix (21 episodes per cell).
Analysis itself supports smaller completed schedules. Fleet success requires all
planned centered lane-dwell completions, terminal DONE, deadline, no logged
collisions and no sampled sub-5-m shared-corridor separation violations. Restricted
time equals completion time on success, otherwise 60 s. Sampled proximity uses
center distance, not bumper clearance. Report <3/<5/<7 m episode counts separately
from collision-sensor notifications. Their counts overlap.

Descriptive 95% time CIs resample whole episodes within each cell; rate CIs are
Wilson intervals. Neither is a substitute for the historical primary inferential
analysis. Single-episode bootstrap intervals are degenerate and not informative.
The archived analysis code preserves the historical Wilcoxon/Pratt and correction
logic; its original protocol/raw inputs are needed to reproduce those inferences.

## Recording and local latency

```bash
python scripts/record_scenario_video_matrix.py --label demo_v2_001
python scripts/benchmark_mec_review_stages.py --output results/latency_001
```

The recorder creates 36 illustrative videos under `results/videos/demo_v2_001`,
with an index and manifest, and deletes its temporary JPEGs after full MP4 decode
validation. Do not run it concurrently with another CARLA campaign on the same API.
Latency benchmarking uses ten synthetic configurations per fleet, 3 warm-ups and
30 measured repetitions for each ranker. It measures local typed-input review and
buffered recording, not network/perception/dispatch/execution or durable fsync.

## Optional training

```bash
python scripts/train_adapted_mappo.py --help
python scripts/train_fleet_mappo.py --help
python scripts/generate_candidate_ranker_dataset.py --help
python scripts/train_candidate_ranker.py --help
```

Use the retained actor manifest for the historical training hyperparameters;
do not assume CLI defaults reproduce the included checkpoint. Different library
versions and hardware can change training. This release does not launch training
automatically or claim that all historical ranker training inputs are bundled.
