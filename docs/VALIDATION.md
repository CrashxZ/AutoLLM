# Release validation

- Backend: 101 focused tests passed in the existing Python 3.10 environment, with
  the CARLA 0.9.15 Python API available. Warnings include dependency deprecations.
- Dashboard: `npm ci` and `npm run build` passed. No dashboard behavior was changed.
- Frontend lint: blocked by the source snapshot's missing ESLint configuration;
  this is a known release limitation, not a passing check.
- Campaign dry runs: 9-block/36-episode smoke and 63-block/252-episode eight-vehicle
  schedules matched the requested design.
- Replay smoke: the portable analysis processed all four method records from the
  existing corrected single-seed block. No new CARLA outcome campaign was run.
- NumPy checkpoint SHA-256 checks passed; Python sources compile successfully.
- The recording entry point was checked with the configured CARLA Python egg;
  the CARLA wheel is not installed directly in this local validation environment.
- Credential-pattern scan found no project API tokens, Overleaf tokens, GitHub
  tokens or private keys in included source files. This is a bounded automated
  check, not an independent security audit.

A startup integration check initially revealed the static-dashboard route shadow
bug; it was repaired in this checkout and a no-lifespan regression test added.
The initial check entered the application's existing startup path and connected to
CARLA; it did not collect any evaluation episodes. Subsequent route checks avoid
startup, and no paper results were changed.

This validates the assembled release on the existing host. It does not establish
successful fresh-machine installation or scientific reproduction of excluded raw
campaigns. Training and full simulator campaigns were not launched for packaging.
