# TODO: update the dashboard for MIND-CAV v2

Status: planned; the retained dashboard is optional and does not yet expose the
evaluated v2 configurations. This task changes the interface, not the scientific
controllers, validators, endpoints, or historical results.

- [ ] Map existing UI controls to the actual v2 API and WebSocket contracts;
      document unsupported controls before changing them.
- [ ] Display backend-reported v2 enablement, proposer configuration, ranker
      variant, and executor/validator settings. Avoid UI-only configuration state.
- [ ] Support deterministic and learned MIND-CAV selection through the existing
      ranker API, with deterministic as the reference configuration.
- [ ] Expose FCFS-GAP and adapted MAPPO where supported by backend adapters.
      Distinguish these evaluated methods from legacy FCFS/MARL dashboard modes;
      identify any missing backend API support rather than relabeling controls.
- [ ] Show typed proposals, cooperation requests, ACK/PLAN/NACK decisions,
      approved joint plans, transaction IDs, and execution outcomes. Distinguish
      approval/commit from dispatch and physical completion.
- [ ] Keep overview/cameras, vehicle proposals, and MEC review in separate tabs;
      preserve UI state across tab changes.
- [ ] Add scenario and fleet-size selection consistent with runner definitions.
      Prevent interactive configuration/reset commands from interfering with an
      active experiment on the same backend.
- [ ] Make camera capture and disk recording explicit controls; avoid automatic
      screenshot accumulation, and show recording status and storage usage when
      supported by the backend.
- [ ] Keep credentials server-side; no API-key entry or secret exposure in the UI.
- [ ] Restore frontend lint configuration and add focused API/state integration
      checks. Verify the production build and existing dashboard functionality.
- [ ] Update installation and usage documentation with supported v2 controls and
      remaining differences from the experiment runner.

Acceptance: displayed settings match backend state; method labels match their
actual adapters; transactions show decision and execution separately; tab changes
retain state; experiments work without the UI; capture does not silently write
frames; existing API routes remain reachable when serving the built dashboard.
