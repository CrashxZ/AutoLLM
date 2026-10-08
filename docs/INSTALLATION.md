# Installation

Use Python 3.10 and CARLA 0.9.15. Install the simulator separately; it is not bundled.
Commands below run from the repository root.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
# Install the matching CARLA Python API, or add its bundled egg to PYTHONPATH.
python -m pip install carla==0.9.15
cp .env.example .env
```

If the CARLA wheel is unavailable for your platform, use the simulator's matching
Python egg: set `CARLA_DIR` in `.env` for the backend, and export `PYTHONPATH` to
that egg for direct Python imports/tests. Set `CARLA_PYTHON_EGG` to the same egg
when using the camera recorder. The default API host is 127.0.0.1:2000; the Traffic
Manager uses port 8005. Set local paths in `.env`; never commit credentials.

Start CARLA in its installation directory:

```bash
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
```

In the repository root, start the backend (it loads the root `.env`):

```bash
python -m uvicorn server.server:app --host 127.0.0.1 --port 8000
```

Optional dashboard, in another terminal:

```bash
cd client/web
npm ci
npm run dev -- --host 127.0.0.1
```

The original dashboard is retained; experimental FCFS-GAP/MAPPO adapters are
selected through the campaign runner, not by assuming every dashboard mode is the
same evaluated baseline. Production dashboard build: `npm run build`.

Tests: `python -m pytest -q server/tests`. The lane-change restoration tests require
the CARLA Python API but no running simulator. Training additionally needs
`pip install -r requirements-training.txt`; supplied actor inference uses NumPy.
The CPU/GPU availability and versions of a fresh installation can differ from the
historical training host; bundled provenance describes that original run.

Video recording requires `ffmpeg` and `ffprobe` on PATH (for example, install the
Ubuntu `ffmpeg` package). The supplied dashboard builds, but its source snapshot
has no ESLint configuration: `npm run lint` currently reports that missing config.
