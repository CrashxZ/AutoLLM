"""A built dashboard must not shadow API routes; no simulator is started."""
from fastapi.testclient import TestClient
from server.server import app


def test_health_remains_available_with_static_dashboard():
    # No context manager: avoid running the CARLA startup lifespan.
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    api_index = next(i for i, route in enumerate(app.routes) if getattr(route, "path", None) == "/health")
    static = [i for i, route in enumerate(app.routes) if getattr(route, "name", None) == "web"]
    assert not static or static[0] > api_index
