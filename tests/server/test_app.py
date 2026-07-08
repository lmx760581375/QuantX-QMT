"""FastAPI workspace smoke tests."""

from fastapi.testclient import TestClient

from quantx.server.app import app


def test_workspace_config_validate_returns_formula_explain():
    client = TestClient(app)
    config = client.get("/api/configs/shuijiao_legacy.yaml").json()["content"]

    response = client.post("/api/configs/validate", json={"content": config})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["name"] == "shuijiao_legacy_formula"
    assert "buy_signal" in payload["formula_order"]
    assert payload["selector"]["where"] == "buy_signal"


def test_workspace_data_status_is_available():
    client = TestClient(app)

    response = client.get("/api/data/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_uri"] == "data/qlib_data_fixed"
    assert "calendar_days" in payload
