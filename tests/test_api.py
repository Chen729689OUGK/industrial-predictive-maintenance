
from fastapi.testclient import TestClient

from src.api import app, THRESHOLD


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["threshold"] == THRESHOLD


def test_prediction_low_risk():
    response = client.post(
        "/predict",
        json={
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 45.0,
            "Tool wear [min]": 200
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert data["risk"] == "Low risk"
    assert 0 <= data["failure_probability"] <= 1


def test_prediction_high_risk():
    response = client.post(
        "/predict",
        json={
            "Type": "L",
            "Air temperature [K]": 303.5,
            "Process temperature [K]": 315.0,
            "Rotational speed [rpm]": 1200,
            "Torque [Nm]": 65.0,
            "Tool wear [min]": 240
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["risk"] == "High risk"
    assert data["failure_probability"] >= THRESHOLD
