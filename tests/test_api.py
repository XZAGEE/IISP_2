import io
import json
import torch
from fastapi.testclient import TestClient
from src.app import app
import pytest

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "AudioCNN"
    assert data["model_version"] == "1.0.0"
    assert len(data["classes"]) == 4
    assert data["accuracy"] >= 80

def test_predict(client):
    with open("samples/test_yes.wav", "rb") as audio_file:
        buffer = io.BytesIO(audio_file.read())
    response = client.post(
        "/predict",
        files={"file": ("test_yes.wav", buffer, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["yes", "no", "up", "down"]
    assert 0 <= data["confidence"] <= 1
    assert data["latency_ms"] >= 0
    assert data["model_version"] == "1.0.0"
    assert len(data["probabilities"]) == 4

def test_predict_rejects_non_wav(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"some text", "text/plain")}
    )
    assert response.status_code == 400
