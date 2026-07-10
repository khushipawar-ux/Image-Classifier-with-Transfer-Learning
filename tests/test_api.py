import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.app.main import app

@pytest.fixture
def client():
    """Pytest fixture to yield TestClient executing FastAPI lifespan events."""
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Tests the health check endpoint returns proper keys and status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "classes" in data
    assert isinstance(data["classes"], list)

def test_predict_endpoint_invalid_file_type(client):
    """Tests that the API rejects non-image file uploads with 400 Bad Request."""
    files = {"file": ("test.txt", b"not-an-image-file-contents", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]

def test_predict_endpoint_missing_file(client):
    """Tests that the API returns 422 Unprocessable Entity when no file is provided."""
    response = client.post("/predict")
    assert response.status_code == 422

def test_predict_endpoint_inference(client):
    """
    Tests the prediction flow. If the model is loaded, it checks that the prediction
    returns correct probability distributions and latency stats. If the model is not loaded,
    it verifies that the API returns a 503 Service Unavailable error.
    """
    # 1. Create a dummy image in memory
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    files = {"file": ("mock_shape.jpg", img_byte_arr, "image/jpeg")}
    
    # 2. Check if the model is currently served
    health_response = client.get("/health")
    model_loaded = health_response.json().get("model_loaded", False)
    
    # 3. Post to prediction endpoint
    response = client.post("/predict", files=files)
    
    if model_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "latency_ms" in data
        assert data["predicted_class"] in ["circle", "square", "triangle"]
        assert len(data["probabilities"]) == 3
        assert 0.0 <= data["confidence"] <= 1.0
    else:
        assert response.status_code == 503
        assert "Model is not loaded" in response.json()["detail"]
