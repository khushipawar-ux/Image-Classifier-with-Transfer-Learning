import os
import io
import time
import json
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime as ort

# Global variables for model and classes
ort_session = None
class_names = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup (model loading) and shutdown.
    """
    global ort_session, class_names
    
    model_path = os.path.abspath("models/classifier.onnx")
    class_names_path = os.path.abspath("models/class_names.json")
    
    print("Loading deployment artifacts...")
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        print(f"Loaded class names: {class_names}")
    else:
        # Default fallback
        class_names = ["circle", "square", "triangle"]
        print(f"Class names mapping not found. Using default fallback: {class_names}")
        
    if os.path.exists(model_path):
        # Load the ONNX model using CPU execution provider
        ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"ONNX model loaded successfully from {model_path}")
    else:
        print(f"Warning: ONNX model file not found at {model_path}. Server running in uninitialized state.")
        
    yield
    # Cleanup resources (if any) during shutdown
    print("Shutting down API server...")

app = FastAPI(
    title="Image Classifier with Transfer Learning",
    description="Serve a fine-tuned ResNet-18 model compiled to ONNX for highly-optimized shape classification.",
    version="1.0.0",
    lifespan=lifespan
)

# Create folders if they don't exist
os.makedirs("src/app/static", exist_ok=True)
os.makedirs("src/app/templates", exist_ok=True)

# Mount static files and validation data folder (to serve samples)
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

templates = Jinja2Templates(directory="src/app/templates")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image to match training transformations:
    - Convert to RGB
    - Resize to 224x224
    - Normalize using ImageNet mean & std dev
    - Change dimensions to Channel-first (CHW) and add Batch dimension (1, 3, 224, 224)
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    
    # Resize matching validation set transforms
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    
    # Scale to [0, 1]
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # ImageNet Mean and Standard Deviation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Normalize
    image_np = (image_np - mean) / std
    
    # HWC to CHW
    image_np = image_np.transpose((2, 0, 1))
    
    # Add batch axis: (1, 3, 224, 224)
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

def softmax(x: np.ndarray) -> np.ndarray:
    """Computes softmax probabilities for raw network outputs (logits)."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: fastapi.Request):
    """Renders the main Web UI page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Returns the serving state and health status of the API."""
    model_loaded = ort_session is not None
    return {
        "status": "healthy" if model_loaded else "uninitialized",
        "model_loaded": model_loaded,
        "classes": class_names,
        "timestamp": time.time()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file upload, processes it, performs ONNX inference,
    and returns predicted probabilities and latency.
    """
    global ort_session, class_names
    
    if ort_session is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure that the model is trained and exported to ONNX."
        )
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        # Read file contents
        contents = await file.read()
        
        # Start timer
        start_time = time.perf_counter()
        
        # Preprocess
        input_tensor = preprocess_image(contents)
        
        # Run inference
        input_name = ort_session.get_inputs()[0].name
        raw_outputs = ort_session.run(None, {input_name: input_tensor})
        logits = raw_outputs[0]
        
        # Post-process probabilities
        probabilities = softmax(logits)[0]
        
        # End timer
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Format results
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        breakdown = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": breakdown,
            "latency_ms": round(latency_ms, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/api/samples")
async def get_samples():
    """
    Scans the validation directories to find and return one sample image path per class,
    providing easy testing capabilities from the web UI.
    """
    samples = {}
    shapes = ["circle", "square", "triangle"]
    for shape in shapes:
        shape_dir = os.path.join("data", "val", shape)
        if os.path.exists(shape_dir):
            files = [f for f in os.listdir(shape_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if files:
                # We return the first image found in the folder
                samples[shape] = f"/data/val/{shape}/{files[0]}"
    return samples
