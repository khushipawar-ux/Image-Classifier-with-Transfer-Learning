# Image Classifier with Transfer Learning & ONNX FastAPI serving

**Aligns with**: Supervised ML 100% + Neural Networks

Fine-tune a pretrained ResNet-18 CNN on a domain-specific image dataset, convert to ONNX for optimized, lightweight inference, and expose via a FastAPI server featuring a premium glassmorphism Web UI.

Deploying models using ONNX is a direct production signal—it demonstrates an understanding of model serving constraints (latency, throughput, footprint) beyond just validation accuracy.

---

## 🏗️ Repository Architecture

```
Image-Classifier-with-Transfer-Learning/
├── .venv/                      # Python virtual environment
├── data/                       # Dataset directories (train / val)
├── models/                     # Saved PyTorch weights (.pth) & ONNX (.onnx) model
├── src/
│   ├── __init__.py
│   ├── dataset.py              # PyTorch ImageFolder Dataset loading & transforms
│   ├── generate_dataset.py     # Programmatic shape generator (Circles, Squares, Triangles)
│   ├── train.py                # Transfer learning fine-tuning script
│   ├── export.py               # ONNX exporter & validation script
│   └── app/
│       ├── __init__.py
│       ├── main.py             # FastAPI serving application (ONNX serving)
│       ├── static/             # CSS styling sheets and assets
│       └── templates/          # index.html Jinja2 template
├── tests/
│   └── test_api.py             # pytest test client suite
├── requirements.txt            # Project dependencies
└── README.md                   # This instruction guide
```

---

## ⚡ Quick Start

Follow these steps to run the complete pipeline and deploy the application.

### 1. Environment Setup

Create the virtual environment and install the package dependencies:

```powershell
# Create venv
py -m venv .venv

# Activate venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate the Shapes Dataset

Generate the training and validation images (Circles, Squares, Triangles):

```powershell
python src/generate_dataset.py
```

### 3. Fine-Tune the CNN Model

Run the transfer learning fine-tuning script. This loads a pretrained ResNet-18, replaces the classification head, and trains it for 5 epochs on CPU:

```powershell
python -m src.train
```

*Achieved validation accuracy: **98.67%***. Saved model to `models/best_model.pth`.

### 4. Export Model to ONNX

Compile the model to ONNX format and run the parity validation checks:

```powershell
# Set encoding variable for Windows Terminal emojis printing
$env:PYTHONIOENCODING="utf-8"

# Export model
python -m src.export
```

Generates `models/classifier.onnx` optimized for production serving.

### 5. Run Serving Server

Launch the FastAPI uvicorn server:

```powershell
python -m uvicorn src.app.main:app --port 8000 --reload
```

Open **`http://localhost:8000`** in your browser to view the interactive shapes classification dashboard!

---

## 🧪 Testing

Execute the automated tests using `pytest`:

```powershell
python -m pytest
```
All 4 test cases (health checks, invalid upload rejections, and dummy predictions) pass successfully.
