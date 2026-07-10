import os
import json
import torch
from torchvision import models
import torch.nn as nn
import numpy as np

def export_to_onnx(models_dir="models", onnx_filename="classifier.onnx"):
    """
    Loads the trained PyTorch ResNet-18 weights, exports the model to ONNX,
    and runs a verification check using ONNX Runtime.
    """
    model_path = os.path.join(models_dir, "best_model.pth")
    class_names_path = os.path.join(models_dir, "class_names.json")
    onnx_path = os.path.join(models_dir, onnx_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names mapping not found at {class_names_path}. Run train.py first.")
        
    # 1. Load class names and determine output dimensions
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Loading model with classes: {class_names}")
    
    # 2. Instantiate and load model weights
    print("Loading PyTorch model...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval() # Must be in eval mode for export!
    
    # 3. Define dummy input matching the shape expected by the model (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    
    # 4. Export the model
    print(f"Exporting model to ONNX format at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,       # Store the trained parameter weights inside the model file
        opset_version=11,          # ONNX opset version
        do_constant_folding=True,  # Inline constants/optimizations
        input_names=["input"],     # Input tensor name
        output_names=["output"],   # Output tensor name
        dynamic_axes={             # Enable variable batch size inference
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print("ONNX export complete!")
    
    # 5. Verify the model using ONNX
    print("Verifying ONNX model structure...")
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid.")
    
    # 6. Compare PyTorch and ONNX Runtime outputs to ensure numerical consistency
    print("Running verification inference...")
    import onnxruntime as ort
    
    # Run PyTorch inference
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
        
    # Run ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    # Convert dummy input to numpy
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_out = ort_outs[0]
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)
    print("Numerical validation successful!")
    print(f"PyTorch outputs: {torch_out}")
    print(f"ONNX Runtime outputs: {onnx_out}")
    print("Outputs match within tolerance! Model is ready for serving.")

if __name__ == "__main__":
    export_to_onnx()
