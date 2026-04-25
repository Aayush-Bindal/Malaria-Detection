"""
MalariaNet — FastAPI Backend
=============================
Serves the malaria detection web app with:
- GET /        → serves the frontend (templates/index.html)
- POST /predict → accepts an uploaded cell image, returns prediction + Grad-CAM

The model (MobileNetV2 fine-tuned on the Kaggle malaria dataset) is loaded once
at startup using a lifespan context manager, so it stays in memory across requests.
"""

import io
import base64
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["Parasitized", "Uninfected"]

# ImageNet normalisation — must match the training transforms exactly
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Same val/test transform used during training — resize + normalise, no augmentation.
# This ensures the model receives inputs in the same distribution it was trained on.
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Model loading (runs once at startup)
# ---------------------------------------------------------------------------

# Global references — populated by the lifespan handler
model = None
device = None
grad_cam = None


def load_model():
    """Build the MobileNetV2 architecture and load trained weights.

    Returns the model, device, and a GradCAM object ready for inference.
    """
    # Use GPU if available; Render.com will use CPU
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the exact architecture used during training
    mdl = models.mobilenet_v2(weights=None)  # No pretrained — we load our own
    mdl.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.4),  # Dropout is disabled in eval mode, but we keep the layer
        nn.Linear(256, 2),
    )

    # Load the best checkpoint saved during training
    mdl.load_state_dict(
        torch.load(MODEL_PATH, map_location=dev, weights_only=True)
    )
    mdl = mdl.to(dev)
    mdl.eval()  # Set to evaluation mode — disables dropout and batch norm updates

    # Grad-CAM setup
    # Target layer: model.features[-1][0] — the Conv2d inside the last feature block.
    # This is identical to the target layer used in the notebook (Section 7).
    target_layer = mdl.features[-1][0]
    gcam = GradCAM(model=mdl, target_layers=[target_layer])

    return mdl, dev, gcam


# ---------------------------------------------------------------------------
# Lifespan — load model once, keep it in memory for all requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; release resources at shutdown.

    If best_model.pth is missing the app still starts — the frontend is
    accessible and /predict returns a clear 503 error telling the user
    to train the model first.  This avoids crashing on startup before
    training has been run.
    """
    global model, device, grad_cam

    if not Path(MODEL_PATH).exists():
        print(
            f"WARNING: '{MODEL_PATH}' not found. "
            "The app will start but /predict will be unavailable. "
            "Train the model first using malaria_model.ipynb."
        )
        device = torch.device("cpu")
    else:
        model, device, grad_cam = load_model()
        print(f"Model loaded on {device} from '{MODEL_PATH}'")

    yield  # App runs here

    # Cleanup
    if model is not None:
        del model, grad_cam
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MalariaNet",
    description="AI-powered malaria detection from red blood cell images",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow all origins so the frontend can call the API
# from any domain (important for local dev and deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory if it exists (for any future static assets)
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend from templates/index.html."""
    html_path = Path("templates/index.html")
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Ensure templates/index.html exists.",
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an uploaded cell image and return prediction + Grad-CAM overlay.

    Request:
        file: An image file (PNG, JPG, etc.)

    Response JSON:
        {
            "label": "Parasitized" or "Uninfected",
            "confidence": 0.97,
            "gradcam_image": "<base64 encoded PNG string>"
        }
    """
    # --- Check model is loaded ---
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train the model first by running "
                "malaria_model.ipynb (Kernel > Restart & Run All), then "
                "restart the server."
            ),
        )

    # --- Validate the uploaded file ---
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image.",
        )

    try:
        # Read the uploaded image bytes
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Open with PIL and convert to RGB
        # (handles PNG with alpha, grayscale, etc.)
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    except HTTPException:
        raise  # Re-raise our own exceptions
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded image: {str(e)}",
        )

    try:
        # --- Preprocess ---
        input_tensor = inference_transform(pil_image).unsqueeze(0).to(device)

        # --- Inference ---
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        label = CLASS_NAMES[predicted_class]

        # --- Grad-CAM ---
        # Generate heatmap for the predicted class
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # [H, W]

        # Prepare the original image as a float [0,1] numpy array at 224x224
        img_resized = pil_image.resize((224, 224))
        img_float = np.array(img_resized).astype(np.float32) / 255.0

        # Create overlay (jet colormap, image_weight=0.6 means 40% heatmap)
        overlay = show_cam_on_image(
            img_float, grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.6,
        )

        # Encode overlay as base64 PNG for JSON response
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        buffer.seek(0)
        gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "gradcam_image": gradcam_base64,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Run with uvicorn when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production; enable during dev
    )
