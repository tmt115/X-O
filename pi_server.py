"""
Run this on the Raspberry Pi 4.
Captures from the Pi camera, runs inference with the PyTorch EfficientNet-B0
model, and serves results over HTTP.
Saves captured images and a prediction log to the SD card.

Install on Pi:
    pip install flask torch torchvision timm numpy pillow picamera2

SD card setup:
    - Insert the SD card and find its mount point with: lsblk
    - It typically mounts at /media/pi/<LABEL> or /mnt/sdcard
    - Set SAVE_DIR below to that path
"""

import csv
import os
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from flask import Flask, jsonify
from picamera2 import Picamera2

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "xo_model.pth"       # path to your saved .pth file

# ImageFolder sorts class folders alphabetically: O=0, X=1, empty=2
CLASS_NAMES = ["O", "X", "empty"]

# Change to your SD card's mount point, e.g. /media/pi/SDCARD
SAVE_DIR  = "/media/pi/SDCARD/xo_data"
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
LOG_FILE  = os.path.join(SAVE_DIR, "predictions.csv")

# ── setup save directories ─────────────────────────────────────────────────────
os.makedirs(IMAGE_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image_file", "prediction", "confidence",
                         *[f"score_{c}" for c in CLASS_NAMES]])

print(f"Saving images to : {IMAGE_DIR}")
print(f"Saving log to    : {LOG_FILE}")

# ── model ──────────────────────────────────────────────────────────────────────
class XOClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cpu")   # Pi 4 has no CUDA
model = XOClassifier(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# Same transforms used during training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── camera ─────────────────────────────────────────────────────────────────────
cam = Picamera2()
cam.configure(cam.create_still_configuration())
cam.start()
print("Camera ready")

# ── server ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)


def capture_and_predict():
    """Capture a frame, run inference, save image + log entry to SD card."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Capture raw RGB frame
    frame = cam.capture_array()
    pil_img = Image.fromarray(frame).convert("RGB")

    # Save original image to SD card
    image_filename = f"{timestamp}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    pil_img.save(image_path)

    # Preprocess and run inference
    tensor = preprocess(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())
    scores = {name: round(float(probs[i].item()), 4) for i, name in enumerate(CLASS_NAMES)}

    # Append to CSV log
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            image_filename,
            CLASS_NAMES[predicted_idx],
            round(confidence, 4),
            *[scores[c] for c in CLASS_NAMES],
        ])

    return {
        "timestamp": timestamp,
        "image_saved": image_path,
        "prediction": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 4),
        "scores": scores,
    }


@app.route("/predict", methods=["GET"])
def predict():
    result = capture_and_predict()
    print(f"[{result['timestamp']}] {result['prediction']}  ({result['confidence']:.2%})")
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
