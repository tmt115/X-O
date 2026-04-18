"""
Run this on the Raspberry Pi 4.
Captures from the Pi camera, runs inference, prints results to the terminal,
and saves each image with the prediction overlaid on it.

Install:
    pip install torch torchvision timm numpy pillow picamera2
"""

import os
import time
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from picamera2 import Picamera2

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "xo_model.pth"
CLASS_NAMES = ["O", "X", "empty"]
SAVE_DIR = os.path.expanduser("~/xo_results")
os.makedirs(SAVE_DIR, exist_ok=True)

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

device = torch.device("cpu")
model = XOClassifier(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

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
print(f"Camera ready — saving annotated images to {SAVE_DIR}")
print("Press Ctrl+C to stop\n")

# ── loop ───────────────────────────────────────────────────────────────────────
while True:
    frame = cam.capture_array()
    img = Image.fromarray(frame).convert("RGB")

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())
    label = CLASS_NAMES[predicted_idx]

    # Draw prediction onto image
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    text = f"{label}  {confidence:.2%}"

    # Black outline + white text so it's readable on any background
    x, y = 10, 10
    for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
        draw.text((x+dx, y+dy), text, fill="black")
    draw.text((x, y), text, fill="white")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(SAVE_DIR, f"{timestamp}_{label}.jpg")
    annotated.save(save_path)

    print(f"Prediction: {label}  ({confidence:.2%})  → saved {save_path}")

    time.sleep(5)
