"""
Run this on the Raspberry Pi 4.
Captures from the Pi camera, runs inference, and prints results to the terminal.

Install:
    pip install torch torchvision timm numpy pillow picamera2
"""

import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from picamera2 import Picamera2

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "xo_model.pth"
CLASS_NAMES = ["O", "X", "empty"]  # alphabetical order from ImageFolder

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
print("Camera ready — press Ctrl+C to stop\n")

# ── loop ───────────────────────────────────────────────────────────────────────
while True:
    frame = cam.capture_array()
    img = Image.fromarray(frame).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())

    print(f"Prediction: {CLASS_NAMES[predicted_idx]}  ({confidence:.2%})")

    time.sleep(1)  # capture every second, adjust as needed
