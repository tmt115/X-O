"""
Run this on your laptop.
Connects to the Pi server and prints the prediction result.

Install:
    pip install requests
"""

import requests
import sys

# ── config ─────────────────────────────────────────────────────────────────────
PI_IP = "raspberrypi.local"   # or use the Pi's IP address, e.g. "192.168.1.42"
PORT = 5000
URL = f"http://{PI_IP}:{PORT}/predict"


def get_prediction():
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        print(f"Prediction : {data['prediction']}")
        print(f"Confidence : {data['confidence']:.2%}")
        print("All scores :")
        for label, score in data["scores"].items():
            print(f"  {label:>6}: {score:.4f}")

    except requests.exceptions.ConnectionError:
        print(f"Could not connect to Pi at {PI_IP}:{PORT}")
        print("Make sure pi_server.py is running and the Pi is on the same network.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Request timed out — Pi may be busy or unreachable.")
        sys.exit(1)


if __name__ == "__main__":
    get_prediction()
