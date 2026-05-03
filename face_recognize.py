import cv2
import os
import json
import time
import requests
from deepface import DeepFace

# ── Config ────────────────────────────────────────────────────────────────────

KNOWN_FACES_DIR = "known-faces"
FALLBACK_DATASET_DIR = "dataset"
CONFIG_FILE  = "config.json"


def load_known_names():
    names = set()

    if os.path.isdir(KNOWN_FACES_DIR):
        for file_name in os.listdir(KNOWN_FACES_DIR):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                name = file_name.split("_")[0].strip().lower()
                names.add(name)
        if names:
            return names

    try:
        with open(CONFIG_FILE) as file_handle:
            return set(json.load(file_handle).get("known_names", []))
    except FileNotFoundError:
        return set()


known_names = load_known_names()
print(f"Loaded owners: {sorted(known_names)}")

# Alert cooldown keeps the server from getting spammed with repeat events.
ALERT_COOLDOWN_SECONDS = 30
last_alert_time = 0

# BUG FIX 3: DeepFace distance threshold — a low distance means a strong
# match. Without this check, DeepFace could return a weak match and still
# label the person as a known owner. Facenet typical threshold: 10.
DISTANCE_THRESHOLD = 10.0

FRAME_SKIP = 15   # run recognition every N frames

# ── Main loop ─────────────────────────────────────────────────────────────────

video = cv2.VideoCapture(0)
print("Starting AgriGuard AI Face Recognition...")
print("Press ESC to quit.\n")

frame_count = 0
name = "Scanning..."
color = (200, 200, 200)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        known_names = load_known_names()

        try:
            db_path = KNOWN_FACES_DIR if os.path.isdir(KNOWN_FACES_DIR) else FALLBACK_DATASET_DIR

            result = DeepFace.find(
                img_path=frame,
                db_path=db_path,
                model_name="Facenet",
                enforce_detection=False,
                silent=True
            )

            name = "unknown"
            color = (0, 0, 255)  # red for unknown

            if len(result) > 0 and len(result[0]) > 0:
                top = result[0].iloc[0]
                distance = top.get("distance", 999)

                # BUG FIX 3: Only accept match if distance is below threshold
                if distance < DISTANCE_THRESHOLD:
                    identity_path = top["identity"]
                    filename = os.path.basename(identity_path)
                    name = filename.split("_")[0].strip().lower()
                    color = (0, 255, 0)  # green for known

        except Exception as e:
            name = "unknown"
            color = (0, 0, 255)

        # ── Alert logic ───────────────────────────────────────────────────────
        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN_SECONDS:
            last_alert_time = now

            if name in known_names:
                print(f"[OK] Known person: {name}")
                alert_type = "known_detected"
            else:
                print(f"[ALERT] Unknown person detected!")
                alert_type = "unknown_detected"

            try:
                requests.post(
                    "http://localhost:5000/alert",
                    json={"name": name, "alert": alert_type},
                    timeout=1
                )
            except Exception:
                pass

    # ── Display ───────────────────────────────────────────────────────────────
    label = f"Person: {name.capitalize()}"
    cv2.putText(frame, label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show distance confidence in corner (debug helper)
    cv2.putText(frame, f"Frame: {frame_count}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("AgriGuard AI", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

video.release()
cv2.destroyAllWindows()
print("AgriGuard stopped.")