from flask import Flask, request, jsonify, send_from_directory, abort
from ultralytics import YOLO
import cv2
import threading
import time
import datetime
import os
import json

app = Flask(__name__)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("yolov8s.pt")

# -----------------------------
# FILE PATHS
# -----------------------------
ZONE_FILE = "zones.json"

# -----------------------------
# GLOBAL STATE
# -----------------------------
animal_zone = None
human_zone = None

video_running = False

animal_inside = False
human_inside = False

alerts = []

animal_in_counter = 0
animal_out_counter = 0
human_in_counter = 0
human_out_counter = 0

ENTER_THRESHOLD = 5
EXIT_THRESHOLD = 5

animal_ids = [15,16,17,18,19,20,21,22,23]
PERSON_ID = 0

# -----------------------------
# COORDINATE CONVERSION (Phone -> Laptop)
# -----------------------------
def get_pixel_coords(normalized_data, frame_width, frame_height):
    """
    Convert normalized coordinates from phone (0-1) to pixel coordinates.
    
    Args:
        normalized_data: dict with keys 'x1', 'y1', 'x2', 'y2' (0-1 range)
        frame_width: actual video frame width in pixels
        frame_height: actual video frame height in pixels
    
    Returns:
        tuple: (x1, y1, x2, y2) in pixel coordinates
    """
    x1 = int(normalized_data['x1'] * frame_width)
    y1 = int(normalized_data['y1'] * frame_height)
    x2 = int(normalized_data['x2'] * frame_width)
    y2 = int(normalized_data['y2'] * frame_height)
    return (x1, y1, x2, y2)

# -----------------------------
# SAVE & LOAD ZONES
# -----------------------------
def save_zones(updated_zone_type, zone_value):
    data = {}

    # Load existing data first
    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, "r") as f:
            data = json.load(f)

    # Update only one part
    data[updated_zone_type] = zone_value

    # Save back
    with open(ZONE_FILE, "w") as f:
        json.dump(data, f)

def load_zones():
    global animal_zone, human_zone

    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, "r") as f:
            data = json.load(f)

            if "animal_zone" in data:
                animal_zone = tuple(data["animal_zone"])

            if "human_zone" in data:
                human_zone = tuple(data["human_zone"])

# -----------------------------
# SET ZONES
# -----------------------------
@app.route("/set_zone", methods=["POST"])
def set_zone():
    global animal_zone, human_zone

    data = request.json

    # Store zones as NORMALIZED coordinates from phone (0-1 range)
    if "animal_zone" in data:
        az = data["animal_zone"]
        animal_zone = (az["x1"], az["y1"], az["x2"], az["y2"])
        save_zones("animal_zone", list(animal_zone))

    if "human_zone" in data:
        hz = data["human_zone"]
        human_zone = (hz["x1"], hz["y1"], hz["x2"], hz["y2"])
        save_zones("human_zone", list(human_zone))

    return jsonify({
        "message": "Zone updated (normalized coordinates stored)",
        "animal_zone": animal_zone,
        "human_zone": human_zone
    })

# -----------------------------
# GET ZONES
# -----------------------------
@app.route("/getzone", methods=["GET"])
def get_zone():
    return jsonify({
        "animal_zone": animal_zone,
        "human_zone": human_zone
    })

# -----------------------------
# SAVE ALERT IMAGE
# -----------------------------
def save_alert_image(frame, label):
    if not os.path.exists("alerts_images"):
        os.makedirs("alerts_images")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alerts_images/{label}_{timestamp}.jpg"

    cv2.imwrite(filename, frame)
    return filename

# -----------------------------
# VIDEO PROCESSING
# -----------------------------
def process_video():
    global video_running, alerts
    global animal_inside, human_inside
    global animal_in_counter, animal_out_counter
    global human_in_counter, human_out_counter

    cap = cv2.VideoCapture("final_video.mp4")

    # Get frame dimensions for coordinate conversion
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while video_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:

            break

        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)

        results = model(frame)

        animal_detected = False
        human_detected = False

        # Convert normalized zones to pixel coordinates for this frame
        animal_zone_px = None
        human_zone_px = None

        if animal_zone:
            animal_zone_px = get_pixel_coords(
                {'x1': animal_zone[0], 'y1': animal_zone[1], 'x2': animal_zone[2], 'y2': animal_zone[3]},
                frame_width, frame_height
            )

        if human_zone:
            human_zone_px = get_pixel_coords(
                {'x1': human_zone[0], 'y1': human_zone[1], 'x2': human_zone[2], 'y2': human_zone[3]},
                frame_width, frame_height
            )

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2

                # ANIMAL - Check against pixel coordinates
                if cls_id in animal_ids and animal_zone_px:
                    zx1, zy1, zx2, zy2 = animal_zone_px
                    if zx1 < cx < zx2 and zy1 < cy < zy2:
                        animal_detected = True

                # HUMAN - Check against pixel coordinates
                if cls_id == PERSON_ID and human_zone_px:
                    zx1, zy1, zx2, zy2 = human_zone_px
                    if zx1 < cx < zx2 and zy1 < cy < zy2:
                        human_detected = True

                color = (0,0,255) if cls_id in animal_ids else (255,0,0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # -----------------------------
        # ANIMAL LOGIC
        # -----------------------------
        if animal_detected:
            animal_in_counter += 1
            animal_out_counter = 0
        else:
            animal_out_counter += 1
            animal_in_counter = 0

        if animal_in_counter >= ENTER_THRESHOLD and not animal_inside:
            animal_inside = True
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000

            alerts.append({
                "type": "animal",
                "event": "ENTER",
                "time": round(t,2)
            })

        if animal_out_counter >= EXIT_THRESHOLD and animal_inside:
            animal_inside = False
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000

            alerts.append({
                "type": "animal",
                "event": "EXIT",
                "time": round(t,2)
            })

        # -----------------------------
        # HUMAN LOGIC
        # -----------------------------
        if human_detected:
            human_in_counter += 1
            human_out_counter = 0
        else:
            human_out_counter += 1
            human_in_counter = 0

        if human_in_counter >= ENTER_THRESHOLD and not human_inside:
            human_inside = True
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000

            image_path = save_alert_image(frame, "human")

            alerts.append({
                "type": "human",
                "event": "ENTER",
                "time": round(t,2),
                "image": image_path
            })

        if human_out_counter >= EXIT_THRESHOLD and human_inside:
            human_inside = False
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000

            alerts.append({
                "type": "human",
                "event": "EXIT",
                "time": round(t,2)
            })

        # DRAW ZONES (using pixel coordinates)
        if animal_zone_px:
            cv2.rectangle(frame, (animal_zone_px[0], animal_zone_px[1]),
                          (animal_zone_px[2], animal_zone_px[3]), (0,255,0), 2)

        if human_zone_px:
            cv2.rectangle(frame, (human_zone_px[0], human_zone_px[1]),
                          (human_zone_px[2], human_zone_px[3]), (255,255,0), 2)

        cv2.imshow("CCTV Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# START / STOP / ALERTS
# -----------------------------
@app.route("/start")
def start():
    global video_running
    if not video_running:
        video_running = True
        threading.Thread(target=process_video, daemon=True).start()
    return jsonify({"message": "started"})

@app.route("/stop_monitoring")
def stop_monitoring():
    global video_running
    video_running = False
    return jsonify({"message": "monitoring stopped"})

@app.route("/alerts")
def get_alerts():
    return jsonify(alerts)

@app.route("/alert", methods=["POST"])
def face_alert():
    data = request.json or {}
    alerts.append({
        "type": "face",
        "event": data.get("alert", "face_detected"),
        "name": data.get("name", "unknown"),
        "time": round(time.time(), 2)
    })
    return jsonify({"message": "alert stored"})


@app.route("/gethistory")
def get_history():
    """Return JSON list of images in the alerts_images folder with download URLs."""
    folder = "alerts_images"
    if not os.path.exists(folder):
        return jsonify([])

    files = []
    for name in sorted(os.listdir(folder), reverse=True):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            mtime = os.path.getmtime(path)
            files.append({
                "file": name,
                "url": f"/alerts_images/{name}",
                "mtime": round(mtime, 2)
            })

    return jsonify(files)


@app.route('/alerts_images/<path:filename>')
def serve_alert_image(filename):
    folder = "alerts_images"
    # prevent path traversal
    safe_path = os.path.join(folder, os.path.basename(filename))
    if not os.path.isfile(safe_path):
        return abort(404)
    return send_from_directory(folder, os.path.basename(filename))

@app.route("/clear")
def clear():
    global alerts
    alerts = []
    return jsonify({"message": "cleared"})

# -----------------------------
if __name__ == "__main__":
    load_zones()
    app.run(debug=True)