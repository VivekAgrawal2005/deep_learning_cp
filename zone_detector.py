"""
AgriGuard AI — Zone Detection Module (v2 — Easy Zone Editor)
=============================================================
ZONE EDITOR CONTROLS:
  Left click on empty area  → Add a new point
  Left click + drag a point → Move that point
  Right click a point       → Delete that point
  Middle click / Ctrl+drag  → Drag entire zone
  ENTER                     → Lock zone & start surveillance
  R                         → Reset / redraw zone
  ESC                       → Quit

INSTALL:
  pip install ultralytics requests
"""

import cv2
import os
import json
import time
import math
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

dataset_path      = "dataset"
CONFIG_FILE       = "config.json"
ALERTS_DIR        = "alerts"
FRAME_SKIP        = 10
ALERT_COOLDOWN    = 30
DISTANCE_THRESH   = 10.0
POINT_RADIUS      = 10        # px — hit area for grabbing a point
HOVER_RADIUS      = 13

HUMAN_CLASS    = 0
ANIMAL_CLASSES = {15:"cat", 16:"dog", 17:"horse", 19:"cow", 20:"sheep"}

C_IDLE    = (0,   200, 200)
C_ALERT   = (0,   0,   255)
C_KNOWN   = (0,   255, 0)
C_UNKNOWN = (0,   0,   255)
C_ANIMAL  = (255, 165, 0)
C_POINT   = (255, 255, 255)
C_HOVER   = (0,   255, 255)
C_DRAG    = (0,   200, 255)
C_EDGE    = (100, 220, 220)

os.makedirs(ALERTS_DIR, exist_ok=True)

# ── Load known names ──────────────────────────────────────────────────────────

try:
    with open(CONFIG_FILE) as f:
        known_names = set(json.load(f)["known_names"])
    print(f"[ZONE] Owners loaded: {sorted(known_names)}")
except FileNotFoundError:
    print("[WARN] config.json not found — run face_train.py first.")
    known_names = set()

# ── Load YOLO ─────────────────────────────────────────────────────────────────

print("[ZONE] Loading YOLOv8n...")
model = YOLO("yolov8n.pt")
print("[ZONE] Ready.\n")

# ══════════════════════════════════════════════════════════════════════════════
#  ZONE EDITOR STATE
# ══════════════════════════════════════════════════════════════════════════════

zone_points  = []       # list of [x, y]  (mutable so we can edit in place)
zone_locked  = False

# drag state
dragging_idx  = -1      # index of point being dragged (-1 = none)
dragging_zone = False   # True when dragging the whole zone
drag_start    = (0, 0)  # mouse position when drag started
zone_snapshot = []      # copy of zone_points at drag start
hover_idx     = -1      # index of point under cursor

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def nearest_point(mx, my):
    """Return index of the closest zone point within HOVER_RADIUS, or -1."""
    best_i, best_d = -1, HOVER_RADIUS
    for i, (px, py) in enumerate(zone_points):
        d = dist(mx, my, px, py)
        if d < best_d:
            best_d, best_i = d, i
    return best_i

def point_in_zone_check(cx, cy, points):
    if len(points) < 3:
        return False
    poly = np.array(points, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

def centroid_of_zone():
    if not zone_points:
        return 0, 0
    xs = [p[0] for p in zone_points]
    ys = [p[1] for p in zone_points]
    return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))

# ── Mouse callback ─────────────────────────────────────────────────────────────

def mouse_cb(event, mx, my, flags, param):
    global dragging_idx, dragging_zone, drag_start, zone_snapshot, hover_idx

    if zone_locked:
        return

    ctrl_held = (flags & cv2.EVENT_FLAG_CTRLKEY)

    # ── Move ──────────────────────────────────────────────────────────────────
    if event == cv2.EVENT_MOUSEMOVE:
        hover_idx = nearest_point(mx, my)
        if dragging_idx >= 0:
            zone_points[dragging_idx] = [mx, my]
        elif dragging_zone:
            dx = mx - drag_start[0]
            dy = my - drag_start[1]
            for i, (ox, oy) in enumerate(zone_snapshot):
                zone_points[i] = [ox + dx, oy + dy]

    # ── Left button down ──────────────────────────────────────────────────────
    elif event == cv2.EVENT_LBUTTONDOWN:
        idx = nearest_point(mx, my)
        if idx >= 0:
            # Start dragging an existing point
            dragging_idx = idx
        elif ctrl_held or point_in_zone_check(mx, my, zone_points):
            # Ctrl held OR clicked inside polygon → drag whole zone
            dragging_zone = True
            drag_start    = (mx, my)
            zone_snapshot = [p[:] for p in zone_points]
        else:
            # Add new point
            # Insert it on the nearest edge for a clean polygon
            if len(zone_points) >= 2:
                best_i, best_d = 0, float("inf")
                n = len(zone_points)
                for i in range(n):
                    ax, ay = zone_points[i]
                    bx, by = zone_points[(i+1) % n]
                    # distance from point to segment
                    seg_len = dist(ax, ay, bx, by)
                    if seg_len == 0:
                        d = dist(mx, my, ax, ay)
                    else:
                        t = max(0, min(1, ((mx-ax)*(bx-ax)+(my-ay)*(by-ay)) / seg_len**2))
                        d = dist(mx, my, ax+t*(bx-ax), ay+t*(by-ay))
                    if d < best_d:
                        best_d, best_i = d, i
                zone_points.insert(best_i + 1, [mx, my])
                dragging_idx = best_i + 1
            else:
                zone_points.append([mx, my])
                dragging_idx = len(zone_points) - 1

    # ── Left button up ────────────────────────────────────────────────────────
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx  = -1
        dragging_zone = False

    # ── Right click → delete point ────────────────────────────────────────────
    elif event == cv2.EVENT_RBUTTONDOWN:
        idx = nearest_point(mx, my)
        if idx >= 0 and len(zone_points) > 3:
            zone_points.pop(idx)
            hover_idx = -1
            print(f"  Point deleted — {len(zone_points)} remaining")
        elif idx >= 0:
            print("  Need at least 3 points — can't delete.")

# ── Draw editor frame ─────────────────────────────────────────────────────────

def draw_editor(frame):
    """Draw the interactive zone editor overlay."""
    display = frame.copy()
    n = len(zone_points)

    if n >= 2:
        # Draw filled polygon (semi-transparent)
        pts = np.array(zone_points, dtype=np.int32)
        overlay = display.copy()
        if n >= 3:
            cv2.fillPoly(overlay, [pts], C_IDLE)
            cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)
        # Draw edges
        for i in range(n):
            ax, ay = zone_points[i]
            bx, by = zone_points[(i+1) % n]
            cv2.line(display, (ax,ay), (bx,by), C_EDGE, 1)

    # Draw points
    for i, (px, py) in enumerate(zone_points):
        is_hover = (i == hover_idx)
        is_drag  = (i == dragging_idx)
        outer_c  = C_DRAG if is_drag else (C_HOVER if is_hover else C_EDGE)
        inner_c  = C_POINT
        outer_r  = POINT_RADIUS + (3 if is_hover or is_drag else 0)
        cv2.circle(display, (px,py), outer_r, outer_c, 2)
        cv2.circle(display, (px,py), 4, inner_c, -1)
        # Show point number
        cv2.putText(display, str(i+1), (px+outer_r+2, py+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, outer_c, 1)

    # Mid-edge "+" hints (where next click will insert)
    if n >= 2:
        for i in range(n):
            ax, ay = zone_points[i]
            bx, by = zone_points[(i+1) % n]
            mx2, my2 = (ax+bx)//2, (ay+by)//2
            cv2.circle(display, (mx2,my2), 4, (120,220,220), -1)

    # Status bar
    h = display.shape[0]
    bar_y = h - 48
    cv2.rectangle(display, (0, bar_y), (display.shape[1], h), (20,20,20), -1)
    tips = [
        f"Points: {n}  |",
        "LClick empty = add pt  |",
        "Drag pt = move  |",
        "RClick pt = delete  |",
        "Drag inside = move zone  |",
        "ENTER = start  |  R = reset"
    ]
    cv2.putText(display, "  ".join(tips), (8, h-28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    cv2.putText(display, "ENTER to start surveillance  (need 3+ points)",
                (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0,220,120) if n >= 3 else (100,100,100), 1)

    return display

# ── Surveillance helpers ──────────────────────────────────────────────────────

def identify_person(frame):
    try:
        result = DeepFace.find(
            img_path=frame, db_path=dataset_path,
            model_name="Facenet", enforce_detection=False, silent=True
        )
        if len(result) > 0 and len(result[0]) > 0:
            top = result[0].iloc[0]
            if top.get("distance", 999) < DISTANCE_THRESH:
                fname = os.path.basename(top["identity"])
                return fname.split("_")[0].strip().lower()
        return "unknown"
    except Exception:
        return "unknown"

last_alert_time = 0

def trigger_alert(frame, name):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        return
    last_alert_time = now
    ts       = time.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(ALERTS_DIR, f"intruder_{ts}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"\n{'='*48}")
    print(f"  INTRUDER ALERT")
    print(f"  Time     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Evidence : {img_path}")
    print(f"{'='*48}\n")
    try:
        import requests
        requests.post("http://localhost:5000/alert",
                      json={"name": name, "image": img_path, "timestamp": ts},
                      timeout=1)
    except Exception:
        pass

def draw_zone_surveillance(display, intruder):
    """Draw locked zone during surveillance."""
    if len(zone_points) < 3:
        return
    pts = np.array(zone_points, dtype=np.int32)
    color = C_ALERT if intruder else C_IDLE
    overlay = display.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
    cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)
    lx, ly = zone_points[0]
    cv2.putText(display, "RESTRICTED ZONE", (lx+4, ly-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global zone_locked, zone_points, hover_idx

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    WIN = "AgriGuard AI"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_cb)

    print("=== AgriGuard Zone Editor ===")
    print("Click to place points, drag to adjust, right-click to remove.")
    print("Press ENTER when your zone looks good.\n")

    frame_count   = 0
    current_label = "Scanning..."

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

        # ── EDITOR MODE ───────────────────────────────────────────────────────
        if not zone_locked:
            display = draw_editor(frame)
            cv2.imshow(WIN, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(zone_points) >= 3:      # ENTER
                zone_locked = True
                cv2.setWindowTitle(WIN, "AgriGuard AI — Surveillance Active")
                print(f"[ZONE] Locked ({len(zone_points)} pts). Surveillance running.\n")
            elif key in (ord("r"), ord("R")):
                zone_points.clear()
                hover_idx = -1
                print("[ZONE] Cleared.")
            elif key == 27:
                break
            continue

        # ── SURVEILLANCE MODE ─────────────────────────────────────────────────
        display = frame.copy()
        intruder_in_zone = False

        if frame_count % FRAME_SKIP == 0:
            yolo_res = model(frame, verbose=False)[0]

            for box in yolo_res.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if conf < 0.45:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = y2    # feet-level centroid

                in_zone = point_in_zone_check(cx, cy, zone_points)

                if cls_id == HUMAN_CLASS:
                    if in_zone:
                        current_label = identify_person(frame)
                        if current_label in known_names:
                            bc    = C_KNOWN
                            label = f"Owner: {current_label.capitalize()}"
                        else:
                            bc               = C_UNKNOWN
                            label            = "INTRUDER"
                            intruder_in_zone = True
                            trigger_alert(frame, current_label)
                    else:
                        bc    = (180, 180, 180)
                        label = "Person (outside zone)"

                    cv2.rectangle(display, (x1,y1), (x2,y2), bc, 2)
                    cv2.putText(display, label, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2)
                    cv2.circle(display, (cx, cy), 4, bc, -1)

                elif cls_id in ANIMAL_CLASSES:
                    aname = ANIMAL_CLASSES[cls_id]
                    cv2.rectangle(display, (x1,y1), (x2,y2), C_ANIMAL, 2)
                    cv2.putText(display, aname.capitalize(), (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_ANIMAL, 2)

        draw_zone_surveillance(display, intruder_in_zone)

        # HUD
        status = "!! INTRUDER IN ZONE !!" if intruder_in_zone else "Zone: Clear"
        sc     = C_ALERT if intruder_in_zone else (0, 220, 120)
        cv2.putText(display, status, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, sc, 2)
        cv2.putText(display, "R = edit zone  |  ESC = quit",
                    (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key in (ord("r"), ord("R")):
            zone_locked = False
            cv2.setWindowTitle(WIN, "AgriGuard AI")
            cv2.setMouseCallback(WIN, mouse_cb)
            print("[ZONE] Back to editor — adjust and press ENTER again.")

    video.release()
    cv2.destroyAllWindows()
    print("[ZONE] Stopped.")

if __name__ == "__main__":
    main()