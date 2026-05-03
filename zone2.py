"""
AgriGuard AI — Zone Detector v3
================================
ZONE EDITOR CONTROLS:
  Left click empty area     → Add point to current zone
  Drag a point              → Move that point
  Right click a point       → Delete that point
  Drag inside polygon       → Move whole zone
  ENTER                     → Save current zone & start new one
  N                         → Name current zone (type in terminal)
  D                         → Done — lock all zones & start surveillance
  R                         → Delete last zone
  ESC                       → Quit

SURVEILLANCE CONTROLS:
  L                         → Toggle night mode (low-light enhancement)
  E                         → Jump back to zone editor
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

dataset_path    = "dataset"
CONFIG_FILE     = "config.json"
ALERTS_DIR      = "alerts"
FRAME_SKIP      = 10
ALERT_COOLDOWN  = 30
DISTANCE_THRESH = 10.0
POINT_RADIUS    = 10
HOVER_RADIUS    = 14

HUMAN_CLASS    = 0
ANIMAL_CLASSES = {15:"cat", 16:"dog", 17:"horse", 19:"cow", 20:"sheep"}

# One color per zone (cycles if more zones than colors)
ZONE_COLORS = [
    (0,   200, 200),   # cyan
    (255, 140,   0),   # orange
    (180,   0, 255),   # purple
    (0,   255, 140),   # mint
    (255,  60, 120),   # pink
    (60,  180, 255),   # sky blue
]

C_KNOWN   = (0,   255, 0)
C_UNKNOWN = (0,   0,   255)
C_ANIMAL  = (255, 165, 0)
C_WHITE   = (255, 255, 255)
C_HOVER   = (0,   255, 255)

os.makedirs(ALERTS_DIR, exist_ok=True)

# ── Load known names ──────────────────────────────────────────────────────────

try:
    with open(CONFIG_FILE) as f:
        known_names = set(json.load(f)["known_names"])
    print(f"[ZONE] Owners: {sorted(known_names)}")
except FileNotFoundError:
    print("[WARN] config.json not found — run face_train.py first.")
    known_names = set()

# ── Load YOLO ─────────────────────────────────────────────────────────────────

print("[ZONE] Loading YOLOv8n...")
model = YOLO("yolov8n.pt")
print("[ZONE] Ready.\n")

# ══════════════════════════════════════════════════════════════════════════════
#  NIGHT MODE
# ══════════════════════════════════════════════════════════════════════════════

night_mode = False

# CLAHE — Contrast Limited Adaptive Histogram Equalization
# Works per-channel in LAB color space so colors stay natural
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def enhance_night(frame):
    """
    Brighten and boost contrast for low-light frames.
    Pipeline:
      1. Convert BGR → LAB
      2. Apply CLAHE only to L (lightness) channel
      3. Convert back to BGR
      4. Mild gamma correction to lift shadows further
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Gamma correction (gamma < 1 = brighter shadows)
    gamma = 0.6
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    enhanced = cv2.LUT(enhanced, lut)

    return enhanced

def apply_mode(frame):
    """Return enhanced frame if night mode is on, original otherwise."""
    return enhance_night(frame) if night_mode else frame

# ══════════════════════════════════════════════════════════════════════════════
#  ZONE DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

class Zone:
    def __init__(self, name, color):
        self.name            = name
        self.color           = color
        self.points          = []        # list of [x, y]
        self.last_alert_time = 0

    def color_alert(self):
        """Brighter/red tint when breached."""
        return (0, 0, 255)

    def contains(self, cx, cy):
        if len(self.points) < 3:
            return False
        poly = np.array(self.points, dtype=np.int32)
        return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

    def can_alert(self):
        return time.time() - self.last_alert_time > ALERT_COOLDOWN

    def mark_alerted(self):
        self.last_alert_time = time.time()


# ── Completed zones list & current zone being drawn ───────────────────────────

zones         = []           # list of Zone objects (locked)
current_zone  = None         # Zone being actively drawn
all_locked    = False        # True when surveillance starts

def new_zone():
    """Start drawing a brand new zone."""
    global current_zone
    idx   = len(zones)
    color = ZONE_COLORS[idx % len(ZONE_COLORS)]
    name  = f"Zone {idx + 1}"
    current_zone = Zone(name, color)
    print(f"\n[EDITOR] Drawing {name}  (color #{idx+1})")
    print("  Click to place points. ENTER to save this zone.")

def save_current_zone():
    """Push current zone into the locked list."""
    global current_zone
    if current_zone and len(current_zone.points) >= 3:
        zones.append(current_zone)
        print(f"  ✔ {current_zone.name} saved ({len(current_zone.points)} pts) — {len(zones)} zone(s) total.")
        current_zone = None
    else:
        print("  Need at least 3 points to save a zone.")

# ── Editor mouse state ────────────────────────────────────────────────────────

dragging_idx   = -1
dragging_zone  = False
drag_start     = (0, 0)
zone_snapshot  = []
hover_idx      = -1
active_zone_i  = -1    # index in `zones` list being hovered/dragged (-1 = current_zone)

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def nearest_in(points, mx, my):
    best_i, best_d = -1, HOVER_RADIUS
    for i, (px, py) in enumerate(points):
        d = dist(mx, my, px, py)
        if d < best_d:
            best_d, best_i = d, i
    return best_i

def insert_on_nearest_edge(points, mx, my):
    """Insert (mx,my) on the closest edge of the polygon."""
    if len(points) < 2:
        points.append([mx, my])
        return len(points) - 1
    best_i, best_d = 0, float("inf")
    n = len(points)
    for i in range(n):
        ax, ay = points[i]
        bx, by = points[(i+1) % n]
        seg = dist(ax,ay,bx,by)
        if seg == 0:
            d = dist(mx,my,ax,ay)
        else:
            t = max(0, min(1, ((mx-ax)*(bx-ax)+(my-ay)*(by-ay)) / seg**2))
            d = dist(mx, my, ax+t*(bx-ax), ay+t*(by-ay))
        if d < best_d:
            best_d, best_i = d, i
    points.insert(best_i+1, [mx, my])
    return best_i + 1

def mouse_cb(event, mx, my, flags, param):
    global dragging_idx, dragging_zone, drag_start, zone_snapshot
    global hover_idx, active_zone_i, current_zone

    if all_locked:
        return

    # ── MOVE ──────────────────────────────────────────────────────────────────
    if event == cv2.EVENT_MOUSEMOVE:
        # Update hover — check current zone first, then saved zones
        hover_idx     = -1
        active_zone_i = -1
        if current_zone:
            i = nearest_in(current_zone.points, mx, my)
            if i >= 0:
                hover_idx, active_zone_i = i, -1
        if hover_idx == -1:
            for zi, z in enumerate(zones):
                i = nearest_in(z.points, mx, my)
                if i >= 0:
                    hover_idx, active_zone_i = i, zi
                    break

        # Apply drag
        if dragging_idx >= 0:
            pts = current_zone.points if active_zone_i == -1 else zones[active_zone_i].points
            pts[dragging_idx] = [mx, my]
        elif dragging_zone:
            dx = mx - drag_start[0]
            dy = my - drag_start[1]
            pts = current_zone.points if active_zone_i == -1 else zones[active_zone_i].points
            for i, (ox, oy) in enumerate(zone_snapshot):
                pts[i] = [ox+dx, oy+dy]

    # ── LEFT DOWN ─────────────────────────────────────────────────────────────
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Hit test: existing point?
        target_pts = None
        if hover_idx >= 0:
            target_pts = current_zone.points if active_zone_i == -1 else zones[active_zone_i].points
            dragging_idx = hover_idx
            return

        # Inside any saved zone polygon → drag whole zone
        for zi, z in enumerate(zones):
            if z.contains(mx, my):
                dragging_zone = True
                drag_start    = (mx, my)
                zone_snapshot = [p[:] for p in z.points]
                active_zone_i = zi
                return

        # Inside current zone polygon → drag whole
        if current_zone and len(current_zone.points) >= 3:
            poly = np.array(current_zone.points, dtype=np.int32)
            if cv2.pointPolygonTest(poly, (float(mx),float(my)), False) >= 0:
                dragging_zone = True
                drag_start    = (mx, my)
                zone_snapshot = [p[:] for p in current_zone.points]
                active_zone_i = -1
                return

        # Add new point to current zone
        if current_zone is None:
            new_zone()
        idx = insert_on_nearest_edge(current_zone.points, mx, my)
        dragging_idx  = idx
        active_zone_i = -1

    # ── LEFT UP ───────────────────────────────────────────────────────────────
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx  = -1
        dragging_zone = False

    # ── RIGHT CLICK → delete point ────────────────────────────────────────────
    elif event == cv2.EVENT_RBUTTONDOWN:
        if hover_idx >= 0:
            pts = current_zone.points if active_zone_i == -1 else zones[active_zone_i].points
            if len(pts) > 3:
                pts.pop(hover_idx)
                hover_idx = -1

# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_one_zone_editor(display, zone, is_current=False):
    """Draw a single zone in editor mode."""
    pts = zone.points
    n   = len(pts)
    color = zone.color

    if n >= 2:
        arr = np.array(pts, dtype=np.int32)
        if n >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [arr], color)
            cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i+1) % n]
            cv2.line(display, (ax,ay), (bx,by), color, 1)

    for i, (px, py) in enumerate(pts):
        is_h = (i == hover_idx and
                ((is_current and active_zone_i == -1) or
                 (not is_current and active_zone_i == zones.index(zone) if zone in zones else False)))
        cr = POINT_RADIUS + (3 if is_h else 0)
        cv2.circle(display, (px,py), cr, C_HOVER if is_h else color, 2)
        cv2.circle(display, (px,py), 4, C_WHITE, -1)

    # Mid-edge insert hints
    if n >= 2:
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i+1) % n]
            cv2.circle(display, ((ax+bx)//2, (ay+by)//2), 4, color, -1)

    # Zone name tag
    if n >= 1:
        lx, ly = pts[0]
        cv2.putText(display, zone.name, (lx+4, ly-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_editor_overlay(frame):
    display = frame.copy()

    # Saved zones
    for z in zones:
        draw_one_zone_editor(display, z, is_current=False)

    # Current zone being drawn
    if current_zone:
        draw_one_zone_editor(display, current_zone, is_current=True)

    # Status panel at bottom
    h, w = display.shape[:2]
    cv2.rectangle(display, (0, h-60), (w, h), (15,15,15), -1)

    zone_info = f"Saved zones: {len(zones)}  |  Drawing: {current_zone.name if current_zone else 'none'}"
    cv2.putText(display, zone_info, (8, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)

    tips = "LClick=add pt  Drag pt=move  Drag inside=move zone  RClick=del pt  ENTER=save zone  N=name  D=done  R=del last"
    cv2.putText(display, tips, (8, h-22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160,160,160), 1)

    ready = len(zones) >= 1 or (current_zone and len(current_zone.points) >= 3)
    hint = "Press D to start surveillance" if ready else "Draw at least one zone (3+ points), then press D"
    cv2.putText(display, hint, (8, h-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0,220,120) if ready else (100,100,100), 1)

    return display


def draw_surveillance_zones(display, breached_zones):
    """Draw all zones during surveillance."""
    for z in zones:
        breached = z in breached_zones
        color    = z.color_alert() if breached else z.color
        if len(z.points) < 3:
            continue
        pts = np.array(z.points, dtype=np.int32)
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.18 if not breached else 0.30, display,
                        0.82 if not breached else 0.70, 0, display)
        cv2.polylines(display, [pts], isClosed=True, color=color,
                      thickness=3 if breached else 2)
        lx, ly = z.points[0]
        cv2.putText(display, z.name, (lx+4, ly-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

# ══════════════════════════════════════════════════════════════════════════════
#  ALERT
# ══════════════════════════════════════════════════════════════════════════════

def trigger_alert(frame, zone, person_name):
    if not zone.can_alert():
        return
    zone.mark_alerted()
    ts       = time.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(ALERTS_DIR, f"{zone.name.replace(' ','_')}_{ts}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"\n{'='*50}")
    print(f"  INTRUDER ALERT — {zone.name}")
    print(f"  Time     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Evidence : {img_path}")
    print(f"{'='*50}\n")
    try:
        import requests
        requests.post("http://localhost:5000/alert", json={
            "zone": zone.name, "image": img_path, "timestamp": ts
        }, timeout=1)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  FACE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
#  NIGHT MODE HUD
# ══════════════════════════════════════════════════════════════════════════════

def draw_night_indicator(display):
    if night_mode:
        cv2.putText(display, "NIGHT MODE ON  [L to toggle]", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 2)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global all_locked, night_mode, current_zone

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    WIN = "AgriGuard AI"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_cb)

    print("=" * 52)
    print("  AgriGuard AI — Multi-Zone Detector v3")
    print("=" * 52)
    print("  Draw zones on camera, press D when done.\n")

    new_zone()   # start with Zone 1 ready to draw

    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

        # ── EDITOR MODE ───────────────────────────────────────────────────────
        if not all_locked:
            display = draw_editor_overlay(frame)
            cv2.imshow(WIN, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:            # ENTER — save current zone, start new
                save_current_zone()
                if len(zones) < 6:
                    new_zone()
                else:
                    print("  Max 6 zones reached. Press D to start surveillance.")

            elif key in (ord("n"), ord("N")):   # Name current/last zone
                target = current_zone if current_zone else (zones[-1] if zones else None)
                if target:
                    name = input(f"  Enter name for zone (current: '{target.name}'): ").strip()
                    if name:
                        target.name = name
                        print(f"  Renamed to '{name}'")

            elif key in (ord("d"), ord("D")):   # Done — lock all zones
                if current_zone and len(current_zone.points) >= 3:
                    save_current_zone()
                if not zones:
                    print("  Draw at least one zone first.")
                else:
                    all_locked = True
                    cv2.setWindowTitle(WIN, "AgriGuard AI — Surveillance Active")
                    print(f"\n[SURV] Surveillance started — {len(zones)} zone(s) active.")
                    print("  L = toggle night mode  |  E = edit zones  |  ESC = quit\n")

            elif key in (ord("r"), ord("R")):   # Delete last saved zone
                if zones:
                    removed = zones.pop()
                    print(f"  Removed {removed.name}.")
                elif current_zone:
                    current_zone.points.clear()
                    print("  Current zone cleared.")

            elif key == 27:
                break
            continue

        # ── SURVEILLANCE MODE ─────────────────────────────────────────────────
        processed = apply_mode(frame)
        display   = processed.copy()

        breached_zones = set()

        if frame_count % FRAME_SKIP == 0:
            yolo_res = model(processed, verbose=False)[0]

            for box in yolo_res.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])

                # Lower confidence threshold in night mode (image is noisier)
                threshold = 0.35 if night_mode else 0.45
                if conf < threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = y2   # feet level

                # Check every zone
                for z in zones:
                    if not z.contains(cx, cy):
                        continue

                    if cls_id == HUMAN_CLASS:
                        person = identify_person(processed)
                        if person in known_names:
                            bc    = C_KNOWN
                            label = f"Owner: {person.capitalize()} [{z.name}]"
                        else:
                            bc    = C_UNKNOWN
                            label = f"INTRUDER [{z.name}]"
                            breached_zones.add(z)
                            trigger_alert(frame, z, person)

                    elif cls_id in ANIMAL_CLASSES:
                        bc    = C_ANIMAL
                        label = f"{ANIMAL_CLASSES[cls_id].capitalize()} [{z.name}]"
                    else:
                        continue

                    cv2.rectangle(display, (x1,y1), (x2,y2), bc, 2)
                    cv2.putText(display, label, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, bc, 2)
                    cv2.circle(display, (cx,cy), 4, bc, -1)
                    break   # only report zone once per detection

                # Draw outside-zone humans in gray
                if cls_id == HUMAN_CLASS:
                    in_any = any(z.contains(cx,cy) for z in zones)
                    if not in_any:
                        cv2.rectangle(display, (x1,y1), (x2,y2), (160,160,160), 1)
                        cv2.putText(display, "Person", (x1, y1-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

        draw_surveillance_zones(display, breached_zones)
        draw_night_indicator(display)

        # ── Zone status sidebar ───────────────────────────────────────────────
        for i, z in enumerate(zones):
            breached = z in breached_zones
            status   = "BREACH" if breached else "Clear"
            sc       = (0,0,255) if breached else (0,200,100)
            cv2.putText(display, f"{z.name}: {status}",
                        (10, 34 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 2)

        h = display.shape[0]
        cv2.putText(display, "L=night mode  E=edit zones  ESC=quit",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140,140,140), 1)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key in (ord("l"), ord("L")):
            night_mode = not night_mode
            state = "ON" if night_mode else "OFF"
            print(f"[NIGHT] Night mode {state}")
        elif key in (ord("e"), ord("E")):
            all_locked = False
            cv2.setWindowTitle(WIN, "AgriGuard AI")
            print("[EDITOR] Back to zone editor.")

    video.release()
    cv2.destroyAllWindows()
    print("[ZONE] AgriGuard stopped.")

if __name__ == "__main__":
    main()