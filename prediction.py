import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pygame
import time
import os

# ─────────────────────────────────────────
# 1. LOAD MODELS
# ─────────────────────────────────────────
print("Loading models...")
eye_model   = tf.keras.models.load_model("eye_model.h5")
mouth_model = tf.keras.models.load_model("mouth_model.h5")
print("Models loaded successfully.")

# ─────────────────────────────────────────
# 2. MEDIAPIPE FACE MESH SETUP
# ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe landmark indices for left eye, right eye, mouth
# These are standard 468-point face mesh indices
LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249,
               263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE  = [33,  7,   163, 144, 145, 153, 154, 155,
               133, 173, 157, 158, 159, 160, 161, 246]
MOUTH      = [61, 146, 91, 181, 84, 17, 314, 405,
              321, 375, 291, 308, 324, 318, 402, 317,
              14, 87, 178, 88, 95, 185, 40, 39,
              37, 0, 267, 269, 270, 409]

# ─────────────────────────────────────────
# 3. ALARM SETUP
# ─────────────────────────────────────────
pygame.mixer.init()
alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing and os.path.exists("alarm.wav"):
        pygame.mixer.music.load("alarm.wav")
        pygame.mixer.music.play(-1)   # -1 = loop
        alarm_playing = True

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

# ─────────────────────────────────────────
# 4. HELPER: CROP ROI FROM LANDMARKS
# ─────────────────────────────────────────
IMG_SIZE = (224, 224)

def get_roi(frame, landmarks, indices, h, w, padding=10):
    """
    Given a list of landmark indices, compute the bounding box,
    add padding, crop the region from the frame, and return it.
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]

    x1 = max(0, min(x_coords) - padding)
    y1 = max(0, min(y_coords) - padding)
    x2 = min(w, max(x_coords) + padding)
    y2 = min(h, max(y_coords) + padding)

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

# ─────────────────────────────────────────
# 5. HELPER: PREPROCESS ROI FOR MODEL
# ─────────────────────────────────────────
def preprocess(roi):
    """
    Resize to 224x224, convert BGR→RGB, normalize to [0,1],
    and add batch dimension. Matches your training pipeline exactly.
    """
    img = cv2.resize(roi, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # shape: (1, 224, 224, 3)
    return img

# ─────────────────────────────────────────
# 6. DROWSINESS SCORING PARAMETERS
# ─────────────────────────────────────────
# score          = 0          # cumulative drowsiness score
# THRESHOLD      = 12         # score above this → alarm # original THRESHOLD= 15                                       #Changed
# EYE_WEIGHT     = 1          # added per frame when eye is closed
# YAWN_WEIGHT    = 2          # added per frame when yawning
# DECAY          = 2          # subtracted per frame when alert  # original DECAY= 1                                    #Changed

score          = 0          # cumulative drowsiness score
THRESHOLD      = 14        # score above this → alarm # original THRESHOLD= 15                                       #Changed
EYE_WEIGHT     = 2          # added per frame when eye is closed
YAWN_WEIGHT    = 1          # added per frame when yawning
DECAY          = 4          # subtracted per frame when alert  # original DECAY= 1                                    #Changed

# ─────────────────────────────────────────
# 7. CAMERA SELECTION HELPERS  (NEW)
# ─────────────────────────────────────────
def open_camera(index):
    """
    Try to open a camera with the DSHOW backend first (fast on Windows).
    Fall back to the default backend if DSHOW isn't available
    (e.g. on Linux/Mac) so the script still works cross-platform.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap

def scan_cameras(max_index=5):
    """
    Probe camera indices 0..max_index-1 and return the ones that
    actually open and deliver a frame.
    """
    available = []
    print("Searching for cameras...\n")

    for i in range(max_index):
        cap = open_camera(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"[{i}] Camera Found")
        cap.release()

    return available

def select_camera():
    """
    Scan for available cameras and let the user pick one at runtime.
    Falls back to manual index entry if scanning finds nothing.
    Returns (cap, camera_index, camera_label).
    """
    print("=" * 40)
    print("Driver Drowsiness Detection - Camera Setup")
    print("=" * 40)

    available = scan_cameras(max_index=5)

    if len(available) == 0:
        print("\nNo camera auto-detected. You can still try entering an index manually.")
        try:
            camera_index = int(input("Enter camera index to try: "))
        except ValueError:
            print("Invalid input.")
            exit()
    elif len(available) == 1:
        camera_index = available[0]
        print(f"\nOnly one camera found. Auto-selecting Camera {camera_index}.")
    else:
        print("\nAvailable Cameras:")
        for i in available:
            print(f"  {i} - Camera Index {i}")
        try:
            camera_index = int(input("\nSelect Camera Index: "))
        except ValueError:
            print("Invalid input.")
            exit()

    cap = open_camera(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {camera_index}.")
        exit()

    camera_label = f"Camera {camera_index}"
    return cap, camera_index, camera_label

# ─────────────────────────────────────────
# 8. OPEN WEBCAM  (runtime selection, NEW)
# ─────────────────────────────────────────
cap, camera_index, camera_label = select_camera()
print(f"\n{camera_label} opened. Press Q to quit.\n")

prev_time = time.time()

# ─────────────────────────────────────────
# 9. MAIN PREDICTION LOOP
# ─────────────────────────────────────────
try:
    while True:

        # ── Read frame with automatic recovery (NEW) ──
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed. Attempting to reconnect...")
            retry = 0
            while retry < 5:
                cap.release()
                time.sleep(1)
                cap = open_camera(camera_index)
                ret, frame = cap.read()
                if ret:
                    print("Camera reconnected successfully.")
                    break
                retry += 1
                print(f"Reconnect attempt {retry}/5 failed.")

            if not ret:
                print("Unable to recover camera. Exiting.")
                break

        h, w = frame.shape[:2]

        # ── Face mesh detection ──
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        eye_label   = "No Face"
        mouth_label = "No Face"
        eye_prob    = 0.0
        mouth_prob  = 0.0

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            # ── Crop LEFT eye ROI ──
            left_roi, left_box   = get_roi(frame, landmarks, LEFT_EYE,  h, w, padding=14) # original padding=12           #Changed
            right_roi, right_box = get_roi(frame, landmarks, RIGHT_EYE, h, w, padding=14) # original padding=12           #Changed
            mouth_roi, mouth_box = get_roi(frame, landmarks, MOUTH,     h, w, padding=12) # original padding=10           #Changed

            eye_closed  = False
            yawning     = False

            # ── Eye prediction (average left + right) ──
            if left_roi.size > 0 and right_roi.size > 0:
                left_pred  = eye_model.predict(preprocess(left_roi),  verbose=0)[0][0]
                right_pred = eye_model.predict(preprocess(right_roi), verbose=0)[0][0]
                eye_prob   = (left_pred + right_pred) / 2.0

                # eye: close=0, open=1  →  prob < 0.5 means closed
                if eye_prob < 0.4:  # original: eye_prob < 0.5                                                           #Changed
                    eye_closed = True
                    eye_label  = f"Closed ({eye_prob:.2f})"
                else:
                    eye_label  = f"Open ({eye_prob:.2f})"

                # Draw eye boxes
                cv2.rectangle(frame, (left_box[0],  left_box[1]),  (left_box[2],  left_box[3]),  (0,255,0), 1)
                cv2.rectangle(frame, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0,255,0), 1)

            # ── Mouth prediction ──
            if mouth_roi.size > 0:
                mouth_prob = mouth_model.predict(preprocess(mouth_roi), verbose=0)[0][0]

                # mouth: no_yawn=0, yawn=1  →  prob > 0.5 means yawning
                if mouth_prob > 0.85:  # original: mouth_prob > 0.5                                                           #Changed
                    yawning     = True
                    mouth_label = f"Yawn ({mouth_prob:.2f})"
                else:
                    mouth_label = f"No Yawn ({mouth_prob:.2f})"

                cv2.rectangle(frame, (mouth_box[0], mouth_box[1]), (mouth_box[2], mouth_box[3]), (255,0,255), 1)

            # # ── Update score ──
            # if eye_closed and yawning:
            #     score += EYE_WEIGHT + YAWN_WEIGHT
            # else:
            #     if eye_closed:
            #         score += EYE_WEIGHT
            #     if yawning:
            #         score += YAWN_WEIGHT
            # if not eye_closed and not yawning:
            #     score = max(0, score - DECAY)
            
            if eye_closed:
                if yawning:
                    score += EYE_WEIGHT + YAWN_WEIGHT
                else:
                    score += EYE_WEIGHT
            elif yawning:
                score = score + YAWN_WEIGHT
            if not eye_closed and not yawning:
                score = max(0, score - DECAY)
        else:
            # No face detected — slowly decay score
            score = max(0, score - DECAY)

        # ── Clamp score ──
        score = min(score, 15) #original score = min(score, 30)                                                           #Changed

        # ── Alarm logic ──
        if score >= THRESHOLD:
            play_alarm()
            alert_status = "DROWSY!"
            alert_color  = (0, 0, 255)   # red
        else:
            stop_alarm()
            alert_status = "Alert"
            alert_color  = (0, 255, 0)   # green

        # ── FPS calculation (NEW) ──
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time

        # ─────────────────────────────────────
        # 10. DRAW HUD ON FRAME (IMPROVED)
        # ─────────────────────────────────────
        # Semi-transparent top bar (taller to fit the extra rows)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        eye_conf_pct   = (eye_prob if eye_prob >= 0.5 else 1 - eye_prob) * 100 if result.multi_face_landmarks else 0.0
        mouth_conf_pct = (mouth_prob if mouth_prob >= 0.5 else 1 - mouth_prob) * 100 if result.multi_face_landmarks else 0.0

        cv2.putText(frame, f"Eye State        : {eye_label}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(frame, f"Mouth State      : {mouth_label}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(frame, f"Eye Confidence   : {eye_conf_pct:.1f}%",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(frame, f"Mouth Confidence : {mouth_conf_pct:.1f}%",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(frame, f"Score            : {score}",
                    (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(frame, f"FPS              : {fps:.1f}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
        cv2.putText(frame, f"Camera           : {camera_label}",
                    (10, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(frame, f"Status: {alert_status}",
                    (10, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.65, alert_color, 2)

        # ─────────────────────────────────────
        # 11. SHOW FRAME
        # ─────────────────────────────────────
        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'or'Q'):
            exit

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # ── Cleanup (always runs, even on crash) ──
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    face_mesh.close()
    print("Session ended.")