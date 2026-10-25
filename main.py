import os
import sys
import time
import cv2
import json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow import keras
import pyttsx3 as tts


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


# Wrap your models so they point to the dynamic unpacking directory at runtime
MODEL_PATH = resource_path("sign_language_classifier.h5")
HAND_LANDMARKER_MODEL_PATH = resource_path("hand_landmarker.task")
LABELS_PATH = resource_path("labels.json")

model = keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    CLASS_LABELS = json.load(f)

engine = tts.init()
engine.setProperty("rate", 150)

print("Labels:", CLASS_LABELS)
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
print("Number of labels:", len(CLASS_LABELS))

# Must match how your training data was collected.
# FLIP_BEFORE_DETECTION: True only if training images were mirrored before detection.
# HAND_SLOTS: which 63-value block each hand goes into.
FLIP_BEFORE_DETECTION = False
HAND_SLOTS = {"Left": 0, "Right": 1}
DEBUG_TOP5 = False


def create_hand_landmarker():
    """
    Create a MediaPipe Tasks HandLandmarker.
    IMAGE mode matches the preprocessing script exactly.
    """
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
    )

    return vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(image_bgr, hand_landmarker):
    """
    Run MediaPipe HandLandmarker and return a 126-element feature vector.
    Left hand fills the first 63 values, right hand the last 63,
    and missing hands are left as zeros (same as preprocessing).
    Each hand is normalized relative to its own wrist coordinate.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hand_landmarker.detect(mp_image)

    # 2 hands * 21 landmarks * 3 coords
    coords = np.zeros(126, dtype=np.float32)

    if not result.hand_landmarks:
        return None, result

    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        if label not in HAND_SLOTS:
            continue
        start_offset = HAND_SLOTS[label] * 63

        # Wrist (index 0) is the anchor for this hand
        wrist = hand_landmarks[0]

        for lm_idx, lm in enumerate(hand_landmarks):
            base_pos = start_offset + (lm_idx * 3)
            coords[base_pos] = lm.x - wrist.x
            coords[base_pos + 1] = lm.y - wrist.y
            coords[base_pos + 2] = lm.z - wrist.z

    return coords, result


def main():
    # Open default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Stabilization + TTS state
    last_label = None
    stable_count = 0
    STABLE_THRESHOLD = 6
    COOLDOWN_SECONDS = 1.0
    last_spoken = 0.0

    hand_landmarker = create_hand_landmarker()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if FLIP_BEFORE_DETECTION:
                frame_bgr = cv2.flip(frame_bgr, 1)

            # 1) Extract features (126-dim) from up to two hands
            features, result = extract_hand_landmarks(frame_bgr, hand_landmarker)

            prediction_text = "No hand detected"

            if features is not None:
                # 2) Run the model once
                input_batch = features[np.newaxis, :]
                preds = model.predict(input_batch, verbose=0)
                class_idx = int(np.argmax(preds, axis=1)[0])

                if DEBUG_TOP5:
                    probs = preds[0]
                    top_indices = probs.argsort()[-5:][::-1]
                    print("top-5:", [(CLASS_LABELS[i], round(float(probs[i]), 3)) for i in top_indices])

                if 0 <= class_idx < len(CLASS_LABELS):
                    predicted_label = CLASS_LABELS[class_idx]
                    prediction_text = f"Predicted: {predicted_label}"

                    # 3) Stability logic
                    if predicted_label == last_label:
                        stable_count += 1
                    else:
                        last_label = predicted_label
                        stable_count = 1

                    now = time.time()

                    # 4) Text-to-speech with cooldown
                    if stable_count >= STABLE_THRESHOLD and now - last_spoken >= COOLDOWN_SECONDS:
                        print("SPEAK:", predicted_label)
                        engine.say(predicted_label)
                        engine.runAndWait()
                        last_spoken = now
                        stable_count = 0
                else:
                    last_label = None
                    stable_count = 0
            else:
                # No hand, clear state
                last_label = None
                stable_count = 0

            # 5) Mirror for display only, so it feels like a mirror to the user
            if FLIP_BEFORE_DETECTION:
                display = frame_bgr
            else:
                display = cv2.flip(frame_bgr, 1)

            h, w, _ = display.shape
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    for lm in hand_landmarks:
                        x = lm.x if FLIP_BEFORE_DETECTION else 1.0 - lm.x
                        x_px = int(x * w)
                        y_px = int(lm.y * h)
                        cv2.circle(display, (x_px, y_px), 4, (0, 255, 0), -1)

            # 6) Overlay prediction text
            cv2.putText(
                display,
                prediction_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Sign Language Interpreter", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        hand_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()