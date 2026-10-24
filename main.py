import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow import keras
import pyttsx3 as tts
import time

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# Wrap your models so they point to the dynamic unpacking directory at runtime
MODEL_PATH = resource_path("sign_language_classifier.h5")
HAND_LANDMARKER_MODEL_PATH = resource_path("hand_landmarker.task")

model = keras.models.load_model(MODEL_PATH)


engine = tts.init()
engine.setProperty("rate", 150)

CLASS_LABELS = [
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9",
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]
print("Model output shape:", model.output_shape)
print("Number of labels:", len(CLASS_LABELS))

def create_hand_landmarker():
    """
    Create a MediaPipe Tasks HandLandmarker for real-time video.
    We use VIDEO mode so it can track across frames.
    """
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    return vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(image_bgr, hand_landmarker, frame_timestamp_ms):
    """
    Run MediaPipe HandLandmarker and return a 126-element feature vector.
    Fills missing hand data with zeros if fewer than 2 hands are detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # Initialize a clean array of 126 zeros (2 hands * 21 landmarks * 3 coords)
    coords = np.zeros(126, dtype=np.float32)

    if not result.hand_landmarks:
        return None, result

    # Loop through detected hands (up to 2)
    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
        if hand_idx >= 2: 
            break  # Ignore a 3rd hand if it somehow appears
            
        # Calculate where this hand's data starts in our 126-element array
        # Hand 0 starts at index 0, Hand 1 starts at index 63
        start_offset = hand_idx * 63
        
        for lm_idx, lm in enumerate(hand_landmarks):
            base_pos = start_offset + (lm_idx * 3)
            coords[base_pos] = lm.x
            coords[base_pos + 1] = lm.y
            coords[base_pos + 2] = lm.z

    return coords, result



def main():
    # Open default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Stabilization + TTS state (same as your code)
    last_label = None
    stable_count = 0
    STABLE_THRESHOLD = 2
    COOLDOWN_SECONDS = 1.0
    last_spoken = 0.0

    # Create the Tasks HandLandmarker
    hand_landmarker = create_hand_landmarker()
    frame_timestamp_ms = 0  # monotonically increasing timestamp

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # We approximate 30 fps increments (doesn't need to be perfect,
            # just strictly increasing)
            frame_timestamp_ms += 33

            # 1) Extract features (63-dim) from the first hand, if any
            features, result = extract_hand_landmarks(
                frame_bgr,
                hand_landmarker,
                frame_timestamp_ms,
            )

            prediction_text = "No hand detected"

            if features is not None:
                # 2) Run your model once
                input_batch = features[np.newaxis, :]
                preds = model.predict(input_batch, verbose=0)
                class_idx = int(np.argmax(preds, axis=1)[0])
                
                # # --------------- Debug -----------------------
                # probs = preds[0]
                # top_indices = probs.argsort()[-5:][::-1]
                # print("top-5:", [(i, CLASS_LABELS[i], float(probs[i])) for i in top_indices])
                # print("argmax:", class_idx, CLASS_LABELS[class_idx])
                
                if 0 <= class_idx < len(CLASS_LABELS):
                    predicted_label = CLASS_LABELS[class_idx]
                    prediction_text = f"Predicted: {predicted_label}"

                    # 3) Stability logic (unchanged)
                    if predicted_label == last_label:
                        stable_count += 1
                    else:
                        last_label = predicted_label
                        stable_count = 1

                    now = time.time()

                    # 4) Text-to-speech with cooldown (unchanged)
                    if stable_count >= STABLE_THRESHOLD and now - last_spoken >= COOLDOWN_SECONDS:
                        print("SPEAK:", predicted_label)
                        engine.say(predicted_label)
                        engine.runAndWait()
                        last_spoken = now
                else:
                    last_label = None
                    stable_count = 0
            else:
                # No hand, clear state
                last_label = None
                stable_count = 0

            # 5) Draw landmarks for visualization (if any hands found)
            h, w, _ = frame_bgr.shape
            if result.hand_landmarks:
                # We only requested num_hands=1, but loop for generality
                for hand_landmarks in result.hand_landmarks:
                    for lm in hand_landmarks:
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        cv2.circle(frame_bgr, (x_px, y_px), 4, (0, 255, 0), -1)

            # 6) Overlay prediction text
            cv2.putText(
                frame_bgr,
                prediction_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Sign Language Interpreter", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        hand_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
