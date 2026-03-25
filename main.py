import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import pyttsx3 as tts
import time

# --- MODEL SETUP ---

MODEL_PATH = "sign_numbers_classifier.h5"
HAND_LANDMARKER_MODEL_PATH = "hand_landmarker.task"  # put this file in your project folder

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

# --- MEDIAPIPE TASKS SETUP ---

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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
        num_hands=1,
    )

    return vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(image_bgr, hand_landmarker, frame_timestamp_ms):
    """
    Run MediaPipe HandLandmarker on a BGR frame and return:

      - a 63-element feature vector: [x0, y0, z0, ..., x20, y20, z20]
      - the full result object (for drawing)

    x and y are normalized in [0, 1], z is relative depth.
    """
    # Convert BGR (OpenCV) -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Wrap into a MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Run the detector in VIDEO mode (needs timestamp)
    result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if not result.hand_landmarks:
        return None, result

    # First detected hand only
    hand_landmarks = result.hand_landmarks[0]

    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32), result


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
                1.0,
                (0, 0, 0),
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
