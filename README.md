# Sign Language Interpreter

A real-time sign language interpreter that uses your webcam to recognize hand signs and **reads them out loud**. It's built on Google's [MediaPipe](https://developers.google.com/mediapipe) hand tracking and a custom TensorFlow/Keras classifier.

> **Beta notice:** The current version (Beta 1.0) only recognizes **ASL fingerspelling** (the letters A–Z) and the **numbers 0–10**. Support for full signs is planned.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Platform](https://img.shields.io/badge/release-macOS%20(Apple%20Silicon)-lightgrey)

---

## Table of Contents

- [Demo](#demo)
- [Roadmap](#roadmap)
- [Features](#features)
- [Supported Signs](#supported-signs)
- [How It Works](#how-it-works)
- [Quick Start (Prebuilt App)](#quick-start-prebuilt-app--apple-silicon)
- [Running From Source](#running-from-source)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Training / Improving the Model](#training--improving-the-model)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Demo

Watch the demo video: **[Demo Video](https://drive.google.com/file/d/1n4d6U1cYlNv9xmkWu9R4qf6L688t3eqC/view?usp=sharing)**

---

## Roadmap

- [x] ASL fingerspelling (A–Z)
- [x] Numbers 0–10
- [x] Text-to-speech output
- [ ] Full-word / dynamic (moving) signs
- [ ] Sentence building / on-screen transcript
- [ ] Learning mode for practicing signs
- [ ] Builds for Intel Mac, Windows, and Linux
- [ ] Mobile build 
---

## Features

- **Real-time recognition** from your default webcam
- **Two-hand tracking:** detects up to 2 hands at once and handles left and right separately
- **Text-to-speech:** says each recognized sign out loud using `pyttsx3`
- **Stable predictions:** a sign is only spoken after it's held steady for several frames, with a cooldown between words
- **Mirrored view** with landmark dots drawn on your hands, so it feels natural to use
- **About 95% model accuracy** in Beta 1.0, up from 75% in earlier versions

---

## Supported Signs

The model classifies **38 classes** (from `labels.json`):

| Category | Signs |
| --- | --- |
| Letters | `a` `b` `c` `d` `e` `f` `g` `h` `i` `j` `k` `l` `m` `n` `o` `p` `q` `r` `s` `t` `u` `v` `w` `x` `y` `z` |
| Numbers | `0` `1` `2` `3` `4` `5` `6` `7` `8` `9` `10` |
| Other | `none` (no sign / neutral hand) |

---

## How It Works

```
Webcam frame ──► MediaPipe HandLandmarker ──► 126-value feature vector ──► Keras classifier ──► Label ──► On-screen text + speech
```

1. **Capture:** OpenCV grabs frames from the default webcam (`cv2.VideoCapture(0)`).
2. **Hand detection:** MediaPipe's `HandLandmarker` (`hand_landmarker.task`) finds up to **2 hands** and **21 landmarks** per hand.
3. **Feature extraction:** each landmark's `(x, y, z)` is taken **relative to that hand's wrist**, so the position of your hand in the frame doesn't matter.
   - Left hand → values `0–62`
   - Right hand → values `63–125`
   - A missing hand is filled with zeros
   - Total: `2 hands × 21 landmarks × 3 coords = 126` features
4. **Classification:** the vector goes into `sign_language_classifier.h5`, and the class with the highest probability is chosen.
5. **Stabilization:** the same prediction has to repeat for **6 frames in a row** (`STABLE_THRESHOLD`).
6. **Speech:** once stable, the label is spoken. Then there's a **1-second cooldown** (`COOLDOWN_SECONDS`) before the next word.
7. **Display:** the frame is mirrored, landmarks are drawn as green dots, and the prediction appears in the top-left corner.

---

## Quick Start (Prebuilt App — Apple Silicon)

The easiest way to try it on an **Apple Silicon Mac (M1/M2/M3/…)**:

1. Download the zip from the latest release: **[Beta 1.0](https://github.com/xhuntx/Sign-language-interpreter/releases/tag/Beta-1.0.0)**
2. Unzip it and open the app.
3. **The first launch asks for camera permission.** Grant it, then **open the app again** to start interpreting.

> **Tip:** If macOS blocks the app because it's from an unidentified developer, right-click the app → **Open**, or go to **System Settings → Privacy & Security** and choose **Open Anyway**.

---

## Running From Source

Works on any OS that supports the dependencies below (macOS, Windows, Linux).

### Prerequisites

- **Python 3.11**
- A webcam
- Speakers/headphones (for speech output)
- **Linux only:** `pyttsx3` needs `espeak` (`sudo apt install espeak`)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/xhuntx/Sign-language-interpreter.git
cd Sign-language-interpreter

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| `mediapipe` | 0.10.9 | Hand landmark detection |
| `opencv-contrib-python` | 4.11.0.86 | Webcam capture and display |
| `numpy` | 1.26.4 | Feature vectors |
| `tensorflow` | 2.16.1 | Model runtime |
| `keras` | 3.15.1 | Loading the `.h5` classifier |
| `pyttsx3` | 2.99 | Offline text-to-speech |
| `protobuf` | 3.20.3 | Required by MediaPipe |

### Run

```bash
python main.py
```

When it starts, it prints the loaded labels and the model's input/output shapes so you can check everything loaded correctly.

---

## Usage

1. Run the app and face the camera with good, even lighting.
2. Hold a sign steady in view. Green dots will appear on your hand.
3. The prediction shows as **`Predicted: <label>`** at the top-left.
4. Hold the sign for a moment and it'll be **spoken out loud**.
5. Press **`q`** to quit.

**Tips for better accuracy**
- Keep your whole hand inside the frame.
- Use a plain background and good lighting.
- Hold each sign still for about half a second before moving to the next one.

---

## Configuration

You can change these constants in `main.py`:

| Setting | Default | Description |
| --- | --- | --- |
| `FLIP_BEFORE_DETECTION` | `False` | Mirror frames before detection. Set to `True` only if your training data was mirrored. |
| `HAND_SLOTS` | `{"Left": 0, "Right": 1}` | Which 63-value block each hand fills in the feature vector. |
| `DEBUG_TOP5` | `False` | Print the top 5 predictions and their probabilities for each frame. |
| `STABLE_THRESHOLD` | `6` | How many consecutive matching frames are needed before speaking. |
| `COOLDOWN_SECONDS` | `1.0` | Minimum time between spoken words. |
| `engine.setProperty("rate", ...)` | `150` | Speech speed (words per minute). |

> **Warning:** `FLIP_BEFORE_DETECTION` and `HAND_SLOTS` **must match how the training data was collected**, or accuracy will drop sharply.

---

## Project Structure

```
Sign-language-interpreter/
├── main.py                       # App entry point: webcam → landmarks → prediction → speech
├── sign_language_classifier.h5   # Trained Keras classification model
├── hand_landmarker.task          # MediaPipe hand landmark model
├── labels.json                   # Class labels (index → sign)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md
```

`main.py` includes a `resource_path()` helper so the model files load correctly both from source and from a **PyInstaller** bundle (that's how the release app is built).

---

## Training / Improving the Model

The model is trained in a separate companion repo:

**[Sign-Language-Interpreter-Model-Creator](https://github.com/xhuntx/Sign-Language-Interpreter-Model-Creator)**

It includes:
- `dataSetCreator.py`: collect and process training data
- `datasetToH5.py`: train and export the classifier to `.h5`
- `processed_dataset/` and `model/`: data and model output

To use a model you trained:
1. Train it in the Model Creator repo.
2. Copy the new `sign_language_classifier.h5` (and an updated `labels.json` if you changed the classes) into this repo.
3. Make sure the label order in `labels.json` matches the model's output order.
4. Run `python main.py`.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `RuntimeError: Could not open webcam` | Another app may be using the camera, or permission hasn't been granted. On macOS, check **System Settings → Privacy & Security → Camera**. |
| App closes on first launch | Expected. The first run requests camera permission. Open it again. |
| Always shows "No hand detected" | Improve lighting and keep your whole hand in the frame. |
| Predictions are wrong or jumpy | Hold signs steadier, or set `DEBUG_TOP5 = True` to see what the model is considering. |
| `protobuf` / MediaPipe errors | Use the pinned `protobuf==3.20.3` and Python 3.11. |
| No speech on Linux | Install `espeak`. |

---

## Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

To improve the model itself, please contribute to the [Model Creator repo](https://github.com/xhuntx/Sign-Language-Interpreter-Model-Creator).

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">Made by <a href="https://github.com/xhuntx">xhuntx</a></p>
