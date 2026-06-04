# KioMind — AI Voice Kiosk

An AI-powered voice kiosk system built on a Rockchip RK3588 SBC (e.g., Orange Pi 5).  
It detects approaching customers via camera and ultrasonic sensor, then activates an interactive ordering screen where users place orders by voice using OpenAI Whisper and GPT.

---

## Documentation

- [Model Conversion Guide](docs/model_conversion.md) — YOLO11n fine-tuning → ONNX → RKNN (RKNN Toolkit2 v1.6.0)

---

## Overview

KioMind replaces a traditional touch-only kiosk with a hands-free, voice-driven ordering experience. A background detection loop continuously monitors the kiosk proximity. When a person is detected both visually (YOLO11n via RKNN) and physically (HC-SR04 ultrasonic sensor), the kiosk wakes up and invites the customer to speak their order. The spoken order is transcribed by Whisper and parsed by GPT into structured menu items, which are then added to the cart automatically.

---

## Features

- **Proximity-triggered display control** — screen turns on only when a customer is nearby (YOLO person detection + ultrasonic sensor), and turns off after 10 seconds of no presence
- **Voice ordering** — 5-second microphone recording in the browser, sent to Whisper for Korean speech-to-text transcription
- **GPT-based order parsing** — natural language order text is converted to a structured list of menu items and quantities
- **Korean TTS feedback** — order confirmation and guidance are read back to the customer via gTTS
- **Tab-based menu UI** — categorized menu (coffee, tea, dessert) rendered in a 1080×1920 kiosk layout
- **Idempotent display commands** — redundant screen-state transitions are skipped via `screen_status.json`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hardware | Rockchip RK3588 (Orange Pi 5 or equivalent) |
| Object Detection | YOLO11n fine-tuned on custom human dataset, converted to RKNN (rknnlite) |
| Proximity Sensor | HC-SR04 ultrasonic via `gpiod` (libgpiod v2) |
| Backend | Python 3, Flask |
| Speech-to-Text | OpenAI Whisper API (`whisper-1`) |
| Order Parsing | OpenAI GPT (`gpt-3.5-turbo`) |
| Text-to-Speech | gTTS (Google Text-to-Speech) |
| Audio Processing | pydub + ffmpeg |
| Frontend | Vanilla HTML/CSS/JS (Jinja2 template) |

---

## Project Structure

```
KioMind/
├── main.py                      # Flask app + background detection loop
├── object_detection.py          # YOLO11n RKNN person detector wrapper
├── ultrasonic_sensor.py         # HC-SR04 distance measurement via gpiod
├── object_and_sensor_test.py    # Hardware integration test script
├── requirements.txt
├── .env.example
├── docs/
│   └── model_conversion.md      # YOLO11n fine-tuning → ONNX → RKNN guide
├── templates/
│   └── kiosk.html               # Full kiosk UI (1080×1920, voice + touch)
├── yolo11n_rknn_model/
│   ├── yolo11n-rk3588.rknn      # RKNN model fine-tuned on custom human dataset
│   └── metadata.yaml
├── kiomind_GPiO/
│   └── test.py                  # Standalone ultrasonic sensor test (OPi.GPIO)
└── test/
    ├── human-detect/
    │   └── object_detect_test.py
    └── kiomind_GPiO/
        └── test.py
```

---

## Hardware Requirements

- Rockchip RK3588-based board (e.g., Orange Pi 5 Plus)
- USB webcam or MIPI CSI camera (accessed as `/dev/video0`)
- HC-SR04 ultrasonic sensor wired to GPIO:
  - TRIG → `gpiochip1` pin 4 (gpio01_A4)
  - ECHO → `gpiochip1` pin 8 (gpio01_B0)
- Display connected via HDMI (controlled by `xset dpms`), portrait orientation (1080×1920)

---

## Prerequisites

```bash
# System packages
sudo apt install python3-pip ffmpeg libgpiod-dev

# Python packages — see requirements.txt
# Note: uses openai>=1.0.0 (v1.x API)
pip install -r requirements.txt

# RKNN Lite runtime (install from Rockchip's official release)
pip install rknnlite
```

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...

# Optional: override the default RKNN model path
# RKNN_MODEL_PATH=yolo11n_rknn_model/yolo11n-rk3588.rknn
```

> **Never commit `.env` to version control.** It is already listed in `.gitignore`.

---

## Running the Application

```bash
# Run on the RK3588 board (requires physical camera + GPIO)
python main.py
```

The Flask server starts on `http://0.0.0.0:5000`.  
Open `http://localhost:5000` in a Chromium kiosk window, or configure it to launch on boot.

For hardware testing before running the full app:

```bash
# Test object detection + ultrasonic sensor together
python object_and_sensor_test.py

# Test ultrasonic sensor only (gpiod)
python test/kiomind_GPiO/test.py

# Test ultrasonic sensor only (OPi.GPIO)
python kiomind_GPiO/test.py

# Test object detection with live bounding box overlay
python test/human-detect/object_detect_test.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serves the kiosk HTML page |
| POST | `/recognize` | Accepts `.webm` audio, returns Whisper transcription |
| POST | `/ai-order` | Accepts transcribed text, returns parsed order list (GPT) |
| POST | `/tts` | Accepts text, generates `voice.mp3` via gTTS |
| GET | `/voice` | Returns the generated `voice.mp3` |
| POST | `/display-control` | Turns the screen on/off and switches the active screen |
| GET | `/screen-status` | Returns current screen name and display state |

---

## How It Works

```
Camera + Ultrasonic Sensor
         │  (background thread, ~1fps)
         ▼
 Person detected AND within 100cm?
         │ YES
         ▼
  POST /display-control → xset dpms force on
  Browser shows "type-screen" (menu + voice button)
         │
         ▼
  Customer presses [Voice Order] button
  → 5-second mic recording (MediaRecorder, audio/webm)
  → POST /recognize → Whisper API → Korean text
  → POST /ai-order  → GPT parsing  → [{name, quantity}, ...]
  → Cart updated automatically
         │
  No person for 10 seconds
         │
         ▼
  POST /display-control → xset dpms force off
```

---

## Model

The RKNN model (`yolo11n_rknn_model/yolo11n-rk3588.rknn`) is a custom-trained model:

- **Base**: YOLO11n pretrained on COCO
- **Fine-tuned on**: Custom human detection dataset from Roboflow
- **Training**: 10 sessions on Windows (GTX 1060 Ti), approximately 2 hours total
- **Conversion**: ONNX → RKNN via RKNN Toolkit2 v1.6.0 on Ubuntu 22.04 (native, not Docker)

See [docs/model_conversion.md](docs/model_conversion.md) for the full pipeline.

---

## Menu

Menu items and prices are defined in `MENU_NAMES` and `PRICE` in `main.py`.  
See `main.py` for the full list.

---

## License

This project is for **personal and research use only**.  
The YOLO11n model is provided under the [AGPL-3.0 License](https://ultralytics.com/license) by Ultralytics.  
OpenAI APIs are subject to [OpenAI's Terms of Use](https://openai.com/policies/terms-of-use).
