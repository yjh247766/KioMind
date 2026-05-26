from flask import Flask, request, jsonify, send_file, render_template
from tempfile import NamedTemporaryFile
from difflib import get_close_matches
from gtts import gTTS
from object_detection import ObjectDetector
from ultrasonic_sensor import measure_distance
from pydub import AudioSegment
from dotenv import load_dotenv

import os
import cv2
import time
import json
import threading
import subprocess
import requests
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ================== Flask Configuration ==================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit

MENU_NAMES = ["아메리카노","카푸치노","카페모카","카라멜마끼아또","카페라떼","바닐라라떼","연유라떼","복숭아아이스티","레몬아이스티","녹차","캐모마일","티라미수","초코케이크","치즈케이크","허니브레드","소금빵"]
MENU = [{"name": name} for name in MENU_NAMES]

# Keys must exactly match MENU_NAMES to avoid KeyError in gpt_parse_order()
PRICE = {
    "아메리카노": 1500, "카푸치노": 2900, "카페모카": 3000, "카라멜마끼아또": 3700, "카페라떼": 2900,
    "바닐라라떼": 3400, "연유라떼": 3900, "복숭아아이스티": 3000, "레몬아이스티": 3000,
    "녹차": 2500, "캐모마일": 2500, "티라미수": 6500, "초코케이크": 6300, "치즈케이크": 6300,
    "허니브레드": 7000, "소금빵": 3500
}

# ================== Flask Routes ==================

@app.route("/")
def home():
    return render_template("kiosk.html")

@app.route("/display-control", methods=["POST"])
def display_control():
    data = request.get_json()
    screen = data.get("screen", "start-screen")
    display_on = data.get("on", True)
    print(f"[display-control] Request: screen={screen}, display_on={display_on}")

    # Compare with current state to avoid redundant transitions
    try:
        with open("screen_status.json") as f:
            current = json.load(f)
            if current.get("screen") == screen and current.get("display") == display_on:
                return jsonify({"status": "skipped"})  # No change, skip
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # File not found on first run, continue normally

    # Execute display power command
    if display_on:
        subprocess.call("DISPLAY=:0 xset dpms force on", shell=True)
    else:
        subprocess.call("DISPLAY=:0 xset dpms force off", shell=True)

    # Persist current state
    with open("screen_status.json", "w") as f:
        json.dump({"screen": screen, "display": display_on}, f)

    return jsonify({"status": "ok"})

@app.route("/screen-status")
def screen_status():
    try:
        with open("screen_status.json") as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default state if status file does not exist yet
        return jsonify({"screen": "start-screen", "display": False})

@app.route("/tts", methods=["POST"])
def tts_text():
    text = request.get_json().get("text", "")
    try:
        tts = gTTS(text, lang='ko')
        tts.save("voice.mp3")
        return jsonify({"status": "ok"})
    except Exception as e:
        print("[gTTS Error]", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/voice")
def voice():
    return send_file("voice.mp3", mimetype="audio/mpeg")

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        audio = request.files["audio"]
        with NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            audio.save(tmp.name)
            print("[DEBUG] Received file size:", os.path.getsize(tmp.name), "bytes")
            print("[DEBUG] Saved to:", tmp.name)

            # Convert webm to wav before sending to Whisper
            wav_path = tmp.name + ".wav"
            AudioSegment.from_file(tmp.name).export(wav_path, format="wav")

            with open(wav_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko"
                )
                print("[DEBUG] Whisper result:", result.text)
                return jsonify({"text": result.text})

    except Exception as e:
        print("[Whisper Error]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/ai-order", methods=["POST"])
def ai_order():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        print("[GPT Input Text]:", text)

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        orders = gpt_parse_order(text)
        print("[GPT Parsed Result]", orders)

        return jsonify(orders)
    except Exception as e:
        print("[GPT Error]", e)
        return jsonify({"error": str(e)}), 500

@app.route("/order", methods=["POST"])
def order():
    items = request.get_json()
    total = sum(item["price"] * item["quantity"] for item in items if "price" in item and "quantity" in item)
    print("[Order Complete]", items, f"Total: {total} KRW")
    tts_order(items, total)
    return jsonify({"msg": f"총 {total:,}원 결제 완료. 감사합니다!", "total": total})

# ================== GPT + TTS Helper Functions ==================

def find_closest_menu(name):
    """Return the closest matching menu item name using fuzzy string matching."""
    matches = get_close_matches(name, MENU_NAMES, n=1, cutoff=0.6)
    return matches[0] if matches else None

def gpt_parse_order(user_text):
    """
    Parse a natural language order string into a structured list using GPT.

    Args:
        user_text: Korean transcription from Whisper,
                   e.g. "아메리카노 두 잔이랑 치즈케이크 하나 주세요"

    Returns:
        A list of dicts: [{"name": "아메리카노", "quantity": 2, "price": 1500}, ...]
        Returns an empty list if GPT cannot identify any menu item.
    """
    menu_str = ', '.join([m['name'] for m in MENU])
    prompt = f"""
너는 카페 키오스크 주문 도우미야.
사용자 입력을 메뉴명과 수량 JSON 배열로 반드시 아래처럼 추출해서 출력해줘.
메뉴: {menu_str}
예:  
[{{"name":"아메리카노","quantity":2}},{{"name":"티라미수","quantity":1}}]
실제 사용자의 입력:
{user_text}
"""
    rsp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    txt = rsp.choices[0].message.content.strip()
    try:
        # Extract the JSON array from the GPT response
        first_bracket = txt.find("[")
        last_bracket = txt.rfind("]")
        json_part = txt[first_bracket : last_bracket + 1]
        arr = json.loads(json_part)
        # Attach price to each parsed item
        for item in arr:
            item["price"] = PRICE.get(item["name"], 0)
        return arr
    except Exception as e:
        print("[GPT Parse Error]", txt, e)
        return []

def tts_order(order_items, total):
    """Generate and save a Korean TTS summary of the completed order."""
    try:
        msg = "주문 내역은 "
        for item in order_items:
            msg += f"{item['name']} {item['quantity']}개, "
        msg += f"총 금액은 {total:,}원 입니다. 감사합니다."
        tts = gTTS(msg, lang='ko')
        tts.save("voice.mp3")
    except Exception as e:
        print("[gTTS Error]", e)

# ================== Detection Loop (Background Thread) ==================

def send_display_command(screen_name, display_on):
    """
    Send a display control request to the Flask server.
    Skips the request if the screen state has not changed.
    """
    try:
        # Load current persisted state
        current = {}
        try:
            with open("screen_status.json") as f:
                current = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # No state file yet, treat as unknown

        # Skip if state is already as requested
        if current.get("screen") == screen_name and current.get("display") == display_on:
            return

        # Only send request when state has changed
        requests.post("http://127.0.0.1:5000/display-control", json={
            "screen": screen_name,
            "on": display_on
        })

    except Exception as e:
        print("[Display Control Error]", e)

def detection_loop():
    """
    Background loop that continuously monitors for customer presence.

    - Reads frames from the camera at ~1 fps and runs YOLO person detection.
    - Also checks the ultrasonic sensor for proximity (within 100 cm).
    - If both conditions are met, turns the display on and shows the order screen.
    - If neither condition is met for 10 seconds, turns the display off.
    """
    model_path = os.getenv("RKNN_MODEL_PATH", "yolo11n_rknn_model/yolo11n-rk3588.rknn")
    detector = ObjectDetector(model_path)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            object_detected = detector.detect(img)
            person_nearby = measure_distance()

            print(f"Detected: {object_detected}, Nearby: {person_nearby}")

            if object_detected and person_nearby:
                send_display_command("type-screen", True)
                time.sleep(1)
                continue

            # Condition not met — start 10-second re-check window
            print("Conditions not met -> re-checking")
            start = time.time()
            recovered = False
            while time.time() - start < 10:
                ret, img = cap.read()
                if not ret:
                    break
                if detector.detect(img) and measure_distance():
                    recovered = True
                    break
                time.sleep(1)

            if not recovered:
                print("No presence for 10 seconds -> turning display off")
                send_display_command("start-screen", False)

    finally:
        cap.release()
        detector.release()
        cv2.destroyAllWindows()

# ================== Entry Point ==================

if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
