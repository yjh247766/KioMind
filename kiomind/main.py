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
import openai
import threading
import subprocess
import requests

load_dotenv()

openai.api_key= os.getenv("OPENAI_API_KEY")


# ================== Flask 설정 ==================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 

MENU_NAMES = ["아메리카노","카푸치노","카페모카","카라멜마끼아또","카페라떼","바닐라라떼","연유라떼","복숭아아이스티","레몬아이스티","녹차","캐모마일","티라미수","초코케이크","치즈케이크","허니브레드","소금빵"]
MENU = [{"name": name} for name in MENU_NAMES]
PRICE = {
    "아메리카노": 1500, "카푸치노": 2900, "카페모카": 3000, "카라멜 마끼아또": 3700, "카페 라떼": 2900,
    "바닐라라떼": 3400, "연유라떼": 3900, "복숭아아이스티": 3000, "레몬아이스티": 3000,
    "녹차": 2500, "캐모마일": 2500, "티라미수": 6500, "초코케이크": 6300, "치즈케이크": 6300,
    "허니브레드": 7000, "소금빵": 3500
}

# ================== Flask 라우터 ==================

@app.route("/")
def home():
    return render_template("kiosk.html")

@app.route("/display-control", methods=["POST"])
def display_control():
    data = request.get_json()
    screen = data.get("screen", "start-screen")
    display_on = data.get("on", True)
    print(f"[display-control] 요청: screen={screen}, display_on={display_on}")
    # 현재 상태와 비교
    try:
        with open("screen_status.json") as f:
            current = json.load(f)
            if current.get("screen") == screen and current.get("display") == display_on:
                return jsonify({"status": "skipped"})  # 변경 없으면 무시
    except:
        pass  # 파일 없으면 계속 진행

    # 화면 제어 명령 실행
    if display_on:
        subprocess.call("DISPLAY=:0 xset dpms force on", shell=True)
    else:
        subprocess.call("DISPLAY=:0 xset dpms force off", shell=True)

    # 상태 저장
    with open("screen_status.json", "w") as f:
        json.dump({"screen": screen, "display": display_on}, f)

    return jsonify({"status": "ok"})

@app.route("/screen-status")
def screen_status():
    try:
        with open("screen_status.json") as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"screen": "start-screen", "display": False})

@app.route("/tts", methods=["POST"])
def tts_text():
    text = request.get_json().get("text", "")
    try:
        tts = gTTS(text, lang='ko')
        tts.save("voice.mp3")
        return jsonify({"status": "ok"})
    except Exception as e:
        print("[gTTS 오류]", e)
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
            print("[DEBUG] 받은 파일 크기:", os.path.getsize(tmp.name), "bytes")
            print("[DEBUG] 저장 경로:", tmp.name)
            
            wav_path = tmp.name + ".wav"
            AudioSegment.from_file(tmp.name).export(wav_path, format="wav")
            
            with open(wav_path, "rb") as audio_file:
                result = openai.Audio.transcribe("whisper-1", audio_file, language="ko")
                print("[DEBUG] Whisper 결과:", result) 
                return jsonify({"text": result["text"]})
                
    except Exception as e:
        print("[Whisper 오류]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/ai-order", methods=["POST"])
def ai_order():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        print("[GPT 요청 텍스트]:", text)

        if not text:
            return jsonify({"error": "GPT 입력 없음"}), 400

        orders = gpt_parse_order(text)
        print("[GPT 파싱 결과]", orders)

        return jsonify(orders)
    except Exception as e:
        print("[GPT 오류 발생]", e)
        return jsonify({"error": str(e)}), 500

@app.route("/order", methods=["POST"])
def order():
    items = request.get_json()
    total = sum(item["price"] * item["quantity"] for item in items if "price" in item and "quantity" in item)
    print("[주문완료]", items, f"총 {total}원")
    tts_order(items, total)
    return jsonify({"msg": f"총 {total:,}원 결제 완료. 감사합니다!", "total": total})

# ================== GPT + TTS 보조 함수 ==================

def find_closest_menu(name):
    matches = get_close_matches(name, MENU_NAMES, n=1, cutoff=0.6)
    return matches[0] if matches else None

def gpt_parse_order(user_text):
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
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    txt = rsp['choices'][0]['message']['content'].strip()
    try:
        first_bracket = txt.find("[")
        last_bracket = txt.rfind("]")
        json_part = txt[first_bracket : last_bracket+1]
        arr = json.loads(json_part)
        for item in arr:
            item["price"] = PRICE.get(item["name"], 0)
        return arr
    except Exception as e:
        print("[GPT 파싱 오류]", txt, e)
        return []

def tts_order(order_items, total):
    try:
        msg = "주문 내역은 "
        for item in order_items:
            msg += f"{item['name']} {item['quantity']}개, "
        msg += f"총 금액은 {total:,}원 입니다. 감사합니다."
        tts = gTTS(msg, lang='ko')
        tts.save("voice.mp3")
    except Exception as e:
        print("[gTTS 오류]", e)

# ================== 감지 루프 (별도 스레드) ==================

def send_display_command(screen_name, display_on):
    try:
        # 기존 상태 불러오기
        current = {}
        try:
            with open("screen_status.json") as f:
                current = json.load(f)
        except:
            pass

        # 상태가 동일하면 무시
        if current.get("screen") == screen_name and current.get("display") == display_on:
            return

        # 상태가 변경되었을 경우에만 요청
        requests.post("http://127.0.0.1:5000/display-control", json={
            "screen": screen_name,
            "on": display_on
        })
    
    except Exception as e:
        print("[화면 제어 실패]", e)

def detection_loop():
    detector = ObjectDetector("/home/KioMind/kiomind/yolo11n_rknn_model/yolo11n-rk3588.rknn")
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

            print(f"감지: {object_detected}, 거리: {person_nearby}")

            if object_detected and person_nearby:
                send_display_command("type-screen", True)
                time.sleep(1)
                continue

            # 재확인 루프
            print("조건 불만족 → 재확인")
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
                print("조건 불만족 10초 경과 → 화면 OFF")
                send_display_command("start-screen", False)
            
    finally:
        cap.release()
        detector.release()
        cv2.destroyAllWindows()

# ================== 실행 ==================

if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
