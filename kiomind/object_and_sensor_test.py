from ultrasonic_sensor import measure_distance
from object_detection import ObjectDetector
import cv2
import time

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

        print(f"감지: {object_detected}, 근접: {person_nearby}")

        if object_detected and person_nearby:
            print("✅ 조건 만족: 실행 중...")
            time.sleep(0.5)
            continue

        # 조건 불만족 → 5초 동안 재확인 루프
        print("⚠️ 조건 불만족: 5초 동안 재확인 중...")
        grace_start = time.time()
        while time.time() - grace_start < 5:
            ret, img = cap.read()
            if not ret:
                break
            object_detected = detector.detect(img)
            person_nearby = measure_distance()
            if object_detected and person_nearby:
                print("✅ 재검사 통과: 계속 실행")
                break
        else:
            print("❌ 조건 계속 불만족: 프로그램 종료")
            break

finally:
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
