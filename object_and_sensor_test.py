from ultrasonic_sensor import measure_distance
from object_detection import ObjectDetector
import cv2
import time

# Hardware integration test: runs both object detection and ultrasonic sensor together.
# Use this script to verify that the camera, RKNN model, and HC-SR04 sensor
# are all working correctly before launching the full application.

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

        print(f"Detected: {object_detected}, Nearby: {person_nearby}")

        if object_detected and person_nearby:
            print("✅ Both conditions met: running...")
            time.sleep(0.5)
            continue

        # Conditions not met — re-check for 5 seconds before giving up
        print("⚠️ Conditions not met: re-checking for 5 seconds...")
        grace_start = time.time()
        while time.time() - grace_start < 5:
            ret, img = cap.read()
            if not ret:
                break
            object_detected = detector.detect(img)
            person_nearby = measure_distance()
            if object_detected and person_nearby:
                print("✅ Re-check passed: continuing")
                break
        else:
            print("❌ Conditions still not met: exiting")
            break

finally:
    cap.release()
    detector.release()
    cv2.destroyAllWindows()
