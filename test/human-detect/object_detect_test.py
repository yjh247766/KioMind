from rknnlite.api import RKNNLite
import numpy as np
import cv2
import time

classNames = ["person"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

RKNN_MODEL_PATH = "/home/KioMind/kiomind/yolo11n_rknn_model/yolo11n-rk3588.rknn"

rknn = RKNNLite()
print("-> Loading RKNN model...")
if rknn.load_rknn(RKNN_MODEL_PATH) != 0:
    print("Model load failed!")
    exit()

print("-> Initializing runtime...")
if rknn.init_runtime() != 0:
    print("Runtime init failed!")
    exit()

def detect_person_and_draw(img, conf_threshold=0.4, nms_threshold=0.4):
    input_img = cv2.resize(img, (640, 640))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.uint8)
    input_img = np.expand_dims(input_img, axis=0)

    outputs = rknn.inference(inputs=[input_img])
    output = outputs[0][0]
    output = np.transpose(output)

    boxes = []
    confidences = []
    class_ids = []

    for i in range(output.shape[0]):
        row = output[i]
        class_probs = row[4:]
        conf = np.max(class_probs)
        cls_id = np.argmax(class_probs)

        if conf > conf_threshold and cls_id < len(classNames):
            x, y, w, h = row[0], row[1], row[2], row[3]
            x1 = int((x - w / 2) * img.shape[1] / 640)
            y1 = int((y - h / 2) * img.shape[0] / 640)
            width = int(w * img.shape[1] / 640)
            height = int(h * img.shape[0] / 640)

            boxes.append([x1, y1, width, height])
            confidences.append(float(conf))
            class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detected = False
    if len(indices) > 0:
        detected = True
        for i in indices.flatten():
            x1, y1, w, h = boxes[i]
            x2, y2 = x1 + w, y1 + h
            conf = confidences[i]
            cls_id = class_ids[i]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{classNames[cls_id]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return detected


cap = cv2.VideoCapture(0)
cap.set(3, 480)  # 가로 480으로 줄임
cap.set(4, 360)  # 세로도 비율에 맞게 줄임

prev_time = time.time()

while True:
    ret, img = cap.read()
    if not ret:
        break

    detected = detect_person_and_draw(img)

    if detected:
        print("Person detected!")
    else:
        print("No person detected.")

    # FPS 계산
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("RKNN Inference", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
rknn.release()
cv2.destroyAllWindows()
