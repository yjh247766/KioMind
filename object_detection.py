from rknnlite.api import RKNNLite
import numpy as np
import cv2
import time

# COCO class index 0 corresponds to 'person'; only this class is used
PERSON_CLASS_ID = 0
classNames = ["person"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ObjectDetector:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError("Model load failed!")
        if self.rknn.init_runtime() != 0:
            raise RuntimeError("Runtime init failed!")

    def detect(self, img, conf_threshold=0.4, nms_threshold=0.4):
        """
        Run YOLO11n inference on a single frame and return whether a person is detected.

        Args:
            img: BGR image as a numpy array (from cv2.VideoCapture)
            conf_threshold: minimum confidence score to accept a detection
            nms_threshold: IoU threshold for Non-Maximum Suppression

        Returns:
            True if at least one person is detected, False otherwise.
        """
        # Preprocess: resize to model input size and convert color space
        input_img = cv2.resize(img, (640, 640))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.uint8)
        input_img = np.expand_dims(input_img, axis=0)

        outputs = self.rknn.inference(inputs=[input_img])
        output = outputs[0][0]
        output = np.transpose(output)  # shape: (num_anchors, 4 + num_classes)

        boxes = []
        confidences = []
        class_ids = []

        for i in range(output.shape[0]):
            row = output[i]
            class_probs = row[4:]
            conf = np.max(class_probs)
            cls_id = np.argmax(class_probs)

            # Accept only high-confidence detections of the 'person' class
            if conf > conf_threshold and cls_id == PERSON_CLASS_ID:
                x, y, w, h = row[0], row[1], row[2], row[3]
                # Convert center-format bbox to top-left corner format
                x1 = int((x - w / 2) * img.shape[1] / 640)
                y1 = int((y - h / 2) * img.shape[0] / 640)
                width = int(w * img.shape[1] / 640)
                height = int(h * img.shape[0] / 640)

                boxes.append([x1, y1, width, height])
                confidences.append(float(conf))
                class_ids.append(cls_id)

        # Apply NMS to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Return True if at least one person remains after NMS
        return len(indices) > 0

    def release(self):
        self.rknn.release()
