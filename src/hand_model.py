import csv
import numpy as np
from tensorflow_core.lite.python.interpreter import Interpreter
from tflitemodel import TFLiteModel

class Palm():
    def __init__(self, palm_model, anchors_path):
        self.image_height = 256
        self.image_width = 256
        self.interp_palm = Interpreter(palm_model)
        self.interp_palm.allocate_tensors()

        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        # reading tflite model paramteres
        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()

        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']

        # 90° rotation matrix used to create the alignment trianlge
        self.R90 = np.r_[[[0, 1], [-1, 0]]]

        # trianlge target coordinates used to move the detected hand
        # into the right position
        self._target_triangle = np.float32([
            [128, 128],
            [128, 0],
            [0, 128]
        ])
        self._target_box = np.float32([
            [0, 0, 1],
            [256, 0, 1],
            [256, 256, 1],
            [0, 256, 1],
        ])

    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x))

    def detect(self, img_norm):
        # predict hand location and 7 initial landmarks
        self.interp_palm.set_tensor(self.in_idx, img_norm[None])
        self.interp_palm.invoke()

        out_reg = self.interp_palm.get_tensor(self.out_reg_idx)[0]
        out_clf = self.interp_palm.get_tensor(self.out_clf_idx)[0, :, 0]

        detection_mask = self._sigm(out_clf) > 0.7
        candidate_detect = out_reg[detection_mask]
        candidate_anchors = self.anchors[detection_mask]

        if candidate_detect.shape[0] == 0:
            return None, False

        # picking the widest suggestion while NMS is not implemented
        max_idx = np.argmax(candidate_detect[:, 3])
        center_wo_offst = candidate_anchors[max_idx, :2] * 256

        # 7 initial keypoints
        keypoints = center_wo_offst + candidate_detect[max_idx, 4:].reshape(-1, 2)
        return keypoints, True

class HandLandMarks(TFLiteModel):
    def __init__(self, model_path):
        self.load_model(model_path)

    def parse(self, image):
        interpretation = self.get_model_output(image)
        landmarks, handflag = interpretation
        keypoints = landmarks.reshape((-1, 2))
        return keypoints, handflag

if __name__ == "__main__":
    import argparse, cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, required=True)
    parser.add_argument("--anchors", default=None, required=True)
    parser.add_argument("--image", default=None, required=True)

    args = parser.parse_args()
    tracker = Palm(palm_model=args.model, anchors_path=args.anchors)
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32)
    image = 2 * ((image / 255.0) - 0.5)
    image = cv2.resize(image, (tracker.image_height, tracker.image_width))
    keypoints = tracker.detect(image)
    print (keypoints)
