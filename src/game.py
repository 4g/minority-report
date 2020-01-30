from camera import Camera
from tqdm import tqdm
from pose_engine import PoseEngine
import cv2
from collections import deque
import pyautogui

GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)


def draw(image, poses, threshold=0.2):
    num_elems = len(poses)
    for index, pose in enumerate(poses):
        for k in pose.keypoints:
            keypoint = pose.keypoints[k]
            y, x = keypoint.yx
            score = keypoint.score
            if score > threshold:
                cv2.circle(image, (x, y), 1 + int(index/(num_elems/3)), (0, 255-index, index), -1)

    return image


def main(video):

    engine = PoseEngine("../models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite")
    cam = Camera(path=video, fps=30)
    cam.start()

    controller_kp = 'right wrist'
    point_q = deque()

    for i in tqdm(range(1000)):
        img, count = cam.get()
        poses, time = engine.DetectPosesInImage(img)
        if poses:
            pose = poses[0]
            control_kp = pose.keypoints[controller_kp]
            score = control_kp.score
            y, x = control_kp.yx
            # pyautogui.moveTo(x, y, duration = 0)

            point_q.append(pose)
            if len(point_q) > 30:
                point_q.popleft()

        img = draw(img, point_q)

        cv2.imshow("apple", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, required=True)

    args = parser.parse_args()
    path = args.path

    if str.isdigit(args.path):
        path = int(args.path)

    main(path)