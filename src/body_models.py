import numpy as np
import cv2
from hand_model import Palm, HandLandMarks

palm_cpu = "../models/palm_detection_without_custom_op.tflite"
palm_anchors = "../models/anchors.csv"

GREEN = (0,255,0)
RED = (0,0,255)

def crop(image, width, height):
    """
    Crops an image to desired width / height ratio
    :param image: image to crop
    :param width: desired width
    :param height: desired height
    :return: returns an image cropped to width/height ratio
    """
    desired_ratio = width / height
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_ratio = image_width / image_height
    new_width, new_height = image_width, image_height

    # if original image is wider than desired image, crop across width
    if image_ratio > desired_ratio:
        new_width = int(image_height * desired_ratio)

    # crop across height otherwise
    elif image_ratio < desired_ratio:
        new_height = int(image_width/desired_ratio)

    image = image[image_height // 2 - new_height // 2 : image_height // 2 + new_height // 2 ,
                    image_width // 2 - new_width // 2: image_width // 2 + new_width // 2]

    return image

class PalmLib:
    def __init__(self):
        self.palm_detector = Palm(palm_cpu, palm_anchors)
        self.width = self.palm_detector.image_width
        self.height = self.palm_detector.image_height

    def detect(self, image):
        image = np.asarray(image, dtype=np.float32)
        image = cv2.resize(image, (self.height, self.width))
        image = 2 * ((image / 255.0) - 0.5)
        keypoints = self.palm_detector.detect(image)
        return keypoints

    def draw(self, image, keypoints):
        if keypoints is not None:
            for point in keypoints:
                x, y = point
                x, y = int(x), int(y)
                cv2.circle(image, (x, y), 3, GREEN, -1)
        return image

    def rescale_keypoints(self, keypoints, shape):
        scale_x = shape[1] / self.width
        scale_y = shape[0] / self.height
        keypoints = [(int(i[0] * scale_x), int(i[1] * scale_y)) for i in keypoints]
        return keypoints

    def demo(self, cam):
        cam.start()
        for i in tqdm(range(10000)):
            frame, count = cam.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            frame = crop(frame, self.width, self.height)
            keypoints, handflag = self.detect(frame)
            if handflag:
                keypoints = self.rescale_keypoints(keypoints, frame.shape)
                self.draw(frame, keypoints)

            cv2.imshow("test", frame)
            cv2.waitKey(10)



if __name__ == "__main__":
    from camera import Camera
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, required=True)
    parser.add_argument("--name", default=None, required=True)

    args = parser.parse_args()
    path = args.path
    
    if str.isdigit(args.path):
        path = int(args.path)

    cam = Camera(path, 30)
    if "palm" in args.name or "hand" in args.name:
        palm = PalmLib()
        palm.demo(cam)