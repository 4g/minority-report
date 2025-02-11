import cv2
import threading
import random
from queue import LifoQueue
from time import sleep

class Camera(threading.Thread):
    def __init__(self, path=0, fps=30, lifosize=1000):
        threading.Thread.__init__(self)
        self.capture = cv2.VideoCapture(path)
        self.capture.set(3, 640)
        self.capture.set(4, 480)
        self.frame_count = 0
        self.q = LifoQueue(lifosize)
        self.fps = fps

    def run(self):
        while self.capture.isOpened():
            retval, frame = self.capture.read()
            if not retval:
                return

            self.frame_count += 1
            self.q.put((frame, self.frame_count))
            sleep(1.0/self.fps)

    def get(self):
        o = self.q.get()
        self.q.queue.clear()
        return o


    def stop(self):
        self.capture.release()

def main(lifosize):
    q = Camera(lifosize=lifosize)
    q.start()

    from tqdm import tqdm
    for i in tqdm(range(100)):
        image, frame_count = q.get()
        delay = random.randint(50, 100)
        # print (f"delay {delay} frame {frame_count}")
        processed_image = dummy(image, delay)
        cv2.imshow("window", processed_image)

    q.stop()

def dummy(frame, t):
    cv2.waitKey(t)
    frame = cv2.flip(frame, 1)
    return frame

if __name__ == "__main__":
    import sys
    lifosize = sys.argv[1]
    main(int(lifosize))
