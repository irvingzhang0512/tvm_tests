import numpy as np
import cv2
import time

VIDEO_PATH = '/hdd02/zhangyiyang/data/videos/Office-Parkour.mp4'


def evaluate():

    # run
    cnt = 0
    cap = cv2.VideoCapture(VIDEO_PATH)
    res, frame = cap.read()
    steps = 100
    start = time.time()

    while res:
        t1 = time.time()

        # image preprocessing
        img = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
        img = np.divide(img, 255.0)
        img = img.transpose((2, 0, 1))
        img = np.flip(img, 0)
        t2 = time.time()

        res, frame = cap.read()
        cnt += 1
        t3 = time.time()
        print('%05f %05f %05f' % (t2-t1, t3-t2, t3-t1))

        if cnt == steps:
            break
    end = time.time()
    print(steps*1./(end-start))


if __name__ == '__main__':
    evaluate()
