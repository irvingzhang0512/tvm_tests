import numpy as np
import cv2
import time

from tvm import autotvm
from tvm import relay

from tvm.relay.testing.darknet import __darknetffi__

# constants
MODEL_NAME = 'yolov3-tiny'
LOG_FILE = "%s.log" % MODEL_NAME
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
VIDEO_PATH = '/hdd02/zhangyiyang/data/videos/Office-Parkour.mp4'

# PARAMS
batch_size = 1
steps = 100
target = 'cuda'
# target = 'cuda -libs=cudnn'
dtype = 'float32'

# darknet files
cfg_path = '/hdd02/zhangyiyang/darknet/cfg/{}'.format(CFG_NAME)
weights_path = '/hdd02/zhangyiyang/data/darknet/{}'.format(WEIGHTS_NAME)
lib_path = '/home/ubuntu/.tvm_test_data/darknet/libdarknet2.0.so'
DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode(
    'utf-8'), weights_path.encode('utf-8'), 0)


def evaluate():
    mod, params = relay.frontend.from_darknet(
        net, dtype=dtype, shape=[batch_size, net.c, net.h, net.w])

#    # if commented, then cv2.resize & cv2.read cost much less time
#     with relay.build_config(opt_level=3):
#         graph, lib, params = relay.build_module.build(
#             mod, target=target, params=params)

    # run
    cnt = 0
    cap = cv2.VideoCapture(VIDEO_PATH)
    res, frame = cap.read()
    start = time.time()

    while res:
        t1 = time.time()
        img = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
        img = np.divide(img, 255.0)
        img = img.transpose((2, 0, 1))
        img = np.flip(img, 0)
        t2 = time.time()

        res, frame = cap.read()
        cnt += 1
        t3 = time.time()
        print('%05f %05f %05f' % (t2 - t1, t3-t2, t3-t1))

        if cnt == steps:
            break
    end = time.time()
    print(steps*1./(end-start))
    cap.release()


if __name__ == '__main__':
    if LOG_FILE is None:
        evaluate()
    else:
        with autotvm.apply_history_best(LOG_FILE):
            evaluate()
