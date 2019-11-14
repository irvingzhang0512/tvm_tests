from PIL import Image
import imageio
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

    # # if commented, then read video frame & image resize cost less time
    # with relay.build_config(opt_level=3):
    #     graph, lib, params = relay.build_module.build(
    #         mod, target=target, params=params)

    # run
    cnt = 0
    cap = imageio.get_reader(VIDEO_PATH)
    start = time.time()
    itr = iter(cap)

    while True:
        t1 = time.time()
        frame = next(itr)
        t2 = time.time()
        img = Image.fromarray(frame)
        img = img.resize((416, 416))
        t3 = time.time()

        cnt += 1
        print('%05f %05f %05f' % (t2 - t1, t3-t2, t3-t1))

        t3 = time.time()
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
