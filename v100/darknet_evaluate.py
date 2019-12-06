import numpy as np
import cv2
import time

import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
import tvm.contrib.graph_runtime as runtime

from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet

# constants
MODEL_NAME = 'yolov3-tiny'
LOG_FILE = "%s.log" % MODEL_NAME
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
VIDEO_PATH = '/hdd02/zhangyiyang/data/videos/Office-Parkour.mp4'

# PARAMS
batch_size = 1
thresh = 0.5
nms_thresh = 0.45
steps = 100
# target='llvm'
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
    # darknet relay
    mod, params = relay.frontend.from_darknet(
        net, dtype=dtype, shape=[batch_size, net.c, net.h, net.w])

    # compile darknet
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            mod, target=target, params=params)
    ctx = tvm.context(str(target), 0)
    m = runtime.create(graph, lib, ctx)
    m.set_input(**params)

    # run
    cnt = 0
    cap = cv2.VideoCapture(VIDEO_PATH)
    res, frame = cap.read()
    start = time.time()

    while res:
        t1 = time.time()

        # image preprocessing
        img = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
        img = np.divide(img, 255.0)
        img = img.transpose((2, 0, 1))
        img = np.flip(img, 0)
        t2 = time.time()

        m.set_input('data', tvm.nd.array(img.astype(dtype)))
        t3 = time.time()

        m.run()
#         ctx.sync()
        t4 = time.time()

        # get outputs
        tvm_out = []
        if MODEL_NAME == 'yolov2':
            layer_out = {}
            layer_out['type'] = 'Region'
            layer_attr = m.get_output(2).asnumpy()
            layer_out['biases'] = m.get_output(1).asnumpy()
            out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                         layer_attr[2], layer_attr[3])
            layer_out['output'] = m.get_output(0).asnumpy().reshape(out_shape)
            layer_out['classes'] = layer_attr[4]
            layer_out['coords'] = layer_attr[5]
            layer_out['background'] = layer_attr[6]
            tvm_out.append(layer_out)

        elif MODEL_NAME == 'yolov3':
            for i in range(3):
                layer_out = {}
                layer_out['type'] = 'Yolo'
                layer_attr = m.get_output(i*4+3).asnumpy()
                layer_out['biases'] = m.get_output(i*4+2).asnumpy()
                layer_out['mask'] = m.get_output(i*4+1).asnumpy()
                out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                             layer_attr[2], layer_attr[3])
                layer_out['output'] = m.get_output(
                    i*4).asnumpy().reshape(out_shape)
                layer_out['classes'] = layer_attr[4]
                tvm_out.append(layer_out)

        elif MODEL_NAME == 'yolov3-tiny':
            for i in range(2):
                layer_out = {}
                layer_out['type'] = 'Yolo'
                layer_attr = m.get_output(i*4+3).asnumpy()
                layer_out['biases'] = m.get_output(i*4+2).asnumpy()
                layer_out['mask'] = m.get_output(i*4+1).asnumpy()
                out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                             layer_attr[2], layer_attr[3])
                layer_out['output'] = m.get_output(
                    i*4).asnumpy().reshape(out_shape)
                layer_out['classes'] = layer_attr[4]
                tvm_out.append(layer_out)
                thresh = 0.560
        t5 = time.time()

        dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
            (416, 416), (416, 416), thresh, 1, tvm_out)
        last_layer = net.layers[net.n - 1]
        tvm.relay.testing.yolo_detection.do_nms_sort(
            dets, last_layer.classes, nms_thresh)
        t6 = time.time()

        res, frame = cap.read()
        cnt += 1
        t7 = time.time()
        print('%05f %05f %05f %05f %05f %05f %05f' %
              (t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t7-t1))

        if cnt % steps == 0:
            end = time.time()
            print(steps/(end-start))
            start = end
    cap.release()


if __name__ == '__main__':
    if LOG_FILE is None:
        evaluate()
    else:
        with autotvm.apply_history_best(LOG_FILE):
            evaluate()
