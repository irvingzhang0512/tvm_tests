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

# show
coco_path = "/hdd02/zhangyiyang/darknet/data/coco.names"
font_path = "/hdd02/zhangyiyang/darknet/data/arial.ttf"
with open(coco_path) as f:
    content = f.readlines()
names = [x.strip() for x in content]


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def darwBbox(dets, img, thresh, names):
    img2 = img * 255
    img2 = img2.astype(np.uint8)
    for det in dets:
        cat = np.argmax(det['prob'])
        if det['prob'][cat] < thresh:
            continue

        imh, imw, _ = img2.shape
        b = det['bbox']
        left = int((b.x-b.w/2.)*imw)
        right = int((b.x+b.w/2.)*imw)
        top = int((b.y-b.h/2.)*imh)
        bot = int((b.y+b.h/2.)*imh)
        pt1 = (left, top)
        pt2 = (right, bot)
        text = names[cat] + " [" + str(round(det['prob'][cat] * 100, 2)) + "]"
        cv2.rectangle(img2, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img2,
                    text,
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img2


def show():
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
    cv2.namedWindow('DarkNet', cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError('Couldn\'t open capture')
    res, frame = cap.read()
    steps = 100
    cnt = 0
    start = time.time()

    while res:
        # image preprocessing
        show_image = cv2.resize(
            frame, (416, 416), interpolation=cv2.INTER_LINEAR)
        img = np.divide(show_image, 255.0)
        img = img.transpose((2, 0, 1))
        img = np.flip(img, 0)
        m.set_input('data', tvm.nd.array(img.astype(dtype)))
        m.run()
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
        dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
            (416, 416), (416, 416), thresh, 1, tvm_out)
        last_layer = net.layers[net.n - 1]
        tvm.relay.testing.yolo_detection.do_nms_sort(
            dets, last_layer.classes, nms_thresh)

        tic = time.time()
        # tvm.relay.testing.yolo_detection.draw_detections(
        #     font_path, img, dets, thresh, names, last_layer.classes)
        img = img.transpose(1, 2, 0)
        img = darwBbox(dets, img, thresh, names)
        img = np.flip(img, 2)
        tac = time.time()
        cv2.imshow('DarkNet', img)
        print(tac-tic, time.time() - tac)
        res, frame = cap.read()
        cv2.waitKey(1)
        cnt += 1
        if cnt % steps == 0:
            end = time.time()
            print(steps * 1./(end-start))
            start = end
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    if LOG_FILE is None:
        show()
    else:
        with autotvm.apply_history_best(LOG_FILE):
            show()
