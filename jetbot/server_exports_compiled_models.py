import os
import tensorflow as tf
import tvm
import tvm.relay.testing
import tvm.relay.testing.tf as tf_testing
import tvm.relay.testing.darknet
import tvm.relay.testing.yolo_detection
from tvm import relay
from tvm.relay.testing.darknet import __darknetffi__
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
set_cuda_target_arch('sm_53')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

target = tvm.target.cuda(model="nano")
target_host = "llvm -target=aarch64-linux-gnu-g++"
dtype = 'float32'
input_shape = (1, 416, 416, 3)


def get_tf_yolov3_tiny(
        model_path=("/hdd02/zhangyiyang/Tensorflow-YOLOv3/"
                    "weights/raw-yolov3-tiny.pb"),
        outputs=['yolov3_tiny/concat_6'],):

    with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        with tf.compat.v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(
                sess, outputs[0])
        print("successfully load tf model")

    mod, params = relay.frontend.from_tensorflow(
        graph_def,
        layout="NCHW",
        shape={'Placeholder': input_shape},
        outputs=outputs,
    )
    print("successfully convert tf model to relay")
    return mod, params


def get_darknet(
        cfg_path='/hdd02/zhangyiyang/darknet/cfg/yolov3-tiny.cfg',
        weights_path='/hdd02/zhangyiyang/data/darknet/yolov3-tiny.weights',
        lib_path='/home/ubuntu/.tvm_test_data/darknet/libdarknet2.0.so',):
    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    net = DARKNET_LIB.load_network(cfg_path.encode(
        'utf-8'), weights_path.encode('utf-8'), 0)
    mod, params = relay.frontend.from_darknet(
        net, dtype='float32', shape=[1, net.c, net.h, net.w])
    print("successfully load darknet relay")
    return mod, params


def exports(source, target_path="./exports"):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if source == 'darknet':
        mod, params = get_darknet()
    elif source == 'tf':
        mod, params = get_tf_yolov3_tiny()
    else:
        raise ValueError('unknown source {}'.format(source))

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                         target=target,
                                         target_host=target_host,
                                         params=params)

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    lib_path = os.path.join(target_path,
                            'yolov3-tiny-{}-lib.tar'.format(source))
    graph_path = os.path.join(target_path,
                              'yolov3-tiny-{}-graph.json'.format(source))
    params_path = os.path.join(target_path,
                               'yolov3-tiny-{}-param.params'.format(source))
    lib.export_library(lib_path)
    print(lib_path)
    with open(graph_path, "w") as fo:
        fo.write(graph)
    with open(params_path, "wb") as fo:
        fo.write(relay.save_param_dict(params))


exports("darknet", "./exports/")
exports("tf", "./exports/")
