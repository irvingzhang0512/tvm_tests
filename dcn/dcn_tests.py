import os

import numpy as np
import torch
import torchvision
from mmcv.ops import DeformConv2d as mmcv_dcn
from torchvision.ops import DeformConv2d as dcn

import tvm
from tvm import relay
from tvm.relay import testing
from tvm.contrib import graph_runtime

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def _output(v1, v2):
    return np.max(np.abs(v1 - v2))


if __name__ == "__main__":
    dtype = "float32"
    target = "llvm"
    target_host = "llvm"
    cur_data = torch.randn([1, 3, 224, 224]).cuda()
    cur_data_np = cur_data.cpu().detach().numpy()
    cur_offset = torch.randn([1, 18, 224, 224]).cuda()
    cur_offset_np = cur_offset.cpu().detach().numpy()
    cur_weight = torch.randn([64, 3, 3, 3]).cuda()
    cur_weight_np = cur_weight.cpu().detach().numpy()

    # pytorch torchvision
    pytorch_model = dcn(3, 64, 3, padding=1, bias=False).cuda().eval()
    pytorch_model.weight = torch.nn.Parameter(cur_weight)
    scripted_model = torch.jit.trace(
        pytorch_model, [cur_data, cur_offset]
    ).eval()
    pytorch_res = pytorch_model(cur_data, cur_offset)

    # mmcv dcn
    mmcv_model = mmcv_dcn(3, 64, 3, padding=1).cuda().eval()
    mmcv_model.weight = torch.nn.Parameter(cur_weight)
    with torch.no_grad():
        mmcv_res = mmcv_model(cur_data, cur_offset)

    # # tvm from pytorch
    # shape_list = [("input0", [1, 3, 224, 224]), ("input1", [1, 18, 224, 224])]
    # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    # ctx = tvm.gpu(0)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(
    #         mod, target=target, target_host=target_host, params=params
    #     )
    # m = graph_runtime.GraphModule(lib["default"](ctx))
    # m.set_input(0, tvm.nd.array(cur_data_np, ctx=ctx))
    # m.set_input(1, tvm.nd.array(cur_offset_np, ctx=ctx))
    # m.run()
    # tvm_output = m.get_output(0)

    # tvm relay
    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
    weight = relay.var("weight", relay.TensorType((64, 3, 3, 3), "float32"))
    offset = relay.var("bias", relay.TensorType((1, 18, 224, 224), "float32"))
    simple_net = relay.nn.deformable_conv2d(
        data,
        offset,
        weight,
        strides=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        deformable_groups=1,
        groups=1,
        channels=64,
        kernel_size=(3, 3),
    )
    simple_net = relay.Function(
        relay.analysis.free_vars(simple_net), simple_net
    )
    net, params2 = testing.create_workload(simple_net)
    ctx2 = tvm.context(target, 0)
    params2["weight"] = tvm.nd.array(cur_weight_np.astype(dtype))
    params2["bias"] = tvm.nd.array(cur_offset_np.astype(dtype))
    lib2 = relay.build_module.build(net, target, params=params2)
    module = graph_runtime.GraphModule(lib2["default"](ctx2))
    module.set_input('data', tvm.nd.array(cur_data_np, ctx=ctx2))
    # module.set_input('bias', tvm.nd.array(cur_offset_np, ctx=ctx2))
    # module.set_input('weight', tvm.nd.array(cur_weight_np, ctx=ctx2))
    module.run()
    tvm_relay_output = module.get_output(0)


    print(
        "pytorch torchvision res & tvm relay res",
        _output(tvm_relay_output.asnumpy(), pytorch_res.cpu().detach().numpy()),
    )
    # print(
    #     "pytorch torchvision weight & tvm params weight",
    #     _output(params["weight"].asnumpy(), cur_weight_np),
    # )
    print(
        "mmcv dcn vs torchvision dcn",
        _output(
            mmcv_res.cpu().detach().numpy(), pytorch_res.cpu().detach().numpy()
        ),
    )
    # print(
    #     "tvm relay outputs vs tvm from pytorch outputs",
    #     _output(tvm_relay_output.asnumpy(), tvm_output.asnumpy()),
    # )
