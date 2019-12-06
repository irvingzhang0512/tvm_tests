# V100 Example

## 1. Overview
+ Environments: V100
+ Target: use TVM-DarkNet to detect videos on V100.
+ Features:
  + [x] Use Official Tutorial codes to build yolov3/yolov3-tiny.
  + [x] Use Official Tutorial codes to Auto-tune the above models.
  + [x] cv2 issues.
+ For more information, please check my [notes](https://zhuanlan.zhihu.com/p/91876198)

## 2. Scripts
+ `darknet_tune`: Auto-tune DarkNet models.
+ `darknet_evaluate`: Evaluate DarkNet inference time.
+ `darknet_show`: Show videos with the tuned mode `cv2`.
+ `darknet_pil_evaluate`: Evaluate DarkNet inference time, use `imageio` & `PIL` instead of `cv2`.
+ [Related Issue](https://discuss.tvm.ai/t/use-tvm-darknet-to-detect-vidoes-after-relay-build-module-build-cv2-ops-cost-much-more-time/4730)
  + `cv2_only_evaluate.py`: test cv2 resize and read frame inference time.
  + `darknet_comment_test.py` & `darknet_pil_comment_test.py`: test cv2/imageio read video frame & cv2/PIL resize inference time before/after `relay.build_module.build`.