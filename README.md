# TVM Test

## 1. DarkNet

### 1.1. Overview
+ Target: use TVM-DarkNet to detect videos.

### 1.2. Scripts
+ `darknet_tune`: Auto-tune DarkNet models.
+ `darknet_evaluate`: Evaluate DarkNet inference time.
+ `darknet_show`: Show videos with the tuned model with `cv2`.
+ `darknet_pil_evaluate`: Evaluate DarkNet inference time, use `imageio` & `PIL` instead of `cv2`.
+ [Related Issue](https://discuss.tvm.ai/t/use-tvm-darknet-to-detect-vidoes-after-relay-build-module-build-cv2-ops-cost-much-more-time/4730)
  + `cv2_only_evaluate.py`: test cv2 resize and read frame inference time.
  + `darknet_comment_test.py` & `darknet_pil_comment_test.py`: test cv2/imageio read video frame & cv2/PIL resize inference time before/after `relay.build_module.build`.