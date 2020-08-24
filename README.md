# TVM Examples

## 1. TVM-DarkNet Example
+ Environments: V100
+ Target: use TVM-DarkNet to detect videos on V100.
+ Features:
  + [x] Use Official Tutorial codes to build yolov3/yolov3-tiny.
  + [x] Use Official Tutorial codes to Auto-tune the above models.
  + [x] cv2 issues.
+ [Notes](https://zhuanlan.zhihu.com/p/91876198)

## 2. Jetbot(Jetson Nano) Example
+ Environments: Server and Jetbot(Jetson Nano).
+ Target: use TVM-DarkNet to detect videos on Jetbot.
+ Features:
  + [x] Server rpc auto-tuning.
  + [x] Server cross compile.
  + [x] Jetbot detect videos.
+ [Notes](https://zhuanlan.zhihu.com/p/95742125)

## 3. DCN
+ Target: convert Torchvision DCN to TVM deformable_conv2d.