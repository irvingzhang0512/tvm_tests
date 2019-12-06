# Jetbot(Jetson Nano) Example

## 1. Overview
+ Environments: Server and Jetbot(Jetson Nano).
+ Target: use TVM-DarkNet to detect videos on Jetbot.
+ Features:
  + [x] Server rpc auto-tuning.
  + [x] Server cross compilation
  + [x] Server libs/graphs/params exports.
  + [x] Jetbot videos detection. 
+ For more information, please check my [notes]()


## 2. Scripts
+ `server_rpc_tune.py`: use Server CPU and RPC to auto-tune on Jetbot.
+ `server_cross_compile.py`: cross compile on Server.
+ `server_exports_compiled_models.py`: exports compiles models.0
+ `jetson_detect_video.py`: run detection on Jetbot.