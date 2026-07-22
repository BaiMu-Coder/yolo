# yolo
航空基金项目：RK3588板端部署yolov8seg模型代码

## Python/PT 效果优先版本

`unit_test/final_version.py` 直接加载转换前的实例分割 `.pt` 模型，保留 mask、RANSAC
椭圆拟合和 LM 位姿解算，并保存标注视频、逐帧图片、mask、位姿 TXT、检测 JSONL 和位姿曲线。

项目已创建 `.venv-yolo` 环境。把 PT 模型放到项目目录（默认文件名 `best.pt`），然后运行：

```bash
.venv-yolo/bin/python unit_test/final_version.py \
  --model best.pt --video input.mp4 --output output_python --no-show
```

摄像头或图片目录输入：

```bash
.venv-yolo/bin/python unit_test/final_version.py --model best.pt --cam 0
.venv-yolo/bin/python unit_test/final_version.py --model best.pt --images images --no-show
```

视频模式在脚本顶部 `Config` 中只需设置输入路径、输出路径和视频名。例如：

```python
input_path = "/data/input"
output_path = "/data/result"
video_name = "test01.mp4"
```

程序读取 `/data/input/test01.mp4`，自动输出到 `/data/result/test01/`，结果视频名为
`test01_result.mp4`。相机内参、物理尺寸、置信度、平滑与其他保存开关也集中在该配置区。
默认会创建可缩放的实时前台窗口；按 `q` 或 `ESC` 退出，按 `s` 保存当前画面。服务器无桌面环境时使用 `--no-show`。

## C++ 串行图片/视频处理

`unit_test/batch_image_video.cpp` 接入现有 RKNN、YOLOv8-seg 后处理、Mask、RANSAC
椭圆拟合及 LM 位姿解算框架，并通过现有 CMake 生成 `batch_image_video`。
C++ 的 Mask/RANSAC、质量评分和内切圆保底逻辑已统一抽离到
`ellipse_fitter/ellipse_fitter.hpp/.cpp`，在线推理池、批处理和单帧测试不再各自保留副本。

编译：

```bash
cmake -S . -B build
cmake --build build --target batch_image_video -j
cmake --install build
```

处理图片文件夹：

```bash
./install/batch_image_video \
  --model ./best.rknn \
  --input-path /data/images \
  --output-path /data/result \
  --show
```

内孔默认使用检测框中心/内切圆。如需对比内孔 Mask 拟合，追加
`--hole-mask`；验收保底时追加 `--force-reference-box`，可强制外/中参考圈使用检测框内切圆。

处理视频：

```bash
./install/batch_image_video \
  --model ./best.rknn \
  --input-path /data/videos \
  --video-name test.mp4 \
  --output-path /data/result \
  --show
```

输出内容：

- `visual/原图名.*`：图片文件夹逐图可视化结果；视频加 `--save-video-frames` 时保存逐帧 JPG。
- `labels/原图名.txt`：YOLO 格式 `class x_center y_center width height confidence`，坐标归一化。
- `ellipses/原图名.txt`：类别、椭圆中心、长轴、短轴、角度、置信度、拟合来源及误差。
- `poses/原图名.txt`：参考类别与中心、孔中心，以及自动/固定距离两套 yaw、pitch、roll、tx、ty、tz。
- 视频输入还会在输出根目录生成 `<视频名>_result.mp4`，编码不可用时自动回退为 AVI。

### C++ 单帧模块化示例

`unit_test/single_frame_pipeline.cpp` 把单帧流程拆成六个互相独立的小模块：
`ImageLoader`、`RknnInference`、`EllipseFitter`、`PoseSolver`、`Visualizer` 和
`ResultWriter`。模块之间只通过 `FrameContext` 传递结果，不修改项目原有接口。

```bash
cmake --build build --target single_frame_pipeline -j
./install/single_frame_pipeline \
  --model ./best.rknn \
  --image /data/input/test.jpg \
  --output /data/result \
  --show
```

单帧入口同样支持 `--hole-mask` 和 `--force-reference-box`。

它会为这一帧保存可视化图片、YOLO 检测 TXT、椭圆 TXT，以及自动距离和固定距离两套位姿 TXT。

## 圆环拟合现场切换

`npu_infer_pool` 默认对参考外/中圈（cls0/cls1）优先使用 Mask 拟合，内孔（cls2）使用检测框中心和内切圆。验收现场图像质量不足时，可在提交推理任务前开启参考圈保底模式：

默认外圈/中圈只采用两级策略：Mask-RANSAC 综合内点率、平均边界误差、圆心偏差和轴比得到质量分，低于门槛时直接改用检测框内切圆，不执行灰度边缘拟合。两圈同时存在时再检查同心度、物理直径比、轴比和方向一致性。视频/摄像头模式最后使用随质量自适应的椭圆 EMA，并拒绝低质量突变。可视化中绿色为 Mask、黄色为检测框内切圆。

```cpp
npu_infer_pool pool(model_path);

// false（默认）：外/中圈优先 Mask 椭圆；
// true：外/中圈都强制使用检测框内切圆。
pool.set_reference_ring_force_box_mode(true);

// 内孔默认已是 false；如需试验 Mask 拟合可显式打开。
pool.set_class2_mask_fit_mode(false);

// 默认开启视频时序滤波；单张图片测试可关闭。
pool.set_temporal_filter_enabled(true);
```

也可通过 `set_class0_mask_fit_mode()` 和 `set_class1_mask_fit_mode()` 分别控制两个参考圈。
使用 `unit_test/final_version.cpp` 的命令行版本时，可直接追加 `--force-reference-box` 开启该保底模式，或追加 `--hole-mask` 对比内孔 Mask 拟合。
