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
