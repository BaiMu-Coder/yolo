# yolo
航空基金项目：RK3588板端部署yolov8seg模型代码

## Python/PT 效果优先版本

`unit_test/final_version.py` 直接加载转换前的实例分割 `.pt` 模型，保留软 mask、
亚像素椭圆拟合、不确定度评估和双圆环联合 LM 位姿解算，并保存标注视频、逐帧图片、
mask、位姿 TXT、检测 JSONL 和位姿曲线。
服务器版本默认采用 FP32；按配置尺寸执行固定画布 letterbox、Mask 亚像素梯度、RANSAC、
椭圆 LM 和位姿 LM 参数均与板端保持一致。

甲方演示机的验证基线是 `ultralytics==8.0.175`，代码同时保留 Ultralytics 8.x
兼容性；效果优先入口通过运行时能力检测直接保留阈值化前的软 Mask。把 PT 模型
放到项目目录（默认文件名 `best.pt`），然后运行：

```bash
.venv-yolo/bin/python unit_test/final_version.py \
  --model best.pt --video input.mp4 --output output_python \
  --imgsz 1024 --no-show
```

PT 模型推理尺寸可在配置区修改 `image_size`，也可在命令行使用
`--imgsz 640`、`--imgsz 1024` 或 `--imgsz 768 1024`（顺序为 H W）。
`batch_mask_ring_visualization.py` 复用同一套参数。PT 模型通常支持运行时尺寸切换；
RKNN 则采用后文说明的多个静态模型文件。

### 检测框误差与外圈椭圆 IoU 评估

`unit_test/yolo_mask_bbox_error.py` 已改为配置区优先。修改文件顶部
`EvaluationConfig` 中的模型、图片、标签和输出路径后直接运行：

```bash
python unit_test/yolo_mask_bbox_error.py
```

脚本保留原有七项检测框像素误差，并统一按几何尺寸确定最外圈：标签侧取外接框面积
最大的多边形，预测侧取外接框面积最大的检测实例，不再按类别或最大 IoU 做目标
对应。七项检测误差和椭圆 IoU 共用这唯一一对目标。最大预测实例复用
`final_version.py` 的稳健 Mask 椭圆拟合，并计算最大标签多边形与填充拟合椭圆的像素 IoU。
最大标签与最大预测的类别还必须一致；类别不一致的图片标记为 `class_mismatch`，
可视化单独保存到 `class_mismatch/`，且不参与检测误差、椭圆 IoU、分组和排行统计。
结果按拟合椭圆完整长轴直径 `max(width, height)` 分为：

- 小目标：`major < 77 px`；
- 中目标：`77 px <= major <= 311 px`；
- 大目标：`major > 311 px`。

输出按正常检测、椭圆 IoU 和类别异常分为三个目录：

```text
误差评估结果/
├── class_mismatch/
├── detection/
│   ├── detection_statistics.xlsx
│   ├── grouped_errors.png
│   ├── visualizations/
│   ├── over_threshold/
│   └── top10_max_error/
└── ellipse_iou/
    ├── ellipse_iou_statistics.xlsx
    ├── overall_and_grouped.png
    ├── visualizations/
    └── lowest_50/
```

检测总体统计覆盖所有检测成功样本；小、中、大分组只统计具有有效拟合外圈长轴的
样本。`top10_max_error/` 按每张图片七项误差中的最大值统一排序，不再按七个指标
重复保存。检测工作簿包含“总体误差”“分组误差”“逐图误差”和
“总体最大误差Top10”工作表；逐图表中每张图片固定一行，失败样本也保留状态。

IoU 工作簿包含总体及三档的样本占比、平均值、最小值、最大值和 P95，以及评估状态、
逐图 IoU 和最低50排行。`overall_and_grouped.png` 用均值柱、P95 标记和最小—最大
范围直观展示统计结果。`ellipse_iou/visualizations/` 保存全部独立 IoU 可视化：
绿色边界为最大外接框对应的标签多边形，红色边界为拟合椭圆；绿色填充表示仅标签、红色表示仅
椭圆、黄色表示交集。IoU 最低的 50 张另存到 `lowest_50/`。

Excel 输出依赖 `openpyxl`。如果服务器没有安装该库，或 Excel 写入失败，脚本不会
停止，而会自动回退为同等内容的 UTF-8-BOM TXT；两张统计 PNG 仍会正常生成。
命令行参数仍可临时覆盖配置，例如 `--imgsz 1024 --device 0`。

板端可使用同步的 RKNN/C++ 评估入口；同样固定使用“最大标签外框 + 最大预测外框”，
不按类别对应。统计口径、长轴分组、Top10、最低50和目录结构与 Python 版一致：

```bash
cmake -S . -B build
cmake --build build --target yolo_mask_bbox_error -j
cmake --install build

./install/yolo_mask_bbox_error \
  --model ./best_1024.rknn \
  --images /data/evaluation/images \
  --labels /data/evaluation/labels \
  --output /data/evaluation/result \
  --threshold 3
```

C++ 版不增加板端表格依赖，直接写 Excel 2003 XML 多工作表：
`detection/detection_statistics.xml` 和
`ellipse_iou/ellipse_iou_statistics.xml`，Excel/WPS 可直接打开。检测工作簿包含
总体、分组、逐图和 Top10；IoU 工作簿包含总体/分组、状态占比、逐图和最低50。
若工作簿创建失败，程序自动回退为制表符分隔 TXT。IoU 同时生成
`ellipse_iou/overall_and_grouped.png`，显示均值、P95、最小—最大范围和分组占比。
模型输入尺寸仍由 RKNN 张量自动读取，640、1024 或相同输出拓扑的其他尺寸无需修改
该评估程序。

仓库内默认模型、视频和输出路径已经按项目根目录自动解析；在带桌面的演示机上也可
直接运行 `.venv-yolo/bin/python unit_test/final_version.py`。

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

程序读取 `/data/input/test01.mp4`，直接输出到 `/data/result/`，结果视频名为
`test01_result.mp4`。输出根目录下会创建与 C++ 批处理一致的 `visual/`、`labels/`、
`ellipses/`、`poses/`，另外保留 Python 的逐帧 JSONL、汇总位姿和曲线。
相机内参、物理尺寸、置信度、平滑与其他保存开关也集中在该配置区。
默认会创建可缩放的实时前台窗口；按 `q` 或 `ESC` 退出，按 `s` 保存当前画面。服务器无桌面环境时使用 `--no-show`。

服务器上也可以完全沿用 `batch_image_video.cpp` 的参数风格：

```bash
python unit_test/final_version.py \
  --model best.pt \
  --input-path /data/images \
  --mode images \
  --output-path /data/result \
  --device 0 \
  --no-show

python unit_test/final_version.py \
  --model best.pt \
  --input-path /data/videos \
  --video-name test.mp4 \
  --mode video \
  --output-path /data/result \
  --save-video-frames \
  --no-show
```

## C++ 串行图片/视频处理

`unit_test/batch_image_video.cpp` 接入现有 RKNN、YOLOv8-seg 后处理、Mask、稳健椭圆拟合
及联合 LM 位姿解算框架，并通过现有 CMake 生成 `batch_image_video`。
C++ 的软 Mask/亚像素轮廓、PROSAC/LO-RANSAC、Sampson LM、质量评分和内切圆保底逻辑已统一抽离到
`ellipse_fitter/ellipse_fitter.hpp/.cpp`，在线推理池、批处理和单帧测试不再各自保留副本。

编译：

```bash
cmake -S . -B build
cmake --build build --target batch_image_video -j
cmake --install build
```

### RKNN 640/1024 模型通用切换

板端采用两个独立的静态 RKNN 模型，例如 `best_640.rknn` 和
`best_1024.rknn`。程序启动时从 `--model` 指定的文件读取输入张量、三个检测网格和
Proto 张量尺寸，预处理与后处理不再维护另一份手写的输入尺寸配置，也不需要额外传
`--input-size`：

```bash
# 640 模型
./install/batch_image_video --model ./best_640.rknn \
  --input-path /data/images --output-path /data/result_640

# 1024 模型：仅替换模型路径
./install/batch_image_video --model ./best_1024.rknn \
  --input-path /data/images --output-path /data/result_1024
```

启动日志会打印实际解析结果。典型 640 模型应显示输入 `640x640`、检测网格
`80x80 / 40x40 / 20x20`、Proto `160x160`；典型 1024 模型通常对应
`128x128 / 64x64 / 32x32` 和 Proto `256x256`。程序以模型内真实张量为准，不依赖
这些典型数值，也支持相同输出拓扑下的其他静态宽高。

模型兼容条件是 3 个检测分支，每个分支均为
`[box, class, class_sum, mask_coeff]`，最后一个输出为 Proto；量化模型的原始输出须为
affine INT8。模型不符合条件时会在初始化阶段明确报错，避免按错误尺寸继续推理。

相机采集分辨率与网络输入分辨率是两件事：只把模型由 640 换成 1024 时，相机标定
内参不变；若同时改变了相机实际输出分辨率，则应使用对应分辨率重新标定，或按图像
缩放关系同步调整 `fx/fy/cx/cy`。1024 相比 640 的特征图和 Proto 像素数约为
2.56 倍，通常能改善小目标和边界细节，但 NPU、Mask 解码和内存带宽开销也会增加，
是否提升最终精度应使用甲方数据集对比验证。

实时相机入口如需调整采集尺寸，使用
`--camera-width W --camera-height H`；这两个参数不会覆盖 RKNN 模型输入尺寸。

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
- `ellipses/原图名.txt`：类别、椭圆参数、拟合来源/误差、轮廓覆盖角、协方差标准差、
  条件数、时序/几何门控状态和二次曲线系数。
- `poses/原图名.txt`：参考类别与中心、孔中心，以及自动/固定距离两套 yaw、pitch、roll、tx、ty、tz。
- 视频输入还会在输出根目录生成 `<视频名>_result.mp4`，编码不可用时自动回退为 AVI。

### 仅 Mask 与双圆环椭圆的演示版本

`batch_mask_ring_visualization` 与 `batch_image_video` 共用同一套 RKNN 推理、Mask
后处理、椭圆拟合、时序处理和结果保存代码，仅精简最终画面：

- 保留实例 Mask 叠加；
- 所有实例显示红色检测框；
- cls0 外圈拟合椭圆使用青色；
- cls1 中圈拟合椭圆使用紫色；
- 不显示内孔椭圆、中心点、可见弧散点、文字和姿态轴。

图片目录：

```bash
./install/batch_mask_ring_visualization \
  --model ./best.rknn \
  --input-path /data/images \
  --output-path /data/ring_demo \
  --show
```

视频：

```bash
./install/batch_mask_ring_visualization \
  --model ./best.rknn \
  --input-path /data/videos \
  --video-name test.mp4 \
  --output-path /data/ring_demo \
  --save-video-frames
```

输出目录仍为 `visual/`、`labels/`、`ellipses/`、`poses/`，视频结果仍保存为
`<视频名>_result.mp4`；TXT 字段与完整版完全一致，方便两种画面直接对照。

服务器 PT 模型可使用对应 Python 入口，参数和输出目录保持一致：

```bash
python unit_test/batch_mask_ring_visualization.py \
  --model best.pt \
  --input-path /data/images \
  --mode images \
  --output-path /data/ring_demo \
  --device 0 \
  --no-show
```

### C++ 串行批量性能分析

`unit_test/single_frame_pipeline.cpp` 把处理流程拆成六个互相独立的小模块：
`ImageLoader`、`RknnInference`、`EllipseFitter`、`PoseSolver`、`Visualizer` 和
`ResultWriter`。现在支持处理单张图片或目录内全部图片，并且只针对算法真实环节计时：
图片读取、预处理、RKNN 推理、后处理、椭圆拟合和位姿解算。

```bash
cmake --build build --target single_frame_pipeline -j
./install/single_frame_pipeline \
  --model ./best.rknn \
  --input-path /data/input/images \
  --output /data/performance_result
```

`--image /data/input/test.jpg` 仍可测试单张图片；子目录数据使用 `--recursive`。
该入口同样支持 `--hole-mask` 和 `--force-reference-box`。

程序默认每处理完一张图片就在屏幕显示一次完整可视化，但可视化绘制和
`imshow/waitKey` 不计入任何阶段耗时。无桌面板端可传 `--no-show`。该性能入口不保存
检测图片、标签、椭圆或位姿，只保存耗时统计：

```text
performance_result/timing/
├── per_image_timing.csv
├── timing_summary.csv
└── charts/
    ├── 01_read.png
    ├── 02_preprocess.png
    ├── 03_rknn_inference.png
    ├── 04_postprocess.png
    ├── 05_ellipse_fit.png
    └── 06_pose_solve.png
```

汇总表逐阶段给出样本数、平均值、P95、最小值、最大值和平均耗时对应的理论 FPS；
每张图同时标出逐图曲线、平均值线和 P95 线。该程序严格串行，测量的是单帧阶段延迟，
适合定位在线多线程入口中的 CPU/NPU 瓶颈。可视化、窗口刷新和统计文件自身的写盘
耗时均明确排除在统计之外。

## 圆环拟合现场切换

`npu_infer_pool` 默认对参考外/中圈（cls0/cls1）优先使用 Mask 拟合，内孔（cls2）使用检测框中心和内切圆。验收现场图像质量不足时，可在提交推理任务前开启参考圈保底模式：

### 椭圆模块快速阅读顺序

新人建议按下面的调用链阅读，不需要先通读整个后处理：

```text
推理入口选择 EllipseFitMode
    -> EllipseFitter::Fit                 总调度及 Box 回退
    -> CollectMaskPoints                  图像/检测框假边剔除和 p=0.5 亚像素修正
    -> CollectOuterRadialEnvelope         C形/环形 Mask 的径向最外层可见弧
    -> BuildCandidate / BuildGlobalCandidate
       -> FitRansac                       PROSAC 式采样和 LO-RANSAC
       -> RefineSampson                   Huber-Sampson LM 和协方差
       -> UpdateGeometryStatistics        覆盖度、标准差及退化门控
    -> MaskCandidateScore                 外层凸包 IoU + 全局 q90 残差选优
    -> RingPairRefiner                    外圈/中圈物理一致性选择
    -> EllipseTemporalFilter              视频质量自适应 EMA
    -> EllipseObservationSigmaPx          转为联合位姿解算的观测权重
    -> PoseEstimatorLM::SolveDual
       -> 完整圈：投影圆周点到观测椭圆
       -> 部分圈：解析投影圆锥曲线 + 双向裁剪 Chamfer
```

主要数据都集中在 `EllipseFitResult`：`ellipse` 是原图坐标下的最终椭圆，
`source` 表示 Mask/Edge/Box 来源，`quality` 用于候选选择和时序滤波，
`angular_coverage_deg`、`occupied_quadrants` 用于识别短弧退化，
`covariance` 和各项 `*_std_*` 表示拟合不确定度。`valid=true` 不代表一定来自
Mask；Box 兜底同样是可用结果，判断来源应读取 `source`。

默认外圈/中圈只采用两级策略：高质量 Mask 椭圆，或检测框内切圆。参考圈不执行灰度 Edge 拟合。
Mask 路径先从分割概率的 0.5 等值线获得亚像素轮廓，并删除图像边界及检测框裁剪产生的假直线。
对普通完整 Mask，同时生成稳健 RANSAC 候选和 Direct/AMS/标准拟合的全轮廓候选；对机械臂遮挡
形成的 C 形/环形 Mask，按角度只保留径向最外侧点并自动识别连续缺口，以开放外圆弧生成第三类候选。
最终按照外层轮廓凸包 IoU（70%）和全局 q90 Sampson 残差（30%）选优，小目标的内点门限随检测框
短边自适应收紧。Python/PT 与共享 C++ `EllipseFitter` 使用相同参数、候选和回退规则。
候选还必须通过内点率、轮廓角度/象限覆盖、检测框偏差、轴比及协方差条件数门控。完整目标不合格时
回退内切圆；出视场目标若可见弧不足则标记无效，避免裁剪后的检测框产生错误几何。需要无条件保底时
再显式开启强制内切圆。

两圈同时存在时，先做尺寸比、中心偏移、轴比和方向的几何门控，但不强制两个投影椭圆在图像上同心。
位姿层把外圈和中圈作为同一组共轴 3D 圆，在同一 LM 中联合重投影，并用各自拟合协方差换算的像素标准差加权。
Mask 拟合可信时权重高，Box 兜底时自动降权。视频/摄像头模式最后使用随质量自适应的椭圆 EMA，
并拒绝低质量突变。可视化中绿色为完整 Mask、橙色及 `PARTIAL` 表示部分可见但仍通过
几何门控、黄色为检测框内切圆、红色表示检测存在但可见弧不足，不能进入位姿解算。

### 出视场圆环处理

当检测框或 Mask 接触原图边界时，拟合器自动进入部分可见模式，不需要新增命令行参数：

1. 删除距原图边界 3 px 内的轮廓点，防止 Mask 被裁剪后生成的直线“封口”参与拟合。
2. 仍使用 PROSAC/LO-RANSAC 和 Huber-Sampson LM，但采用部分弧专用门限：默认至少
   `90°`、覆盖至少两个象限，并检查放宽后的协方差条件数和参数标准差。
3. 保留最终有效弧内点。双圆共轴位姿 LM 不再盲信短弧外推出的完整椭圆，而是把当前
   3D 圆环通过 `C=H^-T C_circle H^-1` 解析投影成图像圆锥曲线，直接最小化可见
   弧点到预测曲线的 Sampson 距离；相机存在畸变时自动退回离散投影方案。
4. 部分弧启用五个 yaw/pitch 初值的轻量多假设搜索，降低单个短弧初值落入局部最优的概率；
   完整目标不执行该搜索，不增加正常路径耗时。
5. 同时增加预测圆周到观测弧的反向最近距离，只保留与可见比例相当的一段预测圆周，
   防止过大或偏心曲线仅在局部擦过短弧。
6. 同类别存在多个检测时，每个外圈×中圈组合会通过共享三维位姿重投影分数复核；
   单一候选时跳过该步骤，不增加常规路径计算量。
7. 部分弧按可见比例扩大 `sigma`、质量分最高限制为 `0.70`，完整圆环会自然获得更高权重。
8. 可见弧小于门限时返回 `valid=false`，候选选择和位姿层会跳过它，避免输出看似正常但
   实际发散的姿态。若验收现场必须无条件出结果，仍可显式使用
   `--force-reference-box`，其优先级高于自动失效策略。

逐帧椭圆 TXT 和 Python JSON 新增 `valid`、`border_truncated`、`partial`、
`visible_arc_ratio`、`removed_border_points` 和 `support_points`，便于现场说明算法为何
接受或拒绝某一帧。常用部分弧门限在
`EllipseFitConfig`（C++）和 `Config`（`unit_test/final_version.py`）中均有中文注释。

### 实时性能策略

位姿 LM 使用数值雅可比，残差点数量和多初值次数会直接影响帧率。在线路径默认采用：

- 椭圆拟合仍保留最多180个点用于稳健统计，位姿优化均匀抽取最多48个可见弧点。
- 只有“没有完整圆环且可见弧不足约半圈”才运行五组多初值，每组粗优化3轮。
- 部分弧最终 LM 最多18～20轮；完整圆环仍保留原迭代预算。
- 在线 `npu_infer_pool` 每帧只解算当前显示的 auto 或 fixed 模式，不再重复求两套位姿。
- 三维候选复核最多处理4个外圈×中圈组合，并使用6轮、无多初值的快速评分。

Python 配置中的 `pose_max_arc_points=48` 控制位姿弧点上限；
`compute_both_pose_modes=false`（默认）只计算当前显示模式。离线需要同时导出自动距离和
固定距离结果时将其设为 `true`，代价是位姿解算耗时接近翻倍。

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
