// 仅用于演示 Mask 与参考双圆环拟合结果。
//
// 复用 batch_image_video.cpp 的推理、拟合、时序、TXT 和视频保存实现，避免两个
// 测试入口以后出现算法参数不一致。这个编译开关只改变可视化内容：
//   - 保留所有实例 Mask；
//   - 画出所有实例的红色检测框；
//   - 只画 cls0 外圈（青色）和 cls1 中圈（紫色）的最终拟合椭圆；
//   - 不画内孔、中心点、可见弧散点、文字和姿态轴。
//
// 命令行参数和输出目录结构与 batch_image_video 完全相同。
#define BATCH_RING_MASK_ONLY 1
#include "batch_image_video.cpp"
