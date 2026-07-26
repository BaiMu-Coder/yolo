#!/usr/bin/env python3
"""PT 版 Mask + 检测框 + 双圆环可视化。

推理、椭圆拟合、时序处理、命令行参数及输出格式全部复用 final_version.py，
只把最终画面切换成与 batch_mask_ring_visualization.cpp 一致的精简样式：
实例 Mask、红色检测框、青色 cls0 外圈和紫色 cls1 中圈。

示例：
python unit_test/batch_mask_ring_visualization.py \
  --model best.pt --input-path /data/images --mode images \
  --output-path /data/ring_demo --device 0 --no-show
"""

from final_version import CFG, main


if __name__ == "__main__":
    CFG.ring_mask_visualization = True
    CFG.draw_masks = True
    CFG.window_name = "Mask + Detection Boxes + Ring Ellipses"
    raise SystemExit(main())
