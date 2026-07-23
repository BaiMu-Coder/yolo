#!/usr/bin/env python3
"""final_version.cpp 的效果优先 Python/PT 版本。

依赖：ultralytics、torch、opencv-python、numpy。
服务器运行示例：
python final_version.py --model best.pt --input-path /data/images --mode images \
  --output-path /data/result --device 0 --no-show
输出目录和可视化效果对齐 batch_image_video.cpp。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# 配置区：数据、模型、算法和保存项都在这里控制
# =============================================================================
@dataclass
class Config:
    # 输入：video / camera / images
    source_mode: str = "video"
    # 视频模式只需设置：输入路径、输出路径、视频名
    input_path: str = "/home/seven.xu2/project/yolo/video"                # 输入视频所在文件夹
    output_path: str = "/home/seven.xu2/project/yolo/output"        # 所有结果保存的根文件夹
    video_name: str = "day--5.1.mp4"        # 输入视频名，输出会自动沿用该名称
    camera_id: int = 0
    model_path: str = "best.pt"          # 换成转换 RKNN 前的分割 PT 模型
    device: str = "0"                  # CPU: "cpu"；第一张显卡: "0"
    start_frame: int = 0
    end_frame: int = -1                  # -1=直到结束
    frame_step: int = 1
    camera_width: int = 640
    camera_height: int = 640
    camera_mjpg: bool = True

    # Ultralytics 推理
    image_size: int = 640
    confidence: float = 0.50
    iou_threshold: float = 0.45
    max_detections: int = 256
    half: bool = True                   # GPU 可改 True；CPU 保持 False
    retina_masks: bool = True            # 原图尺寸高质量 mask，效果优先
    mask_binary_threshold: float = 0.50
    class_names: Tuple[str, ...] = ("outer", "middle", "hole")
    outer_class_id: int = 0
    middle_class_id: int = 1
    hole_class_id: int = 2

    # 椭圆拟合（对应 C++ final_version 的参数）
    mask_fit_hole: bool = False
    force_reference_box: bool = False
    ellipse_deviation_ratio: float = 0.30
    ellipse_ransac_iters: int = 160
    ellipse_local_optimization_iters: int = 2
    ellipse_refinement_iters: int = 15
    ellipse_inlier_px: float = 3.0
    ellipse_robust_delta_px: float = 2.5
    ellipse_min_inlier_ratio: float = 0.42
    ellipse_max_axis_ratio: float = 6.0
    ellipse_max_points: int = 180
    ellipse_min_quality: float = 0.52
    ellipse_min_coverage_deg: float = 160.0
    ellipse_max_covariance_condition: float = 1e12
    # 出视场专用配置：先删除贴图像边界的“封口假轮廓”，再允许开放弧拟合。
    image_border_margin_px: int = 3
    partial_min_coverage_deg: float = 90.0
    partial_min_quadrants: int = 2
    partial_min_quality: float = 0.28
    partial_max_center_std_ratio: float = 0.25
    partial_max_axis_std_ratio: float = 0.45
    partial_max_covariance_condition: float = 1e14
    enable_edge_fallback: bool = True
    enable_ellipse_smoothing: bool = True

    # 相机内参、畸变与锥套物理尺寸（单位 mm，完全对应 C++）
    fx: float = 1639.6
    fy: float = 2165.4
    cx: float = 960.0
    cy: float = 540.0
    distortion: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    physical_scale: float = 20.0 / 45.5
    radius_outer_base_mm: float = 1200.0
    radius_middle_base_mm: float = 980.0
    radius_hole_base_mm: float = 120.0
    length_base_mm: float = 920.0
    pose_fixed_distance_mm: float = 3000.0
    pose_display_fixed: bool = False
    compute_both_pose_modes: bool = False  # 实时默认只解算当前显示模式；离线对比可开启
    pose_max_iters: int = 30
    pose_max_arc_points: int = 48       # 位姿数值雅可比用的均匀弧点上限
    signed_robust_residual: bool = False  # False 精确复现 C++；True 可试验改进

    # 抗抖：原始值和平滑值都会保存，画面默认显示平滑值
    # batch_image_video.cpp 不做位姿 EMA；默认关闭以保证服务器可视化一致。
    enable_pose_smoothing: bool = False
    pose_ema_alpha: float = 0.35
    keep_pose_frames_when_missing: int = 5
    draw_held_pose: bool = True

    output_fourcc: str = "mp4v"
    output_fps: float = 0.0              # 0=继承输入，无法获取则 25
    show_window: bool = True
    window_name: str = "YOLO Segmentation + Pose"
    window_width: int = 1280
    window_height: int = 720
    save_video: bool = True
    save_visual_frames: bool = False      # 对应 C++ --save-video-frames
    visual_frame_every: int = 1
    save_masks: bool = False              # C++ batch 默认不单独保存 Mask
    mask_every: int = 1
    mask_alpha: float = 0.50
    draw_boxes: bool = True
    draw_masks: bool = True
    draw_all_ellipses: bool = True
    draw_pose_axis: bool = True
    draw_fps: bool = False                # batch_image_video.cpp 默认不叠加 FPS


CFG = Config()
# =============================================================================


def configured_input_path(cfg: Config) -> Path:
    """由唯一数据根目录和视频/目录名拼出输入路径。"""
    root = Path(cfg.input_path).expanduser()
    return root / cfg.video_name if cfg.video_name else root


def configured_output_dir(cfg: Config) -> Path:
    """与 batch_image_video.cpp 一致：--output-path 本身就是本次输出根目录。"""
    return Path(cfg.output_path).expanduser()


def configured_output_video_name(cfg: Config) -> str:
    stem = f"camera_{cfg.camera_id}" if cfg.source_mode.lower() == "camera" else configured_input_path(cfg).stem
    return f"{stem}_result.mp4"


Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    class_id: int
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    mask_probability: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class EllipseResult:
    ellipse: Ellipse
    valid: bool = True
    source: str = "box"
    inliers: int = 0
    mean_error_px: Optional[float] = None
    sampled_points: int = 0
    inlier_ratio: float = 0.0
    center_deviation_ratio: float = 0.0
    quality: float = 0.0
    geometry_consistent: bool = True
    temporally_filtered: bool = False
    border_truncated: bool = False
    partial_visibility: bool = False
    removed_border_points: int = 0
    visible_arc_ratio: float = 0.0
    visible_arc_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32), repr=False)
    angular_coverage_deg: float = 0.0
    occupied_quadrants: int = 0
    center_std_px: float = math.inf
    major_axis_std_px: float = math.inf
    minor_axis_std_px: float = math.inf
    angle_std_deg: float = math.inf
    covariance_condition: float = math.inf
    uncertainty_valid: bool = False
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((5, 5), dtype=np.float64),
                                   repr=False)
    conic: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64),
                              repr=False)

    @property
    def from_mask(self) -> bool:
        return self.source == "mask"


@dataclass
class Pose6D:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    tx_mm: float = 0.0
    ty_mm: float = 0.0
    tz_mm: float = 0.0

    def array(self) -> np.ndarray:
        return np.asarray([self.yaw_deg, self.pitch_deg, self.roll_deg,
                           self.tx_mm, self.ty_mm, self.tz_mm], dtype=np.float64)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "Pose6D":
        return cls(*[float(x) for x in values])


class PTModel:
    def __init__(self, cfg: Config):
        model_path = Path(cfg.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"PT 模型不存在: {model_path}")
        self.cfg = cfg
        self.model = YOLO(str(model_path), task="segment")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        cfg = self.cfg
        result = self.model.predict(
            source=frame, imgsz=cfg.image_size, conf=cfg.confidence,
            iou=cfg.iou_threshold, max_det=cfg.max_detections,
            device=cfg.device, half=cfg.half, retina_masks=cfg.retina_masks,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
        masks: Optional[np.ndarray] = None
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            if masks.shape[1:] != frame.shape[:2]:
                masks = np.stack([
                    cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    for m in masks
                ])
        detections: List[Detection] = []
        for i, box in enumerate(xyxy):
            x1 = int(np.clip(round(float(box[0])), 0, frame.shape[1]))
            y1 = int(np.clip(round(float(box[1])), 0, frame.shape[0]))
            x2 = int(np.clip(round(float(box[2])), 0, frame.shape[1]))
            y2 = int(np.clip(round(float(box[3])), 0, frame.shape[0]))
            if x2 <= x1 or y2 <= y1:
                continue
            mask = None
            mask_probability = None
            if masks is not None and i < len(masks):
                probability = np.clip(masks[i], 0.0, 1.0).astype(np.float32)
                mask = (probability >= cfg.mask_binary_threshold).astype(np.uint8) * 255
                # 与 C++ 一致，只保留检测框内的实例 mask。
                clipped = np.zeros_like(mask)
                clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                mask = clipped
                mask_probability = np.zeros_like(probability)
                mask_probability[y1:y2, x1:x2] = probability[y1:y2, x1:x2]
            detections.append(Detection(x1, y1, x2 - x1, y2 - y1,
                                        float(scores[i]), int(classes[i]), mask,
                                        mask_probability))
        return detections


def radial_errors(ellipse: Ellipse, points: np.ndarray) -> np.ndarray:
    (cx, cy), (width, height), angle = ellipse
    a, b = width * 0.5, height * 0.5
    if a < 1e-3 or b < 1e-3:
        return np.full(len(points), 1e9, dtype=np.float32)
    p = points.astype(np.float32) - np.asarray([cx, cy], dtype=np.float32)
    rad = math.radians(-angle)
    c, s = math.cos(rad), math.sin(rad)
    xr = c * p[:, 0] - s * p[:, 1]
    yr = s * p[:, 0] + c * p[:, 1]
    radius = np.sqrt(xr * xr / (a * a) + yr * yr / (b * b))
    return np.abs(radius - 1.0) * min(a, b)


def signed_sampson_errors(ellipse: Ellipse, points: np.ndarray) -> np.ndarray:
    (cx, cy), (width, height), angle = ellipse
    a, b = max(width * 0.5, 1e-6), max(height * 0.5, 1e-6)
    p = points.astype(np.float64) - np.asarray([cx, cy], dtype=np.float64)
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    x, y = c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]
    fx, fy = 2.0 * x / (a * a), 2.0 * y / (b * b)
    gradient = np.hypot(c * fx - s * fy, s * fx + c * fy)
    return (x * x / (a * a) + y * y / (b * b) - 1.0) / np.maximum(gradient, 1e-9)


def conic_sampson_errors(conic: np.ndarray, points: np.ndarray) -> np.ndarray:
    """点到一般二次曲线的带符号 Sampson 像素距离。"""
    points = points.astype(np.float64)
    x, y = points[:, 0], points[:, 1]
    value = (conic[0, 0] * x * x + 2.0 * conic[0, 1] * x * y +
             2.0 * conic[0, 2] * x + conic[1, 1] * y * y +
             2.0 * conic[1, 2] * y + conic[2, 2])
    gx = 2.0 * (conic[0, 0] * x + conic[0, 1] * y + conic[0, 2])
    gy = 2.0 * (conic[0, 1] * x + conic[1, 1] * y + conic[1, 2])
    return value / np.maximum(np.hypot(gx, gy), 1e-12)


def ellipse_from_parameters(parameters: np.ndarray) -> Ellipse:
    """C++ LM 的五参数表示：[cx, cy, log(a), log(b), angle_rad]。"""
    return ((float(parameters[0]), float(parameters[1])),
            (float(2.0 * math.exp(parameters[2])),
             float(2.0 * math.exp(parameters[3]))),
            float(math.degrees(parameters[4])))


def ellipse_conic_matrix(ellipse: Ellipse) -> np.ndarray:
    """生成与 C++ EllipseConicMatrix 相同的 Frobenius 归一化二次曲线矩阵。"""
    (cx, cy), (width, height), angle = ellipse
    a, b = max(1e-6, width * 0.5), max(1e-6, height * 0.5)
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    quadratic = rotation @ np.diag((1.0 / (a * a), 1.0 / (b * b))) @ rotation.T
    center = np.asarray([cx, cy], dtype=np.float64)
    linear = -(quadratic @ center)
    conic = np.asarray([[quadratic[0, 0], quadratic[0, 1], linear[0]],
                        [quadratic[1, 0], quadratic[1, 1], linear[1]],
                        [linear[0], linear[1], center @ quadratic @ center - 1.0]],
                       dtype=np.float64)
    return conic / max(1e-15, float(np.linalg.norm(conic)))


def refine_sampson_lm(ellipse: Ellipse, points: np.ndarray, point_weights: np.ndarray,
                      cfg: Config) -> Optional[Tuple[Ellipse, np.ndarray, float]]:
    """复现 C++ RefineSampson：Huber IRLS + 数值雅可比 + LM + 协方差。"""
    (cx, cy), (width, height), angle = ellipse
    parameters = np.asarray([cx, cy, math.log(max(1.0, width * 0.5)),
                             math.log(max(1.0, height * 0.5)),
                             math.radians(angle)], dtype=np.float64)
    damping = 1e-3
    final_normal: Optional[np.ndarray] = None
    final_sse = 0.0
    final_count = 0
    epsilons = np.asarray((0.02, 0.02, 1e-4, 1e-4, 1e-5), dtype=np.float64)

    for _ in range(cfg.ellipse_refinement_iters):
        current = ellipse_from_parameters(parameters)
        residual = signed_sampson_errors(current, points)
        finite = np.isfinite(residual)
        if int(np.count_nonzero(finite)) < 6:
            return None
        used_points = points[finite]
        used_residual = residual[finite]
        base_weights = np.maximum(0.01, point_weights[finite])
        absolute = np.abs(used_residual)
        robust_weights = np.where(absolute <= cfg.ellipse_robust_delta_px, 1.0,
                                  cfg.ellipse_robust_delta_px / np.maximum(absolute, 1e-12))
        weights = base_weights * robust_weights
        jacobian = np.empty((len(used_points), 5), dtype=np.float64)
        for column, epsilon in enumerate(epsilons):
            perturbed = parameters.copy()
            perturbed[column] += epsilon
            jacobian[:, column] = (
                signed_sampson_errors(ellipse_from_parameters(perturbed), used_points) -
                used_residual) / epsilon
        normal = jacobian.T @ (weights[:, None] * jacobian)
        gradient = jacobian.T @ (weights * used_residual)
        cost = float(np.sum(base_weights * np.where(
            absolute <= cfg.ellipse_robust_delta_px,
            0.5 * used_residual * used_residual,
            cfg.ellipse_robust_delta_px *
            (absolute - 0.5 * cfg.ellipse_robust_delta_px))))
        final_normal = normal.copy()
        final_sse = float(np.sum(weights * used_residual * used_residual))
        final_count = len(used_points)
        damped = normal + damping * np.diag(np.maximum(1.0, np.diag(normal)))
        try:
            delta = np.linalg.lstsq(damped, -gradient, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        trial = parameters + delta
        if math.exp(trial[2]) < 1.0 or math.exp(trial[3]) < 1.0:
            damping *= 5.0
            continue
        trial_residual = signed_sampson_errors(ellipse_from_parameters(trial), points)
        trial_absolute = np.abs(trial_residual)
        trial_cost = float(np.sum(point_weights * np.where(
            trial_absolute <= cfg.ellipse_robust_delta_px,
            0.5 * trial_residual * trial_residual,
            cfg.ellipse_robust_delta_px *
            (trial_absolute - 0.5 * cfg.ellipse_robust_delta_px))))
        if np.isfinite(trial_cost) and trial_cost < cost:
            parameters = trial
            damping = max(1e-9, damping * 0.4)
            if float(delta @ delta) < 1e-10:
                break
        else:
            damping = min(1e9, damping * 5.0)

    if final_normal is None:
        return None
    singular = np.linalg.svd(final_normal, compute_uv=False)
    condition = float(singular[0] / max(1e-15, singular[-1]))
    covariance = np.linalg.pinv(final_normal) * (final_sse / max(1, final_count - 5))
    if not np.all(np.isfinite(covariance)) or not math.isfinite(condition):
        return None
    return ellipse_from_parameters(parameters), covariance, condition


def update_ellipse_statistics(candidate: EllipseResult, points: np.ndarray,
                              det: Detection, cfg: Config,
                              partial_visibility: bool = False) -> None:
    errors = np.abs(signed_sampson_errors(candidate.ellipse, points))
    inliers = points[errors <= cfg.ellipse_inlier_px]
    if len(inliers) < 6:
        return
    (cx, cy), (width, height), angle = candidate.ellipse
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    local = inliers - np.asarray([cx, cy], dtype=np.float64)
    theta = np.mod(np.arctan2((-s * local[:, 0] + c * local[:, 1]) / max(height * 0.5, 1e-6),
                              (c * local[:, 0] + s * local[:, 1]) / max(width * 0.5, 1e-6)),
                   2.0 * math.pi)
    candidate.angular_coverage_deg = 5.0 * len(np.unique(np.minimum(71, (theta * 72 / (2 * math.pi)).astype(int))))
    candidate.occupied_quadrants = len(np.unique(np.minimum(3, (theta * 4 / (2 * math.pi)).astype(int))))
    candidate.visible_arc_ratio = min(1.0, candidate.angular_coverage_deg / 360.0)

    covariance = candidate.covariance
    candidate.center_std_px = float(math.sqrt(max(0.0, covariance[0, 0], covariance[1, 1])))
    width_std = width * math.sqrt(max(0.0, covariance[2, 2]))
    height_std = height * math.sqrt(max(0.0, covariance[3, 3]))
    candidate.major_axis_std_px = float(max(width_std, height_std))
    candidate.minor_axis_std_px = float(min(width_std, height_std))
    candidate.angle_std_deg = float(math.degrees(math.sqrt(max(0.0, covariance[4, 4]))))
    short_side = max(1.0, float(min(det.w, det.h)))
    max_condition = (cfg.partial_max_covariance_condition if partial_visibility
                     else cfg.ellipse_max_covariance_condition)
    max_center_std = (cfg.partial_max_center_std_ratio if partial_visibility else 0.10)
    max_axis_std = (cfg.partial_max_axis_std_ratio if partial_visibility else 0.18)
    candidate.uncertainty_valid = (np.isfinite(candidate.covariance_condition) and
        candidate.covariance_condition <= max_condition and
        candidate.center_std_px <= max_center_std * short_side and
        candidate.major_axis_std_px <= max_axis_std * short_side)


def ransac_ellipse(points: np.ndarray, point_weights: np.ndarray,
                   cfg: Config) -> Optional[EllipseResult]:
    if len(points) < 20:
        return None
    order = np.argsort(-point_weights, kind="stable")
    points, point_weights = points[order], point_weights[order]
    rng = np.random.default_rng(12345)
    best: Optional[EllipseResult] = None
    best_weighted_inliers = -1.0
    for iteration in range(cfg.ellipse_ransac_iters):
        # points 按 mask 梯度质量排序；逐步扩展采样池形成 PROSAC 式初值。
        pool_size = min(len(points), max(
            20, 20 + iteration * max(0, len(points) - 20) //
            max(1, cfg.ellipse_ransac_iters - 1)))
        sample = points[rng.choice(pool_size, 5, replace=False)].reshape(-1, 1, 2).astype(np.float32)
        try:
            ellipse = cv2.fitEllipseDirect(sample)
        except cv2.error:
            continue
        a, b = ellipse[1][0] * 0.5, ellipse[1][1] * 0.5
        if min(a, b) < 2.0 or max(a, b) / max(min(a, b), 1e-3) > cfg.ellipse_max_axis_ratio:
            continue
        errors = np.abs(signed_sampson_errors(ellipse, points))
        inlier_mask = errors <= cfg.ellipse_inlier_px
        count = int(np.count_nonzero(inlier_mask))
        weighted_inliers = float(np.sum(point_weights[inlier_mask]))
        mean = float(np.mean(errors[inlier_mask])) if count else math.inf
        if (best is None or weighted_inliers > best_weighted_inliers or
                (weighted_inliers == best_weighted_inliers and
                 mean < (best.mean_error_px or math.inf))):
            best_weighted_inliers = weighted_inliers
            best = EllipseResult(ellipse=ellipse, source="mask", inliers=count,
                                 mean_error_px=mean, sampled_points=len(points),
                                 inlier_ratio=count / len(points))
    required = max(20, int(math.ceil(cfg.ellipse_min_inlier_ratio * len(points))))
    if best is None or best.inliers < required:
        return None
    # 与 C++ 一致的两轮 LO-RANSAC：当前内点全部重新做 Direct 拟合。
    refined = best.ellipse
    for _ in range(cfg.ellipse_local_optimization_iters):
        mask = np.abs(signed_sampson_errors(refined, points)) <= cfg.ellipse_inlier_px
        if int(np.count_nonzero(mask)) < 5:
            return None
        try:
            refined = cv2.fitEllipseDirect(
                points[mask].reshape(-1, 1, 2).astype(np.float32))
        except cv2.error:
            return None
    errors = np.abs(signed_sampson_errors(refined, points))
    inlier_mask = errors <= cfg.ellipse_inlier_px
    count = int(np.count_nonzero(inlier_mask))
    if count < required:
        return None
    lm = refine_sampson_lm(refined, points[inlier_mask],
                           point_weights[inlier_mask], cfg)
    if lm is None:
        return None
    refined, covariance, condition = lm
    final_errors = np.abs(signed_sampson_errors(refined, points))
    final_mask = final_errors <= cfg.ellipse_inlier_px
    final_count = int(np.count_nonzero(final_mask))
    result = EllipseResult(
        ellipse=refined, source="mask", inliers=final_count,
        mean_error_px=float(np.mean(final_errors[final_mask])) if final_count else math.inf,
        sampled_points=len(points), inlier_ratio=final_count / len(points),
        covariance=covariance, covariance_condition=condition)
    return result


def _score_candidate(candidate: EllipseResult, det: Detection, cfg: Config) -> Optional[EllipseResult]:
    cx, cy = candidate.ellipse[0]
    box_center = (det.x + det.w * 0.5, det.y + det.h * 0.5)
    short_side = max(1.0, float(min(det.w, det.h)))
    candidate.center_deviation_ratio = math.hypot(cx - box_center[0], cy - box_center[1]) / short_side
    major, minor = max(candidate.ellipse[1]), min(candidate.ellipse[1])
    partial = candidate.partial_visibility
    center_limit = 0.80 if partial else cfg.ellipse_deviation_ratio
    minor_limit = 0.10 if partial else 0.20
    coverage_limit = cfg.partial_min_coverage_deg if partial else cfg.ellipse_min_coverage_deg
    quadrant_limit = cfg.partial_min_quadrants if partial else 3
    if (candidate.center_deviation_ratio > center_limit or minor < minor_limit * short_side or
            candidate.angular_coverage_deg < coverage_limit or
            candidate.occupied_quadrants < quadrant_limit or not candidate.uncertainty_valid):
        return None
    error = candidate.mean_error_px if candidate.mean_error_px is not None else math.inf
    error_quality = math.exp(-error / max(0.5, cfg.ellipse_inlier_px))
    # 允许部分弧中心有更大偏差用于“硬通过”，但软评分仍按完整圈标准降权。
    center_quality = max(0.0, 1.0 - candidate.center_deviation_ratio /
                         cfg.ellipse_deviation_ratio)
    if major > (4.0 if partial else 1.45) * max(det.w, det.h):
        return None
    bonus = 0.04 if candidate.source == "mask" else 0.0
    coverage_quality = min(1.0, candidate.angular_coverage_deg / 300.0)
    uncertainty_quality = max(0.0, 1.0 - candidate.center_std_px /
                              max(1.0, 0.10 * short_side))
    candidate.quality = min(1.0, 0.32 * candidate.inlier_ratio + 0.22 * error_quality +
                            0.16 * center_quality + 0.16 * coverage_quality +
                            0.14 * uncertainty_quality + bonus)
    if partial:
        # 短弧可以进入位姿优化，但不应获得与完整闭合轮廓相同的权重。
        candidate.quality *= min(1.0, candidate.visible_arc_ratio /
                                 max(cfg.ellipse_min_coverage_deg / 360.0, 1e-6))
        candidate.quality = min(candidate.quality, 0.70)
    return candidate


def _fit_points(points: np.ndarray, point_weights: np.ndarray, det: Detection,
                cfg: Config, source: str, partial_visibility: bool = False,
                removed_border_points: int = 0) -> Optional[EllipseResult]:
    if len(points) < 20:
        return None
    step = max(1, len(points) // cfg.ellipse_max_points)
    sampled_points = points[::step][:cfg.ellipse_max_points]
    sampled_weights = point_weights[::step][:cfg.ellipse_max_points]
    fitted = ransac_ellipse(sampled_points, sampled_weights, cfg)
    if fitted is None:
        return None
    (cx, cy), size, angle = fitted.ellipse
    fitted.ellipse = ((cx + det.x, cy + det.y), size, angle)
    fitted.source = source
    fitted.border_truncated = partial_visibility
    fitted.partial_visibility = partial_visibility
    fitted.removed_border_points = removed_border_points
    fitted.conic = ellipse_conic_matrix(fitted.ellipse)
    global_points = sampled_points.astype(np.float64) + np.asarray(
        [det.x, det.y], dtype=np.float64)
    update_ellipse_statistics(fitted, global_points, det, cfg, partial_visibility)
    # 只把最终 Sampson 内点交给位姿层，避免离群纹理直接参与重投影优化。
    final_inliers = np.abs(signed_sampson_errors(
        fitted.ellipse, global_points)) <= cfg.ellipse_inlier_px
    fitted.visible_arc_points = global_points[final_inliers].astype(np.float32)
    return _score_candidate(fitted, det, cfg)


def best_ellipse(frame: np.ndarray, det: Detection, cfg: Config,
                 force_box: bool = False, allow_edge: bool = True) -> EllipseResult:
    height, width = frame.shape[:2]
    margin = max(0, cfg.image_border_margin_px)
    border_truncated = (det.x <= margin or det.y <= margin or
                        det.x + det.w >= width - margin or
                        det.y + det.h >= height - margin)
    box_center = (det.x + det.w * 0.5, det.y + det.h * 0.5)
    side = float(min(det.w, det.h))
    fallback = EllipseResult((box_center, (side, side), 0.0), source="box", quality=0.30,
                             border_truncated=border_truncated,
                             partial_visibility=border_truncated,
                             center_std_px=max(2.0, 0.12 * side),
                             major_axis_std_px=0.20 * side,
                             minor_axis_std_px=0.20 * side,
                             angle_std_deg=90.0)
    fallback.conic = ellipse_conic_matrix(fallback.ellipse)
    if force_box:
        return fallback
    candidates: List[EllipseResult] = []
    if det.mask is not None and not force_box:
        roi = det.mask[det.y:det.y + det.h, det.x:det.x + det.w].copy()
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        selected: List[np.ndarray] = []
        area_min = 0.005 * det.w * det.h
        distance_limit = 0.80 * min(det.w, det.h)
        for contour in contours:
            if len(contour) < 5 or abs(cv2.contourArea(contour)) < area_min:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            if math.hypot(cx - det.w * 0.5, cy - det.h * 0.5) <= distance_limit:
                selected.append(contour.reshape(-1, 2))
        if selected:
            points = np.concatenate(selected).astype(np.float32)
            point_weights = np.ones(len(points), dtype=np.float64)
            # findContours 会沿图像裁剪边界人为“封口”。这些直线不是目标真实边缘，
            # 若保留会把短弧拟合强烈拉向画面边缘，因此按原图坐标先删除。
            global_x = points[:, 0] + det.x
            global_y = points[:, 1] + det.y
            keep = ((global_x > margin) & (global_y > margin) &
                    (global_x < width - 1 - margin) &
                    (global_y < height - 1 - margin))
            removed_border_points = int(len(points) - np.count_nonzero(keep))
            points = points[keep]
            point_weights = point_weights[keep]
            partial_visibility = border_truncated or removed_border_points > 0
            # 在软 mask 的 p=0.5 等值线上做一步法向亚像素修正。
            # 梯度越清晰的点越靠前，供 PROSAC 优先采样。
            if det.mask_probability is not None:
                probability = det.mask_probability[det.y:det.y + det.h,
                                                   det.x:det.x + det.w].astype(np.float32)
                gx = cv2.Sobel(probability, cv2.CV_32F, 1, 0, ksize=3) / 8.0
                gy = cv2.Sobel(probability, cv2.CV_32F, 0, 1, ksize=3) / 8.0
                xi = np.clip(np.rint(points[:, 0]).astype(int), 0, max(0, det.w - 1))
                yi = np.clip(np.rint(points[:, 1]).astype(int), 0, max(0, det.h - 1))
                grad_x, grad_y = gx[yi, xi], gy[yi, xi]
                magnitude = np.hypot(grad_x, grad_y)
                safe = np.maximum(magnitude, 1e-6)
                shift = np.clip((0.5 - probability[yi, xi]) / safe, -0.75, 0.75)
                points[:, 0] += shift * grad_x / safe
                points[:, 1] += shift * grad_y / safe
                point_weights = np.clip(magnitude * 4.0, 0.15, 1.0).astype(np.float64)
            if len(points) >= 20:
                fitted = _fit_points(points, point_weights, det, cfg, "mask",
                                     partial_visibility, removed_border_points)
                if fitted is not None:
                    candidates.append(fitted)
    # 截断场景禁用 Edge：图像边界和反光边缘往往比真实开放弧更强，风险高于收益。
    if (allow_edge and not border_truncated and cfg.enable_edge_fallback and
            (not candidates or max(x.quality for x in candidates) < 0.72)):
        roi_gray = cv2.cvtColor(frame[det.y:det.y + det.h, det.x:det.x + det.w], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 1.2)
        edges = cv2.Canny(cv2.equalizeHist(roi_gray), 45, 135, L2gradient=True)
        edges[:3, :], edges[-3:, :], edges[:, :3], edges[:, -3:] = 0, 0, 0, 0
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        edge_points = []
        distance_limit = 0.80 * min(det.w, det.h)
        for contour in contours:
            if len(contour) < 18:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if math.hypot(x + 0.5 * w - 0.5 * det.w,
                          y + 0.5 * h - 0.5 * det.h) <= distance_limit:
                edge_points.append(contour.reshape(-1, 2))
        if edge_points:
            points = np.concatenate(edge_points).astype(np.float32)
            fitted = _fit_points(points, np.ones(len(points), dtype=np.float64),
                                 det, cfg, "edge")
            if fitted is not None:
                candidates.append(fitted)
    if not candidates:
        # 自动模式下，可见弧不足时明确返回 invalid，防止把裁剪框内切圆误当成
        # 完整圆环参与位姿。手动 force_box 已在上方返回，仍保留验收兜底能力。
        if border_truncated:
            fallback.valid = False
            fallback.quality = 0.0
        return fallback
    best = max(candidates, key=lambda item: item.quality + (0.03 if item.source == "mask" else 0.0))
    threshold = cfg.partial_min_quality if best.partial_visibility else cfg.ellipse_min_quality
    if best.quality >= threshold:
        return best
    if border_truncated:
        fallback.valid = False
        fallback.quality = 0.0
    return fallback


class PoseEstimatorLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.k = np.asarray([[cfg.fx, 0.0, cfg.cx], [0.0, cfg.fy, cfg.cy], [0.0, 0.0, 1.0]],
                            dtype=np.float64)
        self.d = np.asarray(cfg.distortion, dtype=np.float64)
        scale = cfg.physical_scale
        self.radius_outer = cfg.radius_outer_base_mm * scale
        self.radius_middle = cfg.radius_middle_base_mm * scale
        self.radius_hole = cfg.radius_hole_base_mm * scale
        self.length = cfg.length_base_mm * scale
        self.points_outer = self._circle(self.radius_outer, 32)
        self.points_middle = self._circle(self.radius_middle, 32)

    @staticmethod
    def _circle(radius: float, count: int) -> np.ndarray:
        theta = np.arange(count) * (2.0 * math.pi / count)
        return np.stack((radius * np.cos(theta), radius * np.sin(theta),
                         np.zeros(count)), axis=1).astype(np.float32)

    @staticmethod
    def _rvec(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        yaw, pitch = np.radians([yaw_deg, pitch_deg])
        ry = np.asarray([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0],
                         [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float64)
        rx = np.asarray([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                         [0, math.sin(pitch), math.cos(pitch)]], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(rx @ ry)  # roll=0: Rz * Rx * Ry
        return rvec

    def _project_circle_conic(self, rvec: np.ndarray, tvec: np.ndarray,
                              radius: float) -> Optional[np.ndarray]:
        """无畸变针孔模型下解析投影平面圆，避免再次拟合离散投影点。"""
        if radius <= 1e-6 or np.linalg.norm(self.d) > 1e-12:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        homography = self.k @ np.column_stack(
            (rotation[:, 0], rotation[:, 1], tvec.reshape(3)))
        determinant = float(np.linalg.det(homography))
        if not math.isfinite(determinant) or abs(determinant) < 1e-12:
            return None
        inverse = np.linalg.inv(homography)
        circle = np.diag((1.0 / (radius * radius),
                          1.0 / (radius * radius), -1.0))
        conic = inverse.T @ circle @ inverse
        norm = float(np.linalg.norm(conic))
        return conic / norm if math.isfinite(norm) and norm > 1e-15 else None

    def _robust(self, values: np.ndarray, delta: float) -> np.ndarray:
        x = values / delta
        out = np.sqrt(2.0 * delta * delta * (np.sqrt(1.0 + x * x) - 1.0))
        return out * np.sign(values) if self.cfg.signed_robust_residual else out

    def _residual(self, x: np.ndarray, ellipse: Ellipse, hole: Tuple[float, float],
                  points: np.ndarray, fixed_tz: Optional[float]) -> np.ndarray:
        if fixed_tz is None:
            yaw, pitch, tx, ty, tz = x
        else:
            yaw, pitch, tx, ty = x
            tz = fixed_tz
        if tz < 100.0 or min(ellipse[1]) < 2.0:
            return np.full(len(points) + 1, 100000.0)
        rvec = self._rvec(float(yaw), float(pitch))
        tvec = np.asarray([tx, ty, tz], dtype=np.float64)
        projected, _ = cv2.projectPoints(points, rvec, tvec, self.k, self.d)
        projected = projected.reshape(-1, 2)
        hole_3d = np.asarray([[0.0, 0.0, -self.length]], dtype=np.float32)
        projected_hole, _ = cv2.projectPoints(hole_3d, rvec, tvec, self.k, self.d)
        hp = projected_hole.reshape(2)
        (ox, oy), (width, height), angle = ellipse
        rad = math.radians(angle)
        dx, dy = projected[:, 0] - ox, projected[:, 1] - oy
        lx = dx * math.cos(rad) + dy * math.sin(rad)
        ly = -dx * math.sin(rad) + dy * math.cos(rad)
        ellipse_error = (lx / (width * 0.5)) ** 2 + (ly / (height * 0.5)) ** 2 - 1.0
        outer = self._robust(ellipse_error * 20.0, 5.0)
        center_distance = math.hypot(hp[0] - hole[0], hp[1] - hole[1]) * 50.0
        return np.concatenate((outer, self._robust(np.asarray([center_distance]), 10.0)))

    def _optimize(self, initial: np.ndarray, ellipse: Ellipse, hole: Tuple[float, float],
                  points: np.ndarray, fixed_tz: Optional[float],
                  max_iters: Optional[int] = None) -> np.ndarray:
        x = initial.astype(np.float64).copy()
        damping = 1e-3
        residual = self._residual(x, ellipse, hole, points, fixed_tz)
        cost = 0.5 * float(residual @ residual)
        for _ in range(self.cfg.pose_max_iters if max_iters is None else max_iters):
            jac = np.empty((len(residual), len(x)), dtype=np.float64)
            for j in range(len(x)):
                eps = 1e-4 * (abs(x[j]) + 1.0)
                trial = x.copy()
                trial[j] += eps
                jac[:, j] = (self._residual(trial, ellipse, hole, points, fixed_tz) - residual) / eps
            a = jac.T @ jac + damping * np.eye(len(x))
            g = jac.T @ residual
            try:
                delta = np.linalg.lstsq(a, -g, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            if float(delta @ delta) < 1e-12:
                break
            trial = x + delta
            new_residual = self._residual(trial, ellipse, hole, points, fixed_tz)
            new_cost = 0.5 * float(new_residual @ new_residual)
            if new_cost < cost:
                x, residual, cost = trial, new_residual, new_cost
                damping = max(damping * 0.5, 1e-9)
            else:
                damping *= 2.0
                if damping > 1e9:
                    break
        return x

    def solve(self, ellipse: Ellipse, hole: Tuple[float, float], use_middle: bool,
              fixed_distance: Optional[float] = None,
              max_iters: Optional[int] = None) -> Pose6D:
        points = self.points_middle if use_middle else self.points_outer
        radius = self.radius_middle if use_middle else self.radius_outer
        (cx, cy), (width, height), _ = ellipse
        image_radius = max(width, height) * 0.5
        fmean = 0.5 * (self.cfg.fx + self.cfg.fy)
        tz = fixed_distance if fixed_distance is not None else (
            fmean * radius / image_radius if image_radius > 1.0 else 200.0)
        dx, dy = hole[0] - cx, hole[1] - cy
        yaw = pitch = 0.0
        if self.length > 10.0:
            yaw = float(np.clip(math.degrees(-math.atan(dx * tz / (fmean * self.length))), -30, 30))
            pitch = float(np.clip(math.degrees(math.atan(dy * tz / (fmean * self.length))), -30, 30))
        tx = (cx - self.cfg.cx) * tz / self.cfg.fx
        ty = (cy - self.cfg.cy) * tz / self.cfg.fy
        if fixed_distance is None:
            out = self._optimize(np.asarray([yaw, pitch, tx, ty, tz]), ellipse,
                                 hole, points, None, max_iters)
            return Pose6D(out[0], out[1], 0.0, out[2], out[3], out[4])
        out = self._optimize(np.asarray([yaw, pitch, tx, ty]), ellipse,
                             hole, points, fixed_distance, max_iters)
        return Pose6D(out[0], out[1], 0.0, out[2], out[3], fixed_distance)

    def _dual_residual(self, x: np.ndarray, outer: Optional[EllipseResult],
                       middle: Optional[EllipseResult], hole: Tuple[float, float],
                       hole_sigma: float, fixed_tz: Optional[float]) -> np.ndarray:
        if fixed_tz is None:
            yaw, pitch, tx, ty, tz = x
        else:
            yaw, pitch, tx, ty = x
            tz = fixed_tz
        observations = ((outer, self.points_outer, self.radius_outer),
                        (middle, self.points_middle, self.radius_middle))
        def pose_support(observation: EllipseResult) -> np.ndarray:
            points = observation.visible_arc_points
            maximum = max(5, self.cfg.pose_max_arc_points)
            if len(points) <= maximum:
                return points
            indices = np.floor(np.arange(maximum) * len(points) / maximum).astype(int)
            return points[indices]
        def observation_count(observation: Optional[EllipseResult],
                              complete_count: int) -> int:
            if observation is None or not observation.valid:
                return 0
            if observation.partial_visibility and len(observation.visible_arc_points) >= 5:
                reverse_count = min(complete_count, max(
                    5, int(round(complete_count * max(
                        0.10, min(1.0, observation.visible_arc_ratio))))))
                return len(pose_support(observation)) + reverse_count
            return complete_count
        expected = sum(observation_count(observation, len(points))
                       for observation, points, _ in observations) + 2
        if tz < 100.0:
            return np.full(expected, 10000.0, dtype=np.float64)
        rvec = self._rvec(float(yaw), float(pitch))
        tvec = np.asarray([tx, ty, tz], dtype=np.float64)
        residuals: List[np.ndarray] = []
        for observation, points, radius in observations:
            if observation is None or not observation.valid:
                continue
            projected, _ = cv2.projectPoints(points, rvec, tvec, self.k, self.d)
            if observation.partial_visibility and len(observation.visible_arc_points) >= 5:
                support = pose_support(observation)
                sigma = ellipse_observation_sigma(observation)
                predicted_conic = self._project_circle_conic(rvec, tvec, radius)
                if predicted_conic is not None:
                    standardized = conic_sampson_errors(
                        predicted_conic, support) / sigma
                else:
                    try:
                        predicted = cv2.fitEllipseDirect(projected.astype(np.float32))
                        standardized = signed_sampson_errors(
                            predicted, support) / sigma
                    except cv2.error:
                        standardized = np.full(len(support),
                                               10000.0, dtype=np.float64)
            else:
                standardized = signed_sampson_errors(
                    observation.ellipse,
                    projected.reshape(-1, 2)) / ellipse_observation_sigma(observation)
            # 联合优化必须保留残差符号，否则雅可比方向会被破坏。
            xh = standardized / 2.5
            robust = np.sqrt(2.0 * 2.5 * 2.5 * (np.sqrt(1.0 + xh * xh) - 1.0))
            residuals.append(np.copysign(robust, standardized))
            if observation.partial_visibility and len(observation.visible_arc_points) >= 5:
                # 反向裁剪 Chamfer：只选择与可见比例相当的预测弧段。
                model_points = projected.reshape(-1, 2).astype(np.float64)
                observed = support.astype(np.float64)
                nearest = np.sqrt(np.min(np.sum(
                    (model_points[:, None, :] - observed[None, :, :]) ** 2,
                    axis=2), axis=1))
                reverse_count = min(len(nearest), max(
                    5, int(round(len(nearest) * max(
                        0.10, min(1.0, observation.visible_arc_ratio))))))
                reverse_standardized = np.sort(nearest)[:reverse_count] / (1.5 * sigma)
                reverse_xh = reverse_standardized / 2.5
                reverse_robust = np.sqrt(
                    2.0 * 2.5 * 2.5 *
                    (np.sqrt(1.0 + reverse_xh * reverse_xh) - 1.0))
                residuals.append(reverse_robust)
        hole_3d = np.asarray([[0.0, 0.0, -self.length]], dtype=np.float32)
        projected_hole, _ = cv2.projectPoints(hole_3d, rvec, tvec, self.k, self.d)
        center_error = (projected_hole.reshape(2) - np.asarray(hole)) / max(0.5, hole_sigma)
        xh = center_error / 2.5
        robust_center = np.sqrt(2.0 * 2.5 * 2.5 * (np.sqrt(1.0 + xh * xh) - 1.0))
        residuals.append(np.copysign(robust_center, center_error))
        return np.concatenate(residuals)

    def _optimize_dual(self, initial: np.ndarray, outer: Optional[EllipseResult],
                       middle: Optional[EllipseResult], hole: Tuple[float, float],
                       hole_sigma: float, fixed_tz: Optional[float],
                       max_iters: Optional[int] = None) -> np.ndarray:
        x = initial.astype(np.float64).copy()
        damping = 1e-3
        residual = self._dual_residual(x, outer, middle, hole, hole_sigma, fixed_tz)
        cost = 0.5 * float(residual @ residual)
        for _ in range(self.cfg.pose_max_iters if max_iters is None else max_iters):
            jacobian = np.empty((len(residual), len(x)), dtype=np.float64)
            for column in range(len(x)):
                epsilon = 1e-4 * (abs(x[column]) + 1.0)
                trial = x.copy()
                trial[column] += epsilon
                jacobian[:, column] = (self._dual_residual(
                    trial, outer, middle, hole, hole_sigma, fixed_tz) - residual) / epsilon
            normal = jacobian.T @ jacobian
            gradient = jacobian.T @ residual
            damped = normal + damping * np.diag(np.maximum(1.0, np.diag(normal)))
            try:
                delta = np.linalg.lstsq(damped, -gradient, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            trial = x + delta
            trial[0:2] = np.clip(trial[0:2], -60.0, 60.0)
            if fixed_tz is None:
                trial[4] = np.clip(trial[4], 100.0, 50000.0)
            trial_residual = self._dual_residual(trial, outer, middle, hole, hole_sigma, fixed_tz)
            trial_cost = 0.5 * float(trial_residual @ trial_residual)
            if trial_cost < cost:
                x, residual, cost = trial, trial_residual, trial_cost
                damping = max(1e-9, damping * 0.5)
                if float(delta @ delta) < 1e-12:
                    break
            else:
                damping = min(1e9, damping * 3.0)
        return x

    def solve_dual(self, outer: Optional[EllipseResult], middle: Optional[EllipseResult],
                   hole: Tuple[float, float], hole_sigma: float,
                   fixed_distance: Optional[float] = None,
                   max_iters: Optional[int] = None,
                   enable_multistart: bool = True) -> Pose6D:
        available = [(outer, False), (middle, True)]
        available = [(item, is_middle) for item, is_middle in available if item is not None]
        if not available:
            return Pose6D()
        initializer, use_middle = min(available,
                                      key=lambda item: ellipse_observation_sigma(item[0]))
        initial = self.solve(initializer.ellipse, hole, use_middle, fixed_distance,
                             0 if initializer.partial_visibility else 8)
        values = initial.array()
        x = values[[0, 1, 3, 4]] if fixed_distance is not None else values[[0, 1, 3, 4, 5]]
        partial_items = [item for item in (outer, middle)
                         if item is not None and item.partial_visibility]
        has_partial = bool(partial_items)
        has_complete = any(item is not None and not item.partial_visibility
                           for item in (outer, middle))
        shortest_visible = min((item.visible_arc_ratio for item in partial_items),
                               default=1.0)
        if (enable_multistart and has_partial and not has_complete and
                shortest_visible < 0.55):
            # 仅对缺少完整圆且短于约半圈的困难帧做多初值，粗搜索每组只跑3轮。
            best_seed, best_cost = x, math.inf
            for yaw_offset, pitch_offset in ((0.0, 0.0), (-12.0, 0.0),
                                             (12.0, 0.0), (0.0, -12.0),
                                             (0.0, 12.0)):
                seed = x.copy()
                seed[0] += yaw_offset
                seed[1] += pitch_offset
                seed = self._optimize_dual(seed, outer, middle, hole, hole_sigma,
                                           fixed_distance, 3)
                residual = self._dual_residual(seed, outer, middle, hole,
                                               hole_sigma, fixed_distance)
                cost = float(residual @ residual)
                if math.isfinite(cost) and cost < best_cost:
                    best_seed, best_cost = seed, cost
            x = best_seed
        requested_iters = self.cfg.pose_max_iters if max_iters is None else max_iters
        effective_iters = (min(requested_iters, 20 if has_complete else 18)
                           if has_partial else requested_iters)
        result = self._optimize_dual(x, outer, middle, hole, hole_sigma,
                                     fixed_distance, effective_iters)
        tz = fixed_distance if fixed_distance is not None else float(result[4])
        if not np.all(np.isfinite(result)) or tz < 100.0:
            return initial
        return Pose6D(float(result[0]), float(result[1]), 0.0,
                      float(result[2]), float(result[3]), float(tz))

    def evaluate_dual_reprojection_score(
            self, outer: Optional[EllipseResult], middle: Optional[EllipseResult],
            hole: Tuple[float, float], hole_sigma: float) -> float:
        """共享三维位姿下的标准化稳健残差分数，用于多双环候选复核。"""
        pose = self.solve_dual(outer, middle, hole, hole_sigma, max_iters=6,
                               enable_multistart=False)
        if not np.all(np.isfinite(pose.array())) or pose.tz_mm < 100.0:
            return 0.0
        x = np.asarray([pose.yaw_deg, pose.pitch_deg, pose.tx_mm,
                        pose.ty_mm, pose.tz_mm], dtype=np.float64)
        residual = self._dual_residual(x, outer, middle, hole, hole_sigma, None)
        if not len(residual) or not np.all(np.isfinite(residual)):
            return 0.0
        rms = math.sqrt(float(residual @ residual) / len(residual))
        return math.exp(-0.5 * rms * rms)

    def draw_axis(self, image: np.ndarray, pose: Pose6D, use_middle: bool) -> None:
        length = self.radius_middle if use_middle else self.radius_outer
        points = np.asarray([[0, 0, 0], [length, 0, 0], [0, length, 0],
                             [0, 0, length * 3.0]], dtype=np.float32)
        projected, _ = cv2.projectPoints(points, self._rvec(pose.yaw_deg, pose.pitch_deg),
                                         pose.array()[3:6], self.k, self.d)
        p = np.rint(projected.reshape(-1, 2)).astype(int)
        cv2.line(image, tuple(p[0]), tuple(p[1]), (0, 0, 255), 3)
        cv2.line(image, tuple(p[0]), tuple(p[2]), (0, 255, 0), 3)
        cv2.line(image, tuple(p[0]), tuple(p[3]), (255, 0, 0), 3)


class PoseSmoother:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.value: Optional[np.ndarray] = None
        self.missing = 0

    def update(self, pose: Optional[Pose6D]) -> Tuple[Optional[Pose6D], bool]:
        if pose is not None:
            current = pose.array()
            if self.value is None or not self.cfg.enable_pose_smoothing:
                self.value = current
            else:
                alpha = self.cfg.pose_ema_alpha
                self.value = alpha * current + (1.0 - alpha) * self.value
            self.missing = 0
            return Pose6D.from_array(self.value), False
        self.missing += 1
        if self.value is not None and self.missing <= self.cfg.keep_pose_frames_when_missing:
            return Pose6D.from_array(self.value), True
        return None, False


def refine_ring_pair(outer: EllipseResult, middle: EllipseResult) -> Tuple[bool, float]:
    """双圆环几何一致性门控；不强制图像椭圆同心。"""
    outer_major, outer_minor = max(outer.ellipse[1]), min(outer.ellipse[1])
    middle_major, middle_minor = max(middle.ellipse[1]), min(middle.ellipse[1])
    if min(outer_major, outer_minor, middle_major, middle_minor) <= 0:
        return False, 0.0
    size_error = abs(middle_major / outer_major - 980.0 / 1200.0) / 0.22
    center_error = math.dist(outer.ellipse[0], middle.ellipse[0]) / max(1.0, outer_major * 0.14)
    axis_error = abs(outer_major / outer_minor - middle_major / middle_minor) / 0.28
    ao = (outer.ellipse[2] + (90.0 if outer.ellipse[1][0] < outer.ellipse[1][1] else 0.0)) % 180.0
    am = (middle.ellipse[2] + (90.0 if middle.ellipse[1][0] < middle.ellipse[1][1] else 0.0)) % 180.0
    angle_error = min(abs(ao - am), 180.0 - abs(ao - am)) / 24.0
    consistent = max(size_error, center_error, axis_error, angle_error) <= 1.0
    score = max(0.0, 1.0 - (0.34 * size_error + 0.34 * center_error +
                            0.18 * axis_error + 0.14 * angle_error))
    outer.geometry_consistent = middle.geometry_consistent = consistent
    if not consistent:
        if outer.quality >= middle.quality:
            middle.quality *= 0.35
        else:
            outer.quality *= 0.35
        return False, score
    for ellipse in (outer, middle):
        ellipse.quality = min(1.0, ellipse.quality + 0.12 * score)
    return True, score


def ellipse_selection_score(ellipse: EllipseResult, detection_score: float) -> float:
    """对应 C++ EllipseSelectionScore。"""
    if not ellipse.valid:
        return -1e6
    geometry_penalty = 1.0 if ellipse.geometry_consistent else 0.72
    return (0.55 * max(0.0, min(1.0, detection_score)) +
            0.45 * max(0.0, min(1.0, ellipse.quality)) * geometry_penalty)


def select_ring_pair(detections: Sequence[Detection], ellipses: List[EllipseResult],
                     outer_class_id: int, middle_class_id: int,
                     estimator: Optional[PoseEstimatorLM] = None,
                     hole: Optional[Tuple[float, float]] = None,
                     hole_sigma: float = 10.0) -> Tuple[int, int]:
    outers = [i for i, det in enumerate(detections)
              if det.class_id == outer_class_id and ellipses[i].valid]
    middles = [i for i, det in enumerate(detections)
               if det.class_id == middle_class_id and ellipses[i].valid]
    if not outers or not middles:
        return (pick_best(detections, outer_class_id, ellipses),
                pick_best(detections, middle_class_id, ellipses))
    best_score, best_pair = -math.inf, (-1, -1)
    for outer_idx in outers:
        for middle_idx in middles:
            outer, middle = copy.deepcopy(ellipses[outer_idx]), copy.deepcopy(ellipses[middle_idx])
            consistent, consistency_score = refine_ring_pair(outer, middle)
            geometry_term = 0.45 * consistency_score if consistent else -0.35
            pose_term = 0.0
            pair_count = len(outers) * len(middles)
            if (estimator is not None and hole is not None and
                    1 < pair_count <= 4):
                pose_term = 0.65 * estimator.evaluate_dual_reprojection_score(
                    outer, middle, hole, hole_sigma) * math.sqrt(
                        max(0.0, outer.quality * middle.quality))
            score = (ellipse_selection_score(outer, detections[outer_idx].score) +
                     ellipse_selection_score(middle, detections[middle_idx].score) +
                     geometry_term + pose_term)
            if score > best_score:
                best_score, best_pair = score, (outer_idx, middle_idx)
    refine_ring_pair(ellipses[best_pair[0]], ellipses[best_pair[1]])
    return best_pair


class EllipseSmoother:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.values: Dict[int, EllipseResult] = {}

    def update(self, class_id: int, current: EllipseResult) -> EllipseResult:
        previous = self.values.get(class_id)
        if previous is None or not self.cfg.enable_ellipse_smoothing:
            self.values[class_id] = current
            return current
        previous_major = max(1.0, max(previous.ellipse[1]))
        jump = math.dist(previous.ellipse[0], current.ellipse[0]) / previous_major
        size_change = abs(max(current.ellipse[1]) - previous_major) / previous_major
        if (jump > 0.45 or size_change > 0.55) and current.quality < 0.62:
            held = EllipseResult(**{**previous.__dict__})
            held.quality *= 0.92
            held.temporally_filtered = True
            # 上一帧可见弧不能与当前帧孔中心混用；保持帧退回低权重完整椭圆先验。
            held.visible_arc_points = np.empty((0, 2), dtype=np.float32)
            held.partial_visibility = False
            self.values[class_id] = held
            return held
        alpha = 0.18 + (0.72 - 0.18) * max(0.0, min(1.0, current.quality))
        pc, cc = np.asarray(previous.ellipse[0]), np.asarray(current.ellipse[0])
        ps, cs = np.asarray(previous.ellipse[1]), np.asarray(current.ellipse[1])
        center, size = (1.0 - alpha) * pc + alpha * cc, (1.0 - alpha) * ps + alpha * cs
        delta = ((current.ellipse[2] - previous.ellipse[2] + 90.0) % 180.0) - 90.0
        current.ellipse = ((float(center[0]), float(center[1])),
                           (float(size[0]), float(size[1])),
                           float((previous.ellipse[2] + alpha * delta) % 180.0))
        current.conic = ellipse_conic_matrix(current.ellipse)
        current.temporally_filtered = True
        self.values[class_id] = current
        return current


def draw_text(image: np.ndarray, text: str, y: int, color: Tuple[int, int, int],
              scale: float = 1.0, thickness: int = 2) -> None:
    cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def pick_best(detections: Sequence[Detection], class_id: int,
              ellipses: Optional[Sequence[EllipseResult]] = None) -> int:
    indices = [i for i, det in enumerate(detections)
               if det.class_id == class_id and
               (ellipses is None or (i < len(ellipses) and ellipses[i].valid))]
    if not indices:
        return -1
    return max(indices, key=lambda i: detections[i].score if ellipses is None else
               ellipse_selection_score(ellipses[i], detections[i].score))


def pose_dict(pose: Optional[Pose6D]) -> Optional[Dict[str, float]]:
    if pose is None or not np.all(np.isfinite(pose.array())) or pose.tz_mm < 100.0:
        return None
    return {k: float(v) for k, v in asdict(pose).items()}


def valid_pose(pose: Optional[Pose6D]) -> Optional[Pose6D]:
    """阻止单帧退化拟合产生的 NaN/Inf 污染平滑器和输出文件。"""
    if pose is None or not np.all(np.isfinite(pose.array())) or pose.tz_mm < 100.0:
        return None
    return pose


def ellipse_observation_sigma(ellipse: EllipseResult) -> float:
    """把拟合误差/协方差统一成像素标准差；Box 兜底自动降权。"""
    major = max(ellipse.ellipse[1])
    if ellipse.source == "box" or not ellipse.uncertainty_valid:
        return max(8.0, 0.12 * major)
    residual = ellipse.mean_error_px if ellipse.mean_error_px is not None else 3.0
    sigma = max(0.5, math.hypot(residual, 0.5 * ellipse.center_std_px))
    sigma /= max(0.25, ellipse.quality)
    sigma *= (1.0 if ellipse.geometry_consistent else 3.0)
    if ellipse.partial_visibility:
        sigma *= 1.5 / max(0.25, ellipse.visible_arc_ratio)
    return sigma


def process_visual(frame: np.ndarray, detections: List[Detection], cfg: Config,
                   estimator: PoseEstimatorLM, smoother: PoseSmoother,
                   ellipse_smoother: EllipseSmoother,
                   frame_id: int, masks_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    vis = frame.copy()
    colors = ((0, 0, 255), (0, 255, 0), (255, 255, 0))
    ellipses: List[EllipseResult] = []
    for i, det in enumerate(detections):
        is_reference = det.class_id in (cfg.outer_class_id, cfg.middle_class_id)
        force_box = ((is_reference and cfg.force_reference_box) or
                     (det.class_id == cfg.hole_class_id and not cfg.mask_fit_hole))
        ellipse = best_ellipse(frame, det, cfg, force_box, allow_edge=not is_reference)
        ellipses.append(ellipse)
        if cfg.draw_masks and det.mask is not None and 0 <= det.class_id < len(colors):
            active = det.mask > 0
            color = np.asarray(colors[det.class_id], dtype=np.float32)
            vis[active] = np.clip(vis[active] * (1.0 - cfg.mask_alpha) + color * cfg.mask_alpha,
                                  0, 255).astype(np.uint8)
        if cfg.draw_boxes:
            cv2.rectangle(vis, (det.x, det.y), (det.x + det.w, det.y + det.h), (0, 0, 255), 2)
            cv2.putText(vis, f"cls={det.class_id} conf={det.score:.2f} "
                        f"q={ellipse.quality:.2f} {ellipse.source}"
                        f"{' PARTIAL' if ellipse.partial_visibility else ''}",
                        (det.x, max(18, det.y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if cfg.draw_all_ellipses:
            color = ((0, 0, 255) if not ellipse.valid else
                     ((0, 165, 255) if ellipse.partial_visibility else
                      ((0, 255, 0) if ellipse.source == "mask" else
                       ((255, 0, 255) if ellipse.source == "edge" else (0, 255, 255)))))
            cv2.ellipse(vis, ellipse.ellipse, color, 2)
            cv2.circle(vis, tuple(np.rint(ellipse.ellipse[0]).astype(int)), 2, (0, 0, 255), -1)
            if ellipse.partial_visibility:
                for point in ellipse.visible_arc_points[::4]:
                    cv2.circle(vis, tuple(np.rint(point).astype(int)), 1,
                               (0, 165, 255), -1)
        if cfg.save_masks and det.mask is not None and frame_id % max(1, cfg.mask_every) == 0:
            cv2.imwrite(str(masks_dir / f"frame_{frame_id:08d}_det_{i:03d}_cls_{det.class_id}.png"), det.mask)

    idx2 = pick_best(detections, cfg.hole_class_id, ellipses)
    pair_hole = ellipses[idx2].ellipse[0] if idx2 >= 0 else None
    pair_hole_sigma = ellipse_observation_sigma(ellipses[idx2]) if idx2 >= 0 else 10.0
    idx0, idx1 = select_ring_pair(
        detections, ellipses, cfg.outer_class_id, cfg.middle_class_id,
        estimator, pair_hole, pair_hole_sigma)
    # C++ 图片批处理每张图都会 Reset；只有视频/摄像头才跨帧使用椭圆 EMA。
    if cfg.source_mode.lower() != "images":
        if idx0 >= 0:
            ellipses[idx0] = ellipse_smoother.update(cfg.outer_class_id, ellipses[idx0])
        if idx1 >= 0:
            ellipses[idx1] = ellipse_smoother.update(cfg.middle_class_id, ellipses[idx1])
        if idx2 >= 0:
            ellipses[idx2] = ellipse_smoother.update(cfg.hole_class_id, ellipses[idx2])
    target_idx = -1
    use_middle = False
    pose_auto = pose_fixed = None
    selected_reference_center: Optional[List[float]] = None
    selected_hole_center: Optional[List[float]] = None
    if idx2 >= 0 and (idx0 >= 0 or idx1 >= 0):
        if idx0 >= 0 and idx1 >= 0:
            score0 = ellipse_selection_score(ellipses[idx0], detections[idx0].score)
            score1 = ellipse_selection_score(ellipses[idx1], detections[idx1].score)
            target_idx = idx0 if score0 >= score1 else idx1
        else:
            target_idx = idx0 if idx0 >= 0 else idx1
        use_middle = detections[target_idx].class_id == cfg.middle_class_id
        hole_center = ellipses[idx2].ellipse[0]
        target = ellipses[target_idx].ellipse
        selected_reference_center = [float(target[0][0]), float(target[0][1])]
        selected_hole_center = [float(hole_center[0]), float(hole_center[1])]
        outer_observation = ellipses[idx0] if idx0 >= 0 else None
        middle_observation = ellipses[idx1] if idx1 >= 0 else None
        hole_sigma = ellipse_observation_sigma(ellipses[idx2])
        if cfg.compute_both_pose_modes or not cfg.pose_display_fixed:
            pose_auto = estimator.solve_dual(
                outer_observation, middle_observation, hole_center, hole_sigma)
        if cfg.compute_both_pose_modes or cfg.pose_display_fixed:
            pose_fixed = estimator.solve_dual(
                outer_observation, middle_observation, hole_center, hole_sigma,
                cfg.pose_fixed_distance_mm)
        pose_auto = valid_pose(pose_auto)
        pose_fixed = valid_pose(pose_fixed)
        cv2.ellipse(vis, target, (255, 255, 0), 4)
        cv2.ellipse(vis, ellipses[idx2].ellipse, (255, 255, 0), 4)
        cv2.circle(vis, tuple(np.rint(hole_center).astype(int)), 6, (255, 255, 0), -1)
        cv2.line(vis, tuple(np.rint(target[0]).astype(int)), tuple(np.rint(hole_center).astype(int)),
                 (255, 255, 0), 3)
        draw_text(vis, f"Ref({'Mid' if use_middle else 'Out'}): ({target[0][0]:.0f}, {target[0][1]:.0f})",
                  100, (255, 255, 255), 0.8)
        draw_text(vis, f"Hole: ({hole_center[0]:.0f}, {hole_center[1]:.0f})", 140, (255, 255, 255), 0.8)

    raw = pose_fixed if cfg.pose_display_fixed else pose_auto
    smooth, stale = smoother.update(raw)
    display_pose = smooth if cfg.enable_pose_smoothing else raw
    if display_pose is not None and (not stale or cfg.draw_held_pose):
        color = (0, 255, 255) if cfg.pose_display_fixed else ((160, 160, 160) if stale else (0, 255, 0))
        suffix = "Fixed" if cfg.pose_display_fixed else ("Held" if stale else "Auto")
        draw_text(vis, f"Yaw:{display_pose.yaw_deg:.1f} Pit:{display_pose.pitch_deg:.1f}", 190, color, 1.3)
        draw_text(vis, f"Dist: {display_pose.tz_mm / 1000.0:.2f}m ({suffix})", 240, color, 1.3)
        if cfg.draw_pose_axis and not stale:
            estimator.draw_axis(vis, display_pose, use_middle)
    else:
        draw_text(vis, "Pose: --", 190, (255, 255, 0), 1.2)
        draw_text(vis, "Dist: --", 240, (255, 255, 0), 1.2)

    det_details = []
    def finite_or_none(value: float) -> Optional[float]:
        return float(value) if math.isfinite(value) else None

    for det, ellipse in zip(detections, ellipses):
        det_details.append({
            "x": det.x, "y": det.y, "w": det.w, "h": det.h,
            "score": det.score, "class_id": det.class_id,
            "class_name": cfg.class_names[det.class_id] if 0 <= det.class_id < len(cfg.class_names) else str(det.class_id),
            "ellipse": {"center": [float(x) for x in ellipse.ellipse[0]],
                        "size": [float(x) for x in ellipse.ellipse[1]],
                        "angle_deg": float(ellipse.ellipse[2]), "source": ellipse.source,
                        "valid": ellipse.valid, "from_mask": ellipse.from_mask,
                        "quality": ellipse.quality,
                        "inlier_ratio": ellipse.inlier_ratio, "inliers": ellipse.inliers,
                        "mean_error_px": (finite_or_none(ellipse.mean_error_px)
                                          if ellipse.mean_error_px is not None else None),
                        "geometry_consistent": ellipse.geometry_consistent,
                        "temporally_filtered": ellipse.temporally_filtered,
                        "border_truncated": ellipse.border_truncated,
                        "partial_visibility": ellipse.partial_visibility,
                        "visible_arc_ratio": ellipse.visible_arc_ratio,
                        "removed_border_points": ellipse.removed_border_points,
                        "support_points": int(len(ellipse.visible_arc_points)),
                        "angular_coverage_deg": ellipse.angular_coverage_deg,
                        "occupied_quadrants": ellipse.occupied_quadrants,
                        "uncertainty_valid": ellipse.uncertainty_valid,
                        "center_std_px": finite_or_none(ellipse.center_std_px),
                        "major_axis_std_px": finite_or_none(ellipse.major_axis_std_px),
                        "minor_axis_std_px": finite_or_none(ellipse.minor_axis_std_px),
                        "angle_std_deg": finite_or_none(ellipse.angle_std_deg),
                        "covariance_condition": finite_or_none(ellipse.covariance_condition),
                        "pose_sigma_px": ellipse_observation_sigma(ellipse),
                        "conic": [[float(value) for value in row]
                                  for row in ellipse.conic]},
        })
    return vis, {
        "frame_id": frame_id, "detections": det_details,
        "selected_reference_class": detections[target_idx].class_id if target_idx >= 0 else None,
        "selected_reference_center": selected_reference_center,
        "selected_hole_center": selected_hole_center,
        "pose_auto": pose_dict(pose_auto), "pose_fixed": pose_dict(pose_fixed),
        "pose_selected_raw": pose_dict(raw), "pose_smoothed": pose_dict(smooth),
        "pose_is_stale": stale,
    }


ELLIPSE_TXT_HEADER = (
    "# class_id center_x_px center_y_px major_axis_px minor_axis_px angle_deg "
    "confidence fit_source valid quality inlier_ratio inliers mean_error_px coverage_deg "
    "quadrants border_truncated partial visible_arc_ratio removed_border_points support_points "
    "center_std_px major_std_px minor_std_px angle_std_deg cov_condition "
    "geometry_ok temporal conic00 conic01 conic02 conic11 conic12 conic22\n"
)
POSE_FRAME_HEADER = (
    "# valid reference_class ref_center_x_px ref_center_y_px hole_center_x_px "
    "hole_center_y_px auto_yaw_deg auto_pitch_deg auto_roll_deg auto_tx_mm auto_ty_mm "
    "auto_tz_mm fixed_yaw_deg fixed_pitch_deg fixed_roll_deg fixed_tx_mm fixed_ty_mm "
    "fixed_tz_mm display_mode\n"
)


def write_batch_frame_outputs(key: str, frame: np.ndarray,
                              detections: Sequence[Detection],
                              detail: Dict[str, Any], cfg: Config,
                              labels_dir: Path, ellipses_dir: Path,
                              poses_dir: Path) -> None:
    """生成与 batch_image_video.cpp 字段和目录一致的三类逐帧 TXT。"""
    height, width = frame.shape[:2]
    with (labels_dir / f"{key}.txt").open("w", encoding="utf-8") as handle:
        for det in detections:
            handle.write(
                f"{det.class_id} {(det.x + det.w * 0.5) / width:.8f} "
                f"{(det.y + det.h * 0.5) / height:.8f} {det.w / width:.8f} "
                f"{det.h / height:.8f} {det.score:.8f}\n")

    def number(value: Any) -> float:
        return float(value) if value is not None else math.inf

    with (ellipses_dir / f"{key}.txt").open("w", encoding="utf-8") as handle:
        handle.write(ELLIPSE_TXT_HEADER)
        for det, item in zip(detections, detail["detections"]):
            ellipse = item["ellipse"]
            size = ellipse["size"]
            conic = ellipse["conic"]
            geometry_fields = [
                ellipse["center"][0], ellipse["center"][1], max(size), min(size),
                ellipse["angle_deg"], det.score,
            ]
            tail_fields = [
                number(ellipse["mean_error_px"]),
                ellipse["angular_coverage_deg"], ellipse["occupied_quadrants"],
                int(ellipse["border_truncated"]), int(ellipse["partial_visibility"]),
                ellipse["visible_arc_ratio"], ellipse["removed_border_points"],
                ellipse["support_points"],
                number(ellipse["center_std_px"]), number(ellipse["major_axis_std_px"]),
                number(ellipse["minor_axis_std_px"]), number(ellipse["angle_std_deg"]),
                number(ellipse["covariance_condition"]),
                int(ellipse["geometry_consistent"]), int(ellipse["temporally_filtered"]),
                conic[0][0], conic[0][1], conic[0][2],
                conic[1][1], conic[1][2], conic[2][2],
            ]
            handle.write(
                f"{det.class_id} " +
                " ".join(f"{float(value):.6f}" for value in geometry_fields) +
                f" {ellipse['source']} {int(ellipse['valid'])} "
                f"{float(ellipse['quality']):.6f} "
                f"{float(ellipse['inlier_ratio']):.6f} {int(ellipse['inliers'])} " +
                " ".join(f"{float(value):.6f}" for value in tail_fields) + "\n")

    with (poses_dir / f"{key}.txt").open("w", encoding="utf-8") as handle:
        handle.write(POSE_FRAME_HEADER)
        automatic, fixed = detail["pose_auto"], detail["pose_fixed"]
        center, hole = detail["selected_reference_center"], detail["selected_hole_center"]
        selected = fixed if cfg.pose_display_fixed else automatic
        valid = selected is not None and center is not None and hole is not None
        if not valid:
            handle.write("0 -1 " + " ".join(["nan"] * 16) + " invalid\n")
            return

        def pose_values(pose: Dict[str, float]) -> List[float]:
            return [pose[name] for name in
                    ("yaw_deg", "pitch_deg", "roll_deg", "tx_mm", "ty_mm", "tz_mm")]

        missing_pose = [math.nan] * 6
        values: List[Any] = [
            1, detail["selected_reference_class"], center[0], center[1], hole[0], hole[1],
            *(pose_values(automatic) if automatic is not None else missing_pose),
            *(pose_values(fixed) if fixed is not None else missing_pose),
            "fixed" if cfg.pose_display_fixed else "auto",
        ]
        handle.write(" ".join(str(value) if isinstance(value, str)
                              else f"{float(value):.9f}" for value in values) + "\n")


class FrameSource:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.capture: Optional[cv2.VideoCapture] = None
        self.images: List[Path] = []
        self.fps = 0.0
        mode = cfg.source_mode.lower()
        if mode == "images":
            root = configured_input_path(cfg)
            if not root.is_dir():
                raise FileNotFoundError(f"图片目录不存在: {root}")
            suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            self.images = sorted(p for p in root.iterdir() if p.suffix.lower() in suffixes)
            if not self.images:
                raise RuntimeError(f"图片目录中没有支持的图片: {root}")
        elif mode in ("video", "camera"):
            source: Any = cfg.camera_id if mode == "camera" else str(configured_input_path(cfg))
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                raise RuntimeError(f"无法打开输入: {source}")
            if mode == "camera":
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
                if cfg.camera_mjpg:
                    self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            else:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        else:
            raise ValueError("source_mode 只能是 video、camera 或 images")

    def __iter__(self) -> Iterable[Tuple[int, str, str, np.ndarray]]:
        """返回 frame_id、与 C++ 相同的输出 key、图片扩展名和图像。"""
        cfg = self.cfg
        if cfg.source_mode.lower() == "images":
            for frame_id, path in enumerate(self.images):
                if frame_id < cfg.start_frame or (cfg.end_frame >= 0 and frame_id > cfg.end_frame):
                    continue
                if (frame_id - cfg.start_frame) % max(1, cfg.frame_step):
                    continue
                image = cv2.imread(str(path))
                if image is not None:
                    yield frame_id, path.stem, path.suffix, image
            return
        assert self.capture is not None
        frame_id = cfg.start_frame if cfg.source_mode.lower() == "video" else 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            current = frame_id
            frame_id += 1
            if cfg.end_frame >= 0 and current > cfg.end_frame:
                break
            if (current - cfg.start_frame) % max(1, cfg.frame_step) == 0:
                if cfg.source_mode.lower() == "camera":
                    key = f"camera_{cfg.camera_id}_frame_{current:08d}"
                else:
                    key = f"{configured_input_path(cfg).stem}_frame_{current:08d}"
                yield current, key, ".jpg", frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


POSE_COLUMNS = ["frame_id", "timestamp_s", "pose_valid", "pose_stale", "mode", "ref_class",
                "raw_yaw_deg", "raw_pitch_deg", "raw_roll_deg", "raw_tx_mm", "raw_ty_mm", "raw_tz_mm",
                "smooth_yaw_deg", "smooth_pitch_deg", "smooth_roll_deg", "smooth_tx_mm", "smooth_ty_mm", "smooth_tz_mm"]


def pose_txt_line(detail: Dict[str, Any], timestamp: float, cfg: Config) -> str:
    def values(pose: Optional[Dict[str, float]]) -> List[str]:
        if pose is None:
            return ["nan"] * 6
        return [f"{pose[k]:.9f}" for k in ("yaw_deg", "pitch_deg", "roll_deg", "tx_mm", "ty_mm", "tz_mm")]
    raw, smooth = detail["pose_selected_raw"], detail["pose_smoothed"]
    fields = [str(detail["frame_id"]), f"{timestamp:.9f}", str(int(raw is not None)),
              str(int(detail["pose_is_stale"])), "fixed" if cfg.pose_display_fixed else "auto",
              str(detail["selected_reference_class"] if detail["selected_reference_class"] is not None else -1)]
    return "\t".join(fields + values(raw) + values(smooth)) + "\n"


def save_pose_plots(history: List[Tuple[int, Pose6D]], out_dir: Path) -> None:
    if not history:
        return
    width, height = 1400, 800
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    frames = np.asarray([x[0] for x in history], dtype=np.float64)
    series = [("Yaw deg", np.asarray([x[1].yaw_deg for x in history]), (0, 0, 220)),
              ("Pitch deg", np.asarray([x[1].pitch_deg for x in history]), (0, 150, 0)),
              ("Distance m", np.asarray([x[1].tz_mm / 1000.0 for x in history]), (220, 0, 0))]
    margin, gap = 80, 30
    plot_h = (height - 2 * margin - 2 * gap) // 3
    for row, (label, values, color) in enumerate(series):
        top = margin + row * (plot_h + gap)
        left, right = margin, width - margin
        cv2.rectangle(canvas, (left, top), (right, top + plot_h), (170, 170, 170), 1)
        vmin, vmax = float(np.min(values)), float(np.max(values))
        if abs(vmax - vmin) < 1e-9:
            vmax, vmin = vmax + 1.0, vmin - 1.0
        xs = left + (frames - frames.min()) / max(frames.max() - frames.min(), 1.0) * (right - left)
        ys = top + plot_h - (values - vmin) / (vmax - vmin) * plot_h
        pts = np.rint(np.stack((xs, ys), axis=1)).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{label}  [{vmin:.3f}, {vmax:.3f}]", (left, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.imwrite(str(out_dir / "pose_curves.png"), canvas)


def parse_args(cfg: Config) -> Config:
    parser = argparse.ArgumentParser(
        description="PT 服务器验证版：处理流程和输出效果对齐 batch_image_video.cpp")
    parser.add_argument("--model", help="覆盖配置区 model_path")
    parser.add_argument("--video", help="使用指定视频")
    parser.add_argument("--images", help="使用指定图片目录")
    parser.add_argument("--cam", type=int, help="使用指定摄像头")
    parser.add_argument("--input-path", help="兼容 C++：图片目录或视频所在目录/完整路径")
    parser.add_argument("--video-name", help="兼容 C++：与 --input-path 拼接的视频文件名")
    parser.add_argument("--mode", choices=("auto", "images", "video", "camera"),
                        help="兼容 C++：auto/images/video；另支持 camera")
    parser.add_argument("--device", help='cpu 或 CUDA 编号，如 "0"')
    parser.add_argument("--output", "--output-path", dest="output",
                        help="输出根目录，直接创建 visual/labels/ellipses/poses")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--save-video-frames", action="store_true")
    parser.add_argument("--display-fixed", action="store_true")
    parser.add_argument("--fixed-distance", type=float)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--frame-step", type=int)
    parser.add_argument("--force-reference-box", action="store_true",
                        help="外/中参考圈强制使用检测框内切圆")
    parser.add_argument("--hole-mask", action="store_true",
                        help="尝试使用 Mask 拟合内孔（默认使用检测框）")
    parser.add_argument("--hole-box", action="store_true",
                        help="内孔使用检测框内切圆（默认）")
    args = parser.parse_args()
    if args.model:
        cfg.model_path = args.model
    if args.video:
        path = Path(args.video).expanduser()
        cfg.source_mode, cfg.input_path, cfg.video_name = "video", str(path.parent), path.name
    if args.images:
        path = Path(args.images).expanduser()
        cfg.source_mode, cfg.input_path, cfg.video_name = "images", str(path.parent), path.name
    if args.cam is not None:
        cfg.source_mode, cfg.camera_id = "camera", args.cam
    if args.input_path:
        cfg.input_path = str(Path(args.input_path).expanduser())
        cfg.video_name = args.video_name or ""
    elif args.video_name:
        cfg.video_name = args.video_name
    if args.mode:
        cfg.source_mode = args.mode
    if cfg.source_mode == "auto":
        cfg.source_mode = "images" if configured_input_path(cfg).is_dir() else "video"
    if args.device:
        cfg.device = args.device
    if args.output:
        cfg.output_path = args.output
    if args.no_show:
        cfg.show_window = False
    if args.show:
        cfg.show_window = True
    if args.no_video:
        cfg.save_video = False
    if args.force_reference_box:
        cfg.force_reference_box = True
    if args.hole_mask:
        cfg.mask_fit_hole = True
    if args.hole_box:
        cfg.mask_fit_hole = False
    if args.save_video_frames:
        cfg.save_visual_frames = True
    if args.display_fixed:
        cfg.pose_display_fixed = True
    if args.fixed_distance is not None:
        cfg.pose_fixed_distance_mm = args.fixed_distance
    if args.start_frame is not None:
        cfg.start_frame = max(0, args.start_frame)
    if args.end_frame is not None:
        cfg.end_frame = args.end_frame
    if args.frame_step is not None:
        cfg.frame_step = max(1, args.frame_step)
    if cfg.device.lower() == "cpu":
        cfg.half = False
    return cfg


def main() -> int:
    cfg = parse_args(CFG)
    out_dir = configured_output_dir(cfg)
    visual_dir = out_dir / "visual"
    labels_dir = out_dir / "labels"
    ellipses_dir = out_dir / "ellipses"
    poses_dir = out_dir / "poses"
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for directory in (visual_dir, labels_dir, ellipses_dir, poses_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if cfg.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        saved_config = asdict(cfg)
        saved_config["resolved_input_path"] = str(configured_input_path(cfg))
        saved_config["resolved_output_dir"] = str(out_dir)
        saved_config["resolved_output_video_name"] = configured_output_video_name(cfg)
        json.dump(saved_config, handle, ensure_ascii=False, indent=2)

    source: Optional[FrameSource] = None
    writer: Optional[cv2.VideoWriter] = None
    history: List[Tuple[int, Pose6D]] = []
    try:
        source = FrameSource(cfg)
        model = PTModel(cfg)
        estimator, smoother = PoseEstimatorLM(cfg), PoseSmoother(cfg)
        ellipse_smoother = EllipseSmoother(cfg)
        source_fps = source.fps if source.fps > 1e-3 else 25.0
        fps_out = cfg.output_fps if cfg.output_fps > 0 else source_fps / max(1, cfg.frame_step)
        if cfg.show_window:
            cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cfg.window_name, cfg.window_width, cfg.window_height)
        last_time, started = time.perf_counter(), time.perf_counter()
        ema_fps, processed = 0.0, 0
        with (out_dir / "pose.txt").open("w", encoding="utf-8", buffering=1) as pose_file, \
             (out_dir / "detections.jsonl").open("w", encoding="utf-8", buffering=1) as json_file:
            pose_file.write("\t".join(POSE_COLUMNS) + "\n")
            for frame_id, output_key, image_extension, frame in source:
                detections = model.infer(frame)
                vis, detail = process_visual(frame, detections, cfg, estimator, smoother,
                                             ellipse_smoother, frame_id, masks_dir)
                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_time, 1e-9)
                last_time = now
                ema_fps = instant_fps if ema_fps <= 0 else 0.9 * ema_fps + 0.1 * instant_fps
                if cfg.draw_fps:
                    draw_text(vis, f"FPS: {ema_fps:.1f}", 40, (255, 255, 255), 1.0)
                timestamp = (now - started) if cfg.source_mode == "camera" else frame_id / source_fps
                detail.update(timestamp_s=timestamp, processing_fps=ema_fps)
                pose_file.write(pose_txt_line(detail, timestamp, cfg))
                json_file.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
                write_batch_frame_outputs(output_key, frame, detections, detail, cfg,
                                          labels_dir, ellipses_dir, poses_dir)
                if detail["pose_smoothed"] is not None and not detail["pose_is_stale"]:
                    history.append((frame_id, Pose6D(**detail["pose_smoothed"])))

                if (cfg.source_mode.lower() in ("video", "camera") and
                        cfg.save_video and writer is None):
                    writer = cv2.VideoWriter(str(out_dir / configured_output_video_name(cfg)),
                                             cv2.VideoWriter_fourcc(*cfg.output_fourcc), fps_out,
                                             (vis.shape[1], vis.shape[0]))
                    if not writer.isOpened():
                        # 与 C++ 一致：服务器缺少 MP4 编码器时自动回退 MJPG/AVI。
                        writer.release()
                        fallback_name = f"{configured_input_path(cfg).stem}_result.avi"
                        writer = cv2.VideoWriter(
                            str(out_dir / fallback_name),
                            cv2.VideoWriter_fourcc(*"MJPG"), fps_out,
                            (vis.shape[1], vis.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("无法创建 MP4 或 MJPG/AVI 输出视频")
                if writer is not None:
                    writer.write(vis)
                save_visual = (cfg.source_mode.lower() == "images" or
                               (cfg.save_visual_frames and
                                frame_id % max(1, cfg.visual_frame_every) == 0))
                if save_visual:
                    suffix = image_extension if cfg.source_mode.lower() == "images" else ".jpg"
                    output_image = visual_dir / f"{output_key}{suffix}"
                    parameters = ([cv2.IMWRITE_JPEG_QUALITY, 95]
                                  if suffix.lower() in (".jpg", ".jpeg") else [])
                    cv2.imwrite(str(output_image), vis, parameters)
                processed += 1
                print(f"\rframe={frame_id} det={len(detections)} fps={ema_fps:.1f}", end="", flush=True)
                if cfg.show_window:
                    cv2.imshow(cfg.window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("s"):
                        cv2.imwrite(str(out_dir / f"manual_frame_{frame_id:08d}.jpg"), vis,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        save_pose_plots(history, out_dir)
        print(f"\n完成：{processed} 帧，全部结果在 {out_dir.resolve()}")
        return 0
    except KeyboardInterrupt:
        print("\n用户中止，已保留此前结果。")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.release()
        if source is not None:
            source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
