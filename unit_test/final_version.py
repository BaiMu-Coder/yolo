#!/usr/bin/env python3
"""final_version.cpp 的效果优先 Python/PT 版本。

依赖：ultralytics、torch、opencv-python、numpy。
运行：../.venv-yolo/bin/python final_version.py
常用覆盖：--model xxx.pt --video xxx.mp4 --no-show
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
    ellipse_inlier_px: float = 3.0
    ellipse_min_inlier_ratio: float = 0.42
    ellipse_max_axis_ratio: float = 6.0
    ellipse_max_points: int = 180
    ellipse_min_quality: float = 0.52
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
    pose_max_iters: int = 30
    signed_robust_residual: bool = False  # False 精确复现 C++；True 可试验改进

    # 抗抖：原始值和平滑值都会保存，画面默认显示平滑值
    enable_pose_smoothing: bool = True
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
    save_visual_frames: bool = True
    visual_frame_every: int = 1
    save_masks: bool = True
    mask_every: int = 1
    mask_alpha: float = 0.50
    draw_boxes: bool = True
    draw_masks: bool = True
    draw_all_ellipses: bool = True
    draw_pose_axis: bool = True
    draw_fps: bool = True


CFG = Config()
# =============================================================================


def configured_input_path(cfg: Config) -> Path:
    """由唯一数据根目录和视频/目录名拼出输入路径。"""
    root = Path(cfg.input_path).expanduser()
    return root / cfg.video_name if cfg.video_name else root


def configured_output_dir(cfg: Config) -> Path:
    """默认把同一输入的所有结果归档到以输入名称命名的文件夹。"""
    if cfg.source_mode.lower() == "camera":
        stem = f"camera_{cfg.camera_id}"
    else:
        stem = configured_input_path(cfg).stem
    return Path(cfg.output_path).expanduser() / stem


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


@dataclass
class EllipseResult:
    ellipse: Ellipse
    source: str = "box"
    inliers: int = 0
    mean_error_px: Optional[float] = None
    sampled_points: int = 0
    inlier_ratio: float = 0.0
    center_deviation_ratio: float = 0.0
    quality: float = 0.0
    geometry_consistent: bool = True
    temporally_filtered: bool = False

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
            if masks is not None and i < len(masks):
                mask = (masks[i] >= cfg.mask_binary_threshold).astype(np.uint8) * 255
                # 与 C++ 一致，只保留检测框内的实例 mask。
                clipped = np.zeros_like(mask)
                clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                mask = clipped
            detections.append(Detection(x1, y1, x2 - x1, y2 - y1,
                                        float(scores[i]), int(classes[i]), mask))
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


def ransac_ellipse(points: np.ndarray, cfg: Config) -> Optional[EllipseResult]:
    if len(points) < 20:
        return None
    rng = np.random.default_rng(12345)
    best: Optional[EllipseResult] = None
    best_inlier_mask: Optional[np.ndarray] = None
    for _ in range(cfg.ellipse_ransac_iters):
        sample = points[rng.choice(len(points), 5, replace=False)].reshape(-1, 1, 2).astype(np.float32)
        try:
            ellipse = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        a, b = ellipse[1][0] * 0.5, ellipse[1][1] * 0.5
        if min(a, b) < 2.0 or max(a, b) / max(min(a, b), 1e-3) > cfg.ellipse_max_axis_ratio:
            continue
        errors = radial_errors(ellipse, points)
        inlier_mask = errors <= cfg.ellipse_inlier_px
        count = int(np.count_nonzero(inlier_mask))
        mean = float(np.mean(errors[inlier_mask])) if count else math.inf
        if best is None or count > best.inliers or (count == best.inliers and mean < (best.mean_error_px or math.inf)):
            best = EllipseResult(ellipse=ellipse, source="mask", inliers=count,
                                 mean_error_px=mean, sampled_points=len(points),
                                 inlier_ratio=count / len(points))
            best_inlier_mask = inlier_mask
    required = max(20, int(math.ceil(cfg.ellipse_min_inlier_ratio * len(points))))
    if best is None or best.inliers < required or best_inlier_mask is None:
        return None
    refined_points = points[best_inlier_mask]
    try:
        refined = cv2.fitEllipse(refined_points.reshape(-1, 1, 2).astype(np.float32))
    except cv2.error:
        return None
    return EllipseResult(ellipse=refined, source="mask", inliers=len(refined_points),
                         mean_error_px=float(np.mean(radial_errors(refined, refined_points))),
                         sampled_points=len(points), inlier_ratio=len(refined_points) / len(points))


def _score_candidate(candidate: EllipseResult, det: Detection, cfg: Config) -> Optional[EllipseResult]:
    cx, cy = candidate.ellipse[0]
    box_center = (det.x + det.w * 0.5, det.y + det.h * 0.5)
    short_side = max(1.0, float(min(det.w, det.h)))
    candidate.center_deviation_ratio = math.hypot(cx - box_center[0], cy - box_center[1]) / short_side
    major, minor = max(candidate.ellipse[1]), min(candidate.ellipse[1])
    if candidate.center_deviation_ratio > cfg.ellipse_deviation_ratio or minor < 0.2 * short_side:
        return None
    error = candidate.mean_error_px if candidate.mean_error_px is not None else math.inf
    error_quality = math.exp(-error / max(0.5, cfg.ellipse_inlier_px))
    center_quality = max(0.0, 1.0 - candidate.center_deviation_ratio / cfg.ellipse_deviation_ratio)
    shape_quality = max(0.0, min(1.0, 1.0 - (major / max(minor, 1e-3) - 1.0) /
                                 max(1.0, cfg.ellipse_max_axis_ratio - 1.0)))
    bonus = 0.05 if candidate.source == "mask" else 0.0
    candidate.quality = min(1.0, 0.42 * candidate.inlier_ratio + 0.25 * error_quality +
                            0.20 * center_quality + 0.13 * shape_quality + bonus)
    return candidate


def _fit_points(points: np.ndarray, det: Detection, cfg: Config, source: str) -> Optional[EllipseResult]:
    if len(points) < 20:
        return None
    step = max(1, len(points) // cfg.ellipse_max_points)
    fitted = ransac_ellipse(points[::step][:cfg.ellipse_max_points], cfg)
    if fitted is None:
        return None
    (cx, cy), size, angle = fitted.ellipse
    fitted.ellipse = ((cx + det.x, cy + det.y), size, angle)
    fitted.source = source
    return _score_candidate(fitted, det, cfg)


def best_ellipse(frame: np.ndarray, det: Detection, cfg: Config,
                 force_box: bool = False, allow_edge: bool = True) -> EllipseResult:
    box_center = (det.x + det.w * 0.5, det.y + det.h * 0.5)
    side = float(min(det.w, det.h))
    fallback = EllipseResult((box_center, (side, side), 0.0), source="box", quality=0.38)
    if force_box:
        return fallback
    candidates: List[EllipseResult] = []
    if det.mask is not None and not force_box:
        roi = det.mask[det.y:det.y + det.h, det.x:det.x + det.w].copy()
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            points = np.concatenate(selected)
            if len(points) >= 20:
                fitted = _fit_points(points, det, cfg, "mask")
                if fitted is not None:
                    candidates.append(fitted)
    if allow_edge and cfg.enable_edge_fallback and (not candidates or max(x.quality for x in candidates) < 0.72):
        roi_gray = cv2.cvtColor(frame[det.y:det.y + det.h, det.x:det.x + det.w], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 1.2)
        edges = cv2.Canny(cv2.equalizeHist(roi_gray), 45, 135, L2gradient=True)
        edges[:3, :], edges[-3:, :], edges[:, :3], edges[:, -3:] = 0, 0, 0, 0
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        edge_points = [c.reshape(-1, 2) for c in contours if len(c) >= 18]
        if edge_points:
            fitted = _fit_points(np.concatenate(edge_points), det, cfg, "edge")
            if fitted is not None:
                candidates.append(fitted)
    if not candidates:
        return fallback
    best = max(candidates, key=lambda item: item.quality + (0.03 if item.source == "mask" else 0.0))
    return best if best.quality >= cfg.ellipse_min_quality else fallback


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
                  points: np.ndarray, fixed_tz: Optional[float]) -> np.ndarray:
        x = initial.astype(np.float64).copy()
        damping = 1e-3
        residual = self._residual(x, ellipse, hole, points, fixed_tz)
        cost = 0.5 * float(residual @ residual)
        for _ in range(self.cfg.pose_max_iters):
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
              fixed_distance: Optional[float] = None) -> Pose6D:
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
            out = self._optimize(np.asarray([yaw, pitch, tx, ty, tz]), ellipse, hole, points, None)
            return Pose6D(out[0], out[1], 0.0, out[2], out[3], out[4])
        out = self._optimize(np.asarray([yaw, pitch, tx, ty]), ellipse, hole, points, fixed_distance)
        return Pose6D(out[0], out[1], 0.0, out[2], out[3], fixed_distance)

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


def refine_ring_pair(outer: EllipseResult, middle: EllipseResult) -> bool:
    """双圆环联合约束；一致时融合圆心，不一致时惩罚较差候选。"""
    outer_major, outer_minor = max(outer.ellipse[1]), min(outer.ellipse[1])
    middle_major, middle_minor = max(middle.ellipse[1]), min(middle.ellipse[1])
    if min(outer_major, outer_minor, middle_major, middle_minor) <= 0:
        return False
    size_error = abs(middle_major / outer_major - 980.0 / 1200.0) / 0.22
    center_error = math.dist(outer.ellipse[0], middle.ellipse[0]) / max(1.0, outer_major * 0.14)
    axis_error = abs(outer_major / outer_minor - middle_major / middle_minor) / 0.28
    ao = (outer.ellipse[2] + (90.0 if outer.ellipse[1][0] < outer.ellipse[1][1] else 0.0)) % 180.0
    am = (middle.ellipse[2] + (90.0 if middle.ellipse[1][0] < middle.ellipse[1][1] else 0.0)) % 180.0
    angle_error = min(abs(ao - am), 180.0 - abs(ao - am)) / 24.0
    consistent = max(size_error, center_error, axis_error, angle_error) <= 1.0
    outer.geometry_consistent = middle.geometry_consistent = consistent
    if not consistent:
        if outer.quality >= middle.quality:
            middle.quality *= 0.35
        else:
            outer.quality *= 0.35
        return False
    score = max(0.0, 1.0 - (0.34 * size_error + 0.34 * center_error +
                            0.18 * axis_error + 0.14 * angle_error))
    weight_sum = max(0.05, outer.quality) + max(0.05, middle.quality)
    common = ((np.asarray(outer.ellipse[0]) * max(0.05, outer.quality) +
               np.asarray(middle.ellipse[0]) * max(0.05, middle.quality)) / weight_sum)
    for ellipse in (outer, middle):
        center = np.asarray(ellipse.ellipse[0])
        fused = center + 0.35 * (common - center)
        ellipse.ellipse = ((float(fused[0]), float(fused[1])), ellipse.ellipse[1], ellipse.ellipse[2])
        ellipse.quality = min(1.0, ellipse.quality + 0.12 * score)
    return True


def select_ring_pair(detections: Sequence[Detection], ellipses: List[EllipseResult],
                     outer_class_id: int, middle_class_id: int) -> Tuple[int, int]:
    outers = [i for i, det in enumerate(detections) if det.class_id == outer_class_id]
    middles = [i for i, det in enumerate(detections) if det.class_id == middle_class_id]
    if not outers or not middles:
        return (pick_best(detections, outer_class_id, ellipses),
                pick_best(detections, middle_class_id, ellipses))
    best_score, best_pair = -math.inf, (-1, -1)
    for outer_idx in outers:
        for middle_idx in middles:
            outer, middle = copy.deepcopy(ellipses[outer_idx]), copy.deepcopy(ellipses[middle_idx])
            consistent = refine_ring_pair(outer, middle)
            score = (0.55 * detections[outer_idx].score + 0.45 * outer.quality +
                     0.55 * detections[middle_idx].score + 0.45 * middle.quality +
                     (0.45 if consistent else -0.35))
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
        current.temporally_filtered = True
        self.values[class_id] = current
        return current


def draw_text(image: np.ndarray, text: str, y: int, color: Tuple[int, int, int],
              scale: float = 1.0, thickness: int = 2) -> None:
    cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def pick_best(detections: Sequence[Detection], class_id: int,
              ellipses: Optional[Sequence[EllipseResult]] = None) -> int:
    indices = [i for i, det in enumerate(detections) if det.class_id == class_id]
    if not indices:
        return -1
    return max(indices, key=lambda i: detections[i].score if ellipses is None else
               0.55 * detections[i].score + 0.45 * ellipses[i].quality)


def pose_dict(pose: Optional[Pose6D]) -> Optional[Dict[str, float]]:
    if pose is None or not np.all(np.isfinite(pose.array())) or pose.tz_mm < 100.0:
        return None
    return {k: float(v) for k, v in asdict(pose).items()}


def valid_pose(pose: Optional[Pose6D]) -> Optional[Pose6D]:
    """阻止单帧退化拟合产生的 NaN/Inf 污染平滑器和输出文件。"""
    if pose is None or not np.all(np.isfinite(pose.array())) or pose.tz_mm < 100.0:
        return None
    return pose


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
            name = cfg.class_names[det.class_id] if 0 <= det.class_id < len(cfg.class_names) else str(det.class_id)
            cv2.putText(vis, f"{name} {det.score:.3f}", (det.x, max(18, det.y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if cfg.draw_all_ellipses:
            color = (0, 255, 0) if ellipse.source == "mask" else ((255, 0, 255) if ellipse.source == "edge" else (0, 255, 255))
            cv2.ellipse(vis, ellipse.ellipse, color, 2)
            cv2.circle(vis, tuple(np.rint(ellipse.ellipse[0]).astype(int)), 2, (0, 0, 255), -1)
        if cfg.save_masks and det.mask is not None and frame_id % max(1, cfg.mask_every) == 0:
            cv2.imwrite(str(masks_dir / f"frame_{frame_id:08d}_det_{i:03d}_cls_{det.class_id}.png"), det.mask)

    idx0, idx1 = select_ring_pair(detections, ellipses,
                                  cfg.outer_class_id, cfg.middle_class_id)
    idx2 = pick_best(detections, cfg.hole_class_id, ellipses)
    if idx0 >= 0:
        ellipses[idx0] = ellipse_smoother.update(cfg.outer_class_id, ellipses[idx0])
    if idx1 >= 0:
        ellipses[idx1] = ellipse_smoother.update(cfg.middle_class_id, ellipses[idx1])
    if idx2 >= 0:
        ellipses[idx2] = ellipse_smoother.update(cfg.hole_class_id, ellipses[idx2])
    target_idx = -1
    use_middle = False
    pose_auto = pose_fixed = None
    if idx2 >= 0 and (idx0 >= 0 or idx1 >= 0):
        if idx0 >= 0 and idx1 >= 0:
            score0 = 0.65 * ellipses[idx0].quality + 0.35 * detections[idx0].score
            score1 = 0.65 * ellipses[idx1].quality + 0.35 * detections[idx1].score
            target_idx = idx0 if score0 >= score1 else idx1
        else:
            target_idx = idx0 if idx0 >= 0 else idx1
        use_middle = detections[target_idx].class_id == cfg.middle_class_id
        hole_center = ellipses[idx2].ellipse[0]
        target = ellipses[target_idx].ellipse
        pose_auto = estimator.solve(target, hole_center, use_middle)
        pose_fixed = estimator.solve(target, hole_center, use_middle, cfg.pose_fixed_distance_mm)
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
    for det, ellipse in zip(detections, ellipses):
        det_details.append({
            "x": det.x, "y": det.y, "w": det.w, "h": det.h,
            "score": det.score, "class_id": det.class_id,
            "class_name": cfg.class_names[det.class_id] if 0 <= det.class_id < len(cfg.class_names) else str(det.class_id),
            "ellipse": {"center": [float(x) for x in ellipse.ellipse[0]],
                        "size": [float(x) for x in ellipse.ellipse[1]],
                        "angle_deg": float(ellipse.ellipse[2]), "source": ellipse.source,
                        "from_mask": ellipse.from_mask, "quality": ellipse.quality,
                        "inlier_ratio": ellipse.inlier_ratio, "inliers": ellipse.inliers,
                        "mean_error_px": ellipse.mean_error_px,
                        "geometry_consistent": ellipse.geometry_consistent,
                        "temporally_filtered": ellipse.temporally_filtered},
        })
    return vis, {
        "frame_id": frame_id, "detections": det_details,
        "selected_reference_class": detections[target_idx].class_id if target_idx >= 0 else None,
        "pose_auto": pose_dict(pose_auto), "pose_fixed": pose_dict(pose_fixed),
        "pose_selected_raw": pose_dict(raw), "pose_smoothed": pose_dict(smooth),
        "pose_is_stale": stale,
    }


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

    def __iter__(self) -> Iterable[Tuple[int, np.ndarray]]:
        cfg = self.cfg
        if cfg.source_mode.lower() == "images":
            for frame_id, path in enumerate(self.images):
                if frame_id < cfg.start_frame or (cfg.end_frame >= 0 and frame_id > cfg.end_frame):
                    continue
                if (frame_id - cfg.start_frame) % max(1, cfg.frame_step):
                    continue
                image = cv2.imread(str(path))
                if image is not None:
                    yield frame_id, image
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
                yield current, frame

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
    parser = argparse.ArgumentParser(description="PT YOLOv8-seg + RANSAC ellipse + LM pose")
    parser.add_argument("--model", help="覆盖配置区 model_path")
    parser.add_argument("--video", help="使用指定视频")
    parser.add_argument("--images", help="使用指定图片目录")
    parser.add_argument("--cam", type=int, help="使用指定摄像头")
    parser.add_argument("--device", help='cpu 或 CUDA 编号，如 "0"')
    parser.add_argument("--output", help="覆盖输出目录")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--force-reference-box", action="store_true",
                        help="外/中参考圈强制使用检测框内切圆")
    parser.add_argument("--hole-mask", action="store_true",
                        help="尝试使用 Mask 拟合内孔（默认使用检测框）")
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
    if args.device:
        cfg.device = args.device
    if args.output:
        cfg.output_path = args.output
    if args.no_show:
        cfg.show_window = False
    if args.no_video:
        cfg.save_video = False
    if args.force_reference_box:
        cfg.force_reference_box = True
    if args.hole_mask:
        cfg.mask_fit_hole = True
    return cfg


def main() -> int:
    cfg = parse_args(CFG)
    out_dir = configured_output_dir(cfg)
    frames_dir, masks_dir = out_dir / "frames", out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_visual_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
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
            for frame_id, frame in source:
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
                if detail["pose_smoothed"] is not None and not detail["pose_is_stale"]:
                    history.append((frame_id, Pose6D(**detail["pose_smoothed"])))

                if cfg.save_video and writer is None:
                    writer = cv2.VideoWriter(str(out_dir / configured_output_video_name(cfg)),
                                             cv2.VideoWriter_fourcc(*cfg.output_fourcc), fps_out,
                                             (vis.shape[1], vis.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("输出视频创建失败，可尝试 fourcc='MJPG' 且文件名改为 .avi")
                if writer is not None:
                    writer.write(vis)
                if cfg.save_visual_frames and frame_id % max(1, cfg.visual_frame_every) == 0:
                    cv2.imwrite(str(frames_dir / f"frame_{frame_id:08d}.jpg"), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
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
