// 串行批量性能分析：读取 -> 预处理 -> RKNN -> 后处理 -> 椭圆 -> 位姿。
// 可视化只用于逐图屏幕检查，不计时、不保存；磁盘只写耗时 CSV 和曲线图。

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"
#include "ellipse_fitter.hpp"
#include "image_process.hpp"
#include "pose_estimator_lm.hpp"
#include "yolov8seg.hpp"

namespace fs = std::filesystem;

// ============================== 公共数据结构 ==============================

struct AppConfig {
    fs::path model_path;
    fs::path input_path;
    fs::path output_path;
    bool show = true;
    bool recursive = false;
    bool display_fixed_pose = false;
    bool force_reference_box = false;
    bool fit_hole_from_mask = false;
    double fixed_distance_mm = 3000.0;
    float ellipse_center_deviation_ratio = 0.30f;
};

using Clock = std::chrono::steady_clock;

enum class TimingStage : size_t {
    Read = 0,
    Preprocess,
    RknnInference,
    Postprocess,
    EllipseFit,
    PoseSolve,
    Count,
};

constexpr size_t kTimingStageCount =
    static_cast<size_t>(TimingStage::Count);

const std::array<const char*, kTimingStageCount> kTimingStageKeys = {
    "read", "preprocess", "rknn_inference", "postprocess",
    "ellipse_fit", "pose_solve"};

const std::array<const char*, kTimingStageCount> kTimingStageTitles = {
    "Image Read", "Preprocess + Input", "RKNN Inference",
    "Output + Postprocess", "Ellipse Fitting", "Pose Solving"};

struct FrameTiming {
    fs::path image;
    std::string status = "ok";
    std::array<double, kTimingStageCount> milliseconds{};
};

struct InferenceTiming {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
};

static double elapsed_ms(Clock::time_point begin,
                         Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct PoseResult {
    bool valid = false;
    bool use_middle_ring = false;
    int reference_class = -1;
    cv::Point2f reference_center;
    cv::Point2f hole_center;
    Pose6D automatic;
    Pose6D fixed;
};

struct FrameContext {
    cv::Mat image;
    object_detect_result_list detections{};
    std::vector<EllipseFitResult> ellipses;
    PoseResult pose;
    cv::Mat visualization;
};

// ============================== 1. 图像读取模块 ==============================

class ImageLoader {
public:
    bool run(const fs::path& path, FrameContext& frame) const {
        frame.image = cv::imread(path.string(), cv::IMREAD_COLOR);
        return !frame.image.empty();
    }
};

// ============================== 2. RKNN推理模块 ==============================

class RknnInference {
public:
    explicit RknnInference(const fs::path& model_path) : model_(model_path.string()) {
        const int status = model_.init();
        if (status != RKNN_SUCC) {
            throw std::runtime_error("RKNN model init failed: " + std::to_string(status));
        }
    }

    bool run(FrameContext& frame, InferenceTiming& timing) {
        const auto preprocess_begin = Clock::now();
        cv::Mat inference_image = frame.image.clone();
        image_process preprocessor(inference_image);
        const cv::Size model_input = model_.model_input_size();
        if (preprocessor.image_preprocessing(model_input.width,
                                             model_input.height) != 0)
            return false;

        int input_bytes = 0;
        uint8_t* input = preprocessor.get_image_buffer(&input_bytes);
        if (input == nullptr) return false;
        if (model_.set_input_data(input, input_bytes) != RKNN_SUCC) return false;
        timing.preprocess_ms = elapsed_ms(preprocess_begin);

        const auto inference_begin = Clock::now();
        if (model_.rknn_model_inference() != RKNN_SUCC) return false;
        timing.inference_ms = elapsed_ms(inference_begin);

        const auto postprocess_begin = Clock::now();
        if (model_.get_output_data() != RKNN_SUCC) return false;
        letterbox transform = preprocessor.get_letterbox();
        const int status = model_.post_process(frame.detections, transform);
        model_.release_output_data();
        timing.postprocess_ms = elapsed_ms(postprocess_begin);
        return status == RKNN_SUCC;
    }

private:
    yolov8seg model_;
};

// ============================== 3. 椭圆拟合模块 ==============================

class EllipseStage {
public:
    explicit EllipseStage(const AppConfig& config)
        : force_reference_box_(config.force_reference_box),
          fit_hole_from_mask_(config.fit_hole_from_mask) {
        EllipseFitConfig fit_config;
        fit_config.center_deviation_ratio = config.ellipse_center_deviation_ratio;
        fitter_ = EllipseFitter(fit_config);
    }

    void run(FrameContext& frame) const {
        frame.ellipses.clear();
        frame.ellipses.reserve(frame.detections.count);
        const auto& masks = frame.detections.results_mask[0].each_of_mask;
        const auto& probabilities = frame.detections.results_mask[0].each_of_mask_probability;
        for (int i = 0; i < frame.detections.count; ++i) {
            const auto& detection = frame.detections.results_box[i];
            const uint8_t* mask = (i < static_cast<int>(masks.size()) && masks[i])
                                      ? masks[i].get()
                                      : nullptr;
            const uint8_t* probability =
                (i < static_cast<int>(probabilities.size()) && probabilities[i])
                    ? probabilities[i].get() : nullptr;
            // 拟合模式与在线推理一致：
            // 外/中参考圈只走“Mask -> Box”，不尝试容易受反光影响的灰度 Edge；
            // 内孔默认 Box，--hole-mask 才允许“Mask -> Edge -> Box”；
            // --force-reference-box 的优先级最高，强制外/中圈使用框内切圆。
            const bool is_reference = detection.cls_id == 0 || detection.cls_id == 1;
            const bool force_box = (is_reference && force_reference_box_) ||
                                   (detection.cls_id == 2 && !fit_hole_from_mask_);
            const cv::Rect box(detection.x, detection.y, detection.w, detection.h);
            const EllipseFitMode fit_mode = force_box
                                                ? EllipseFitMode::ForceBox
                                                : (is_reference
                                                       ? EllipseFitMode::PreferMaskNoEdge
                                                       : EllipseFitMode::PreferMask);
            frame.ellipses.push_back(fitter_.Fit(frame.image, box, mask, fit_mode,
                                                 probability));
        }
    }

private:
    EllipseFitter fitter_;
    bool force_reference_box_ = false;
    bool fit_hole_from_mask_ = false;
};

// ============================== 4. 位姿解算模块 ==============================

class PoseSolver {
public:
    explicit PoseSolver(double fixed_distance_mm) : fixed_distance_mm_(fixed_distance_mm) {
        cv::Mat K = (cv::Mat_<double>(3, 3) <<
            1639.6, 0, 960,
            0, 2165.4, 540,
            0, 0, 1);
        cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
        const double scale = 20.0 / 45.5;
        DrogueModel model;
        model.radius_cls0_mm = 1200.0 * scale;
        model.radius_cls1_mm = 980.0 * scale;
        model.radius_hole_mm = 120.0 * scale;
        model.length_L_mm = 920.0 * scale;
        estimator_.Reset(K, D, model);
    }

    void run(FrameContext& frame) {
        std::vector<int> class_ids(frame.detections.count);
        std::vector<float> detection_confidences(frame.detections.count);
        for (int i = 0; i < frame.detections.count; ++i) {
            class_ids[i] = frame.detections.results_box[i].cls_id;
            detection_confidences[i] = frame.detections.results_box[i].prop;
        }
        const int hole = best_detection(frame.detections, frame.ellipses, 2);
        int outer_count = 0, middle_count = 0;
        for (int i = 0; i < frame.detections.count; ++i) {
            if (!frame.ellipses[i].valid) continue;
            outer_count += class_ids[i] == 0;
            middle_count += class_ids[i] == 1;
        }
        std::function<float(int, int)> pose_pair_score;
        if (hole >= 0 && outer_count * middle_count > 1 &&
            outer_count * middle_count <= 4) {
            pose_pair_score = [&](int outer_index, int middle_index) {
                const auto& outer_fit = frame.ellipses[outer_index];
                const auto& middle_fit = frame.ellipses[middle_index];
                const PoseEllipseObservation outer_obs{
                    outer_fit.ellipse, EllipseObservationSigmaPx(outer_fit), true,
                    outer_fit.visible_arc_points, outer_fit.partial_visibility,
                    outer_fit.visible_arc_ratio};
                const PoseEllipseObservation middle_obs{
                    middle_fit.ellipse, EllipseObservationSigmaPx(middle_fit), true,
                    middle_fit.visible_arc_points, middle_fit.partial_visibility,
                    middle_fit.visible_arc_ratio};
                const float reprojection = static_cast<float>(
                    estimator_.EvaluateDualReprojectionScore(
                    outer_obs, middle_obs, frame.ellipses[hole].ellipse.center,
                    EllipseObservationSigmaPx(frame.ellipses[hole])));
                return reprojection * std::sqrt(
                    std::max(0.0f, outer_fit.quality * middle_fit.quality));
            };
        }
        const RingPairSelection pair = ring_refiner_.SelectAndRefine(
            class_ids, detection_confidences, frame.ellipses, 0, 1,
            pose_pair_score);
        const int outer = pair.outer_index;
        const int middle = pair.middle_index;
        frame.pose = {};
        if (hole < 0 || (outer < 0 && middle < 0) ||
            frame.ellipses.size() < static_cast<size_t>(frame.detections.count)) return;

        int reference = -1;
        if (outer >= 0 && middle >= 0) {
            const float outer_score = EllipseSelectionScore(
                frame.ellipses[outer], frame.detections.results_box[outer].prop);
            const float middle_score = EllipseSelectionScore(
                frame.ellipses[middle], frame.detections.results_box[middle].prop);
            reference = outer_score >= middle_score ? outer : middle;
        } else {
            reference = outer >= 0 ? outer : middle;
        }

        frame.pose.valid = true;
        frame.pose.reference_class = frame.detections.results_box[reference].cls_id;
        frame.pose.use_middle_ring = frame.pose.reference_class == 1;
        frame.pose.reference_center = frame.ellipses[reference].ellipse.center;
        frame.pose.hole_center = frame.ellipses[hole].ellipse.center;
        std::optional<PoseEllipseObservation> outer_observation;
        std::optional<PoseEllipseObservation> middle_observation;
        if (outer >= 0)
            outer_observation = PoseEllipseObservation{frame.ellipses[outer].ellipse,
                EllipseObservationSigmaPx(frame.ellipses[outer]), true,
                frame.ellipses[outer].visible_arc_points,
                frame.ellipses[outer].partial_visibility,
                frame.ellipses[outer].visible_arc_ratio};
        if (middle >= 0)
            middle_observation = PoseEllipseObservation{frame.ellipses[middle].ellipse,
                EllipseObservationSigmaPx(frame.ellipses[middle]), true,
                frame.ellipses[middle].visible_arc_points,
                frame.ellipses[middle].partial_visibility,
                frame.ellipses[middle].visible_arc_ratio};
        const double hole_sigma = EllipseObservationSigmaPx(frame.ellipses[hole]);
        frame.pose.automatic = estimator_.SolveDual(
            outer_observation, middle_observation, frame.pose.hole_center,
            hole_sigma, std::nullopt);
        frame.pose.fixed = estimator_.SolveDual(
            outer_observation, middle_observation, frame.pose.hole_center,
            hole_sigma, fixed_distance_mm_);
    }

    void draw_axis(cv::Mat& image, const Pose6D& pose, bool use_middle_ring) const {
        estimator_.DrawAxis(image, pose, use_middle_ring);
    }

private:
    static int best_detection(const object_detect_result_list& detections,
                              const std::vector<EllipseFitResult>& ellipses,
                              int class_id) {
        int best = -1;
        float best_score = -1.0f;
        for (int i = 0; i < detections.count; ++i) {
            const auto& detection = detections.results_box[i];
            if (i >= static_cast<int>(ellipses.size()) || !ellipses[i].valid) continue;
            const float score = i < static_cast<int>(ellipses.size())
                                    ? EllipseSelectionScore(ellipses[i], detection.prop)
                                    : detection.prop;
            if (detection.cls_id == class_id && score > best_score) {
                best = i;
                best_score = score;
            }
        }
        return best;
    }

    double fixed_distance_mm_;
    PoseEstimatorLM estimator_;
    RingPairRefiner ring_refiner_;
};

// ============================== 5. 可视化模块 ==============================

class Visualizer {
public:
    explicit Visualizer(bool display_fixed_pose) : display_fixed_pose_(display_fixed_pose) {}

    void run(FrameContext& frame, const PoseSolver& pose_solver) const {
        frame.visualization = frame.image.clone();
        const cv::Scalar colors[3] = {{0, 0, 255}, {0, 255, 0}, {255, 255, 0}};
        const auto& masks = frame.detections.results_mask[0].each_of_mask;

        for (int i = 0; i < frame.detections.count; ++i) {
            const auto& detection = frame.detections.results_box[i];
            if (i < static_cast<int>(masks.size()) && masks[i] &&
                detection.cls_id >= 0 && detection.cls_id < 3) {
                cv::Mat mask(frame.image.rows, frame.image.cols, CV_8UC1, masks[i].get());
                cv::Mat active = mask > 0;
                cv::Mat overlay = frame.visualization.clone();
                overlay.setTo(colors[detection.cls_id], active);
                cv::addWeighted(overlay, 0.5, frame.visualization, 0.5, 0, frame.visualization);
            }
            cv::rectangle(frame.visualization,
                          {detection.x, detection.y, detection.w, detection.h}, {0, 0, 255}, 2);
            const auto& ellipse = frame.ellipses[i];
            const cv::Scalar fit_color = !ellipse.valid
                                             ? cv::Scalar(0, 0, 255)
                                         : ellipse.partial_visibility
                                             ? cv::Scalar(0, 165, 255)
                                         : ellipse.source == EllipseSource::Mask
                                             ? cv::Scalar(0, 255, 0)
                                             : (ellipse.source == EllipseSource::Edge
                                                    ? cv::Scalar(255, 0, 255)
                                                    : cv::Scalar(0, 255, 255));
            cv::ellipse(frame.visualization, ellipse.ellipse, fit_color, 2);
            if (ellipse.partial_visibility) {
                for (size_t p = 0; p < ellipse.visible_arc_points.size(); p += 4)
                    cv::circle(frame.visualization, ellipse.visible_arc_points[p],
                               1, {0, 165, 255}, -1);
            }
            cv::putText(frame.visualization,
                        "cls=" + std::to_string(detection.cls_id) +
                            " " + cv::format("%s q=%.2f%s", EllipseSourceName(ellipse.source),
                                             ellipse.quality,
                                             ellipse.partial_visibility ? " PARTIAL" : ""),
                        {detection.x, std::max(18, detection.y - 5)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 2);
        }

        if (!frame.pose.valid) {
            draw_text(frame.visualization, "Pose: --", 190, {255, 255, 0});
            return;
        }
        const Pose6D& pose = display_fixed_pose_ ? frame.pose.fixed : frame.pose.automatic;
        pose_solver.draw_axis(frame.visualization, pose, frame.pose.use_middle_ring);
        cv::line(frame.visualization, frame.pose.reference_center, frame.pose.hole_center,
                 {255, 255, 0}, 3);
        draw_text(frame.visualization,
                  cv::format("Yaw: %.2f  Pitch: %.2f", pose.yaw_deg, pose.pitch_deg),
                  190, {0, 255, 0});
        draw_text(frame.visualization,
                  cv::format("XYZ(mm): %.1f %.1f %.1f", pose.tx_mm, pose.ty_mm, pose.tz_mm),
                  230, {0, 255, 0});
    }

private:
    static void draw_text(cv::Mat& image, const std::string& text, int y,
                          const cv::Scalar& color) {
        cv::putText(image, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, 0.9, {0, 0, 0}, 5);
        cv::putText(image, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
    }

    bool display_fixed_pose_;
};

// ============================== 6. 结果保存模块 ==============================

class ResultWriter {
public:
    explicit ResultWriter(fs::path output_root) : output_root_(std::move(output_root)) {
        fs::create_directories(output_root_ / "visual");
        fs::create_directories(output_root_ / "labels");
        fs::create_directories(output_root_ / "ellipses");
        fs::create_directories(output_root_ / "poses");
    }

    bool run(const fs::path& relative_key, const FrameContext& frame) const {
        fs::path visual_path = output_root_ / "visual" / relative_key;
        visual_path.replace_extension(".jpg");
        fs::path label_path = output_root_ / "labels" / relative_key;
        label_path.replace_extension(".txt");
        fs::path ellipse_path = output_root_ / "ellipses" / relative_key;
        ellipse_path.replace_extension(".txt");
        fs::path pose_path = output_root_ / "poses" / relative_key;
        pose_path.replace_extension(".txt");
        fs::create_directories(visual_path.parent_path());
        fs::create_directories(label_path.parent_path());
        fs::create_directories(ellipse_path.parent_path());
        fs::create_directories(pose_path.parent_path());
        if (!cv::imwrite(visual_path.string(), frame.visualization)) return false;
        write_yolo(label_path, frame);
        write_ellipses(ellipse_path, frame);
        write_pose(pose_path, frame.pose);
        return true;
    }

private:
    static void write_yolo(const fs::path& path, const FrameContext& frame) {
        std::ofstream output(path);
        output << std::fixed << std::setprecision(8);
        for (int i = 0; i < frame.detections.count; ++i) {
            const auto& d = frame.detections.results_box[i];
            output << d.cls_id << ' '
                   << (d.x + d.w * 0.5) / frame.image.cols << ' '
                   << (d.y + d.h * 0.5) / frame.image.rows << ' '
                   << static_cast<double>(d.w) / frame.image.cols << ' '
                   << static_cast<double>(d.h) / frame.image.rows << ' '
                   << d.prop << '\n';
        }
    }

    static void write_ellipses(const fs::path& path, const FrameContext& frame) {
        std::ofstream output(path);
        output << "# class_id center_x center_y major_axis minor_axis angle confidence source valid quality inlier_ratio inliers mean_error_px coverage_deg quadrants border_truncated partial visible_arc_ratio removed_border_points support_points center_std major_std minor_std angle_std cov_condition geometry_ok temporal conic00 conic01 conic02 conic11 conic12 conic22\n"
               << std::fixed << std::setprecision(6);
        for (int i = 0; i < frame.detections.count; ++i) {
            const auto& d = frame.detections.results_box[i];
            const auto& e = frame.ellipses[i];
            output << d.cls_id << ' ' << e.ellipse.center.x << ' ' << e.ellipse.center.y << ' '
                   << std::max(e.ellipse.size.width, e.ellipse.size.height) << ' '
                   << std::min(e.ellipse.size.width, e.ellipse.size.height) << ' '
                   << e.ellipse.angle << ' ' << d.prop << ' '
                   << EllipseSourceName(e.source) << ' ' << e.valid << ' ' << e.quality << ' '
                   << e.inlier_ratio << ' ' << e.inliers << ' ' << e.mean_error_px << ' '
                   << e.angular_coverage_deg << ' ' << e.occupied_quadrants << ' '
                   << e.border_truncated << ' ' << e.partial_visibility << ' '
                   << e.visible_arc_ratio << ' ' << e.removed_border_points << ' '
                   << e.visible_arc_points.size() << ' '
                   << e.center_std_px << ' ' << e.major_axis_std_px << ' '
                   << e.minor_axis_std_px << ' ' << e.angle_std_deg << ' '
                   << e.covariance_condition << ' ' << e.geometry_consistent << ' '
                   << e.temporally_filtered << ' ' << e.conic(0, 0) << ' '
                   << e.conic(0, 1) << ' ' << e.conic(0, 2) << ' '
                   << e.conic(1, 1) << ' ' << e.conic(1, 2) << ' '
                   << e.conic(2, 2) << '\n';
        }
    }

    static void write_pose_values(std::ostream& output, const Pose6D& pose) {
        output << pose.yaw_deg << ' ' << pose.pitch_deg << ' ' << pose.roll_deg << ' '
               << pose.tx_mm << ' ' << pose.ty_mm << ' ' << pose.tz_mm;
    }

    static void write_pose(const fs::path& path, const PoseResult& pose) {
        std::ofstream output(path);
        output << "# valid ref_class ref_cx ref_cy hole_cx hole_cy "
                  "auto_yaw auto_pitch auto_roll auto_tx auto_ty auto_tz "
                  "fixed_yaw fixed_pitch fixed_roll fixed_tx fixed_ty fixed_tz\n"
               << std::fixed << std::setprecision(9);
        if (!pose.valid) {
            output << "0 -1";
            for (int i = 0; i < 16; ++i) output << " nan";
            output << '\n';
            return;
        }
        output << "1 " << pose.reference_class << ' ' << pose.reference_center.x << ' '
               << pose.reference_center.y << ' ' << pose.hole_center.x << ' '
               << pose.hole_center.y << ' ';
        write_pose_values(output, pose.automatic);
        output << ' ';
        write_pose_values(output, pose.fixed);
        output << '\n';
    }

    fs::path output_root_;
};

// ============================== 7. 批量计时统计模块 ==============================

static bool is_supported_image(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension == ".jpg" || extension == ".jpeg" ||
           extension == ".png" || extension == ".bmp" ||
           extension == ".tif" || extension == ".tiff" ||
           extension == ".webp";
}

static std::vector<fs::path> list_images(const AppConfig& config) {
    if (fs::is_regular_file(config.input_path)) return {config.input_path};
    if (!fs::is_directory(config.input_path))
        throw std::runtime_error(
            "Input path does not exist: " + config.input_path.string());
    std::vector<fs::path> images;
    if (config.recursive) {
        for (const auto& entry :
             fs::recursive_directory_iterator(config.input_path))
            if (entry.is_regular_file() &&
                is_supported_image(entry.path()))
                images.push_back(entry.path());
    } else {
        for (const auto& entry : fs::directory_iterator(config.input_path))
            if (entry.is_regular_file() &&
                is_supported_image(entry.path()))
                images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    return images;
}

static fs::path relative_key(const fs::path& image,
                             const AppConfig& config) {
    if (fs::is_regular_file(config.input_path)) return image.filename();
    std::error_code error;
    const fs::path relative =
        fs::relative(image, config.input_path, error);
    return error ? image.filename() : relative;
}

static double timing_percentile(std::vector<double> values,
                                double percent) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * percent / 100.0;
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper) return values[lower];
    return values[lower] * (upper - position) +
           values[upper] * (position - lower);
}

struct TimingStatistics {
    int count = 0;
    double mean = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double p95 = 0.0;
};

static std::vector<double> successful_stage_values(
    const std::vector<FrameTiming>& timings, size_t stage) {
    std::vector<double> values;
    for (const auto& timing : timings)
        if (timing.status == "ok")
            values.push_back(timing.milliseconds[stage]);
    return values;
}

static TimingStatistics calculate_timing_statistics(
    const std::vector<double>& values) {
    TimingStatistics result;
    result.count = static_cast<int>(values.size());
    if (values.empty()) return result;
    result.mean =
        std::accumulate(values.begin(), values.end(), 0.0) /
        values.size();
    result.minimum = *std::min_element(values.begin(), values.end());
    result.maximum = *std::max_element(values.begin(), values.end());
    result.p95 = timing_percentile(values, 95.0);
    return result;
}

static void write_timing_tables(
    const fs::path& timing_root,
    const std::vector<FrameTiming>& timings) {
    fs::create_directories(timing_root);
    std::ofstream per_image(timing_root / "per_image_timing.csv");
    per_image << "index,image,status";
    for (const char* key : kTimingStageKeys) per_image << ',' << key << "_ms";
    per_image << '\n' << std::fixed << std::setprecision(6);
    for (size_t index = 0; index < timings.size(); ++index) {
        const auto& timing = timings[index];
        per_image << index + 1 << ",\"" << timing.image.string()
                  << "\"," << timing.status;
        for (double value : timing.milliseconds) per_image << ',' << value;
        per_image << '\n';
    }

    std::ofstream summary(timing_root / "timing_summary.csv");
    summary << "stage,count,mean_ms,p95_ms,min_ms,max_ms,mean_fps_equivalent\n"
            << std::fixed << std::setprecision(6);
    for (size_t stage = 0; stage < kTimingStageCount; ++stage) {
        const TimingStatistics statistics = calculate_timing_statistics(
            successful_stage_values(timings, stage));
        summary << kTimingStageKeys[stage] << ',' << statistics.count
                << ',' << statistics.mean << ',' << statistics.p95
                << ',' << statistics.minimum << ',' << statistics.maximum
                << ',' << (statistics.mean > 1e-9
                                ? 1000.0 / statistics.mean : 0.0)
                << '\n';
    }
}

static void draw_stage_timing_chart(
    const fs::path& path, const std::string& title,
    const std::vector<double>& values,
    const TimingStatistics& statistics) {
    constexpr int width = 1500;
    constexpr int height = 760;
    constexpr int left = 105;
    constexpr int right = width - 55;
    constexpr int top = 105;
    constexpr int bottom = height - 120;
    cv::Mat chart(height, width, CV_8UC3, cv::Scalar(248, 248, 248));
    const double graph_max = std::max(
        1.0, std::max(statistics.maximum, statistics.p95 * 1.15));
    for (int tick = 0; tick <= 5; ++tick) {
        const int y = bottom - (bottom - top) * tick / 5;
        const double value = graph_max * tick / 5.0;
        cv::line(chart, {left, y}, {right, y}, {220, 220, 220}, 1);
        cv::putText(chart, cv::format("%.2f", value),
                    {18, y + 6}, cv::FONT_HERSHEY_SIMPLEX,
                    0.50, {60, 60, 60}, 1, cv::LINE_AA);
    }
    cv::rectangle(chart, {left, top}, {right, bottom},
                  {70, 70, 70}, 2);
    cv::putText(chart, title + " timing (ms)",
                {70, 48}, cv::FONT_HERSHEY_SIMPLEX,
                0.95, {25, 25, 25}, 2, cv::LINE_AA);
    cv::putText(
        chart,
        cv::format("n=%d  mean=%.3f ms  P95=%.3f ms  min=%.3f  max=%.3f",
                   statistics.count, statistics.mean, statistics.p95,
                   statistics.minimum, statistics.maximum),
        {70, 80}, cv::FONT_HERSHEY_SIMPLEX,
        0.58, {45, 45, 45}, 1, cv::LINE_AA);

    const auto y_for = [&](double value) {
        return bottom - cvRound(
            std::clamp(value / graph_max, 0.0, 1.0) *
            (bottom - top));
    };
    const int mean_y = y_for(statistics.mean);
    const int p95_y = y_for(statistics.p95);
    cv::line(chart, {left, mean_y}, {right, mean_y},
             {0, 170, 0}, 2);
    cv::line(chart, {left, p95_y}, {right, p95_y},
             {0, 120, 255}, 2);

    std::vector<cv::Point> points;
    points.reserve(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        const double ratio = values.size() <= 1
            ? 0.5
            : static_cast<double>(index) / (values.size() - 1);
        const int x = left + cvRound(ratio * (right - left));
        points.emplace_back(x, y_for(values[index]));
    }
    if (points.size() > 1)
        cv::polylines(chart, points, false, {210, 80, 30},
                      2, cv::LINE_AA);
    else if (points.size() == 1)
        cv::circle(chart, points.front(), 4, {210, 80, 30}, -1);
    cv::putText(chart, "Blue=per-image  Green=mean  Orange=P95",
                {left, height - 48}, cv::FONT_HERSHEY_SIMPLEX,
                0.55, {45, 45, 45}, 1, cv::LINE_AA);
    cv::putText(chart, "Image index ->",
                {width / 2 - 65, height - 18},
                cv::FONT_HERSHEY_SIMPLEX, 0.50,
                {45, 45, 45}, 1, cv::LINE_AA);
    fs::create_directories(path.parent_path());
    cv::imwrite(path.string(), chart);
}

static void write_timing_charts(
    const fs::path& chart_root,
    const std::vector<FrameTiming>& timings) {
    fs::create_directories(chart_root);
    for (size_t stage = 0; stage < kTimingStageCount; ++stage) {
        const std::vector<double> values =
            successful_stage_values(timings, stage);
        const TimingStatistics statistics =
            calculate_timing_statistics(values);
        std::ostringstream filename;
        filename << std::setw(2) << std::setfill('0') << stage + 1
                 << '_' << kTimingStageKeys[stage] << ".png";
        draw_stage_timing_chart(
            chart_root / filename.str(), kTimingStageTitles[stage],
            values, statistics);
    }
}

// ============================== 命令行与串行主流程 ==============================

static AppConfig parse_args(int argc, char** argv) {
    AppConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + key);
            return argv[++i];
        };
        if (key == "--model") config.model_path = next();
        else if (key == "--image" || key == "--input-path" ||
                 key == "--images")
            config.input_path = next();
        else if (key == "--output") config.output_path = next();
        else if (key == "--fixed-distance") config.fixed_distance_mm = std::stod(next());
        else if (key == "--show") config.show = true;
        else if (key == "--no-show") config.show = false;
        else if (key == "--recursive") config.recursive = true;
        else if (key == "--display-fixed") config.display_fixed_pose = true;
        else if (key == "--force-reference-box") config.force_reference_box = true;
        else if (key == "--hole-mask") config.fit_hole_from_mask = true;
        else if (key == "--help" || key == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " --model best.rknn --input-path images --output result [--no-show]"
                         " [--display-fixed] [--fixed-distance 3000]"
                         " [--force-reference-box] [--hole-mask] [--recursive]\n"
                         "  Visualization is displayed by default; use --no-show on a"
                         " headless board.\n"
                         "  --image input.jpg remains supported for one image.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (config.model_path.empty() || config.input_path.empty() ||
        config.output_path.empty()) {
        throw std::runtime_error(
            "--model, --input-path/--image and --output are required");
    }
    return config;
}

int main(int argc, char** argv) {
    try {
        const AppConfig config = parse_args(argc, argv);
        const std::vector<fs::path> images = list_images(config);
        if (images.empty())
            throw std::runtime_error("No supported images found");
        // 与 final_version.cpp 保持一致，避免 OpenCV 内部线程数差异污染对比。
        cv::setNumThreads(1);

        // 全目录严格串行执行，测到的是单帧延迟，而不是多线程吞吐率。
        ImageLoader loader;
        RknnInference inference(config.model_path);
        EllipseStage ellipse_stage(config);
        PoseSolver pose_solver(config.fixed_distance_mm);
        Visualizer visualizer(config.display_fixed_pose);
        std::vector<FrameTiming> timings;
        timings.reserve(images.size());

        if (config.show) {
            cv::namedWindow("Single Frame Pipeline", cv::WINDOW_NORMAL);
            cv::resizeWindow("Single Frame Pipeline", 1280, 720);
        }

        for (size_t index = 0; index < images.size(); ++index) {
            FrameContext frame;
            FrameTiming timing;
            timing.image = images[index];

            const auto read_begin = Clock::now();
            if (!loader.run(images[index], frame)) {
                timing.milliseconds[
                    static_cast<size_t>(TimingStage::Read)] =
                    elapsed_ms(read_begin);
                timing.status = "image_read_error";
                timings.push_back(timing);
                std::cerr << '[' << index + 1 << '/' << images.size()
                          << "] read failed: " << images[index] << '\n';
                continue;
            }
            timing.milliseconds[
                static_cast<size_t>(TimingStage::Read)] =
                elapsed_ms(read_begin);

            InferenceTiming inference_timing;
            if (!inference.run(frame, inference_timing)) {
                timing.milliseconds[
                    static_cast<size_t>(TimingStage::Preprocess)] =
                    inference_timing.preprocess_ms;
                timing.milliseconds[
                    static_cast<size_t>(TimingStage::RknnInference)] =
                    inference_timing.inference_ms;
                timing.milliseconds[
                    static_cast<size_t>(TimingStage::Postprocess)] =
                    inference_timing.postprocess_ms;
                timing.status = "rknn_pipeline_error";
                timings.push_back(timing);
                std::cerr << '[' << index + 1 << '/' << images.size()
                          << "] inference failed: " << images[index] << '\n';
                continue;
            }
            timing.milliseconds[
                static_cast<size_t>(TimingStage::Preprocess)] =
                inference_timing.preprocess_ms;
            timing.milliseconds[
                static_cast<size_t>(TimingStage::RknnInference)] =
                inference_timing.inference_ms;
            timing.milliseconds[
                static_cast<size_t>(TimingStage::Postprocess)] =
                inference_timing.postprocess_ms;

            auto stage_begin = Clock::now();
            ellipse_stage.run(frame);
            timing.milliseconds[
                static_cast<size_t>(TimingStage::EllipseFit)] =
                elapsed_ms(stage_begin);

            stage_begin = Clock::now();
            pose_solver.run(frame);
            timing.milliseconds[
                static_cast<size_t>(TimingStage::PoseSolve)] =
                elapsed_ms(stage_begin);

            timings.push_back(timing);

            const double algorithm_total_ms = std::accumulate(
                timing.milliseconds.begin(),
                timing.milliseconds.end(),
                0.0);
            std::cout << '[' << index + 1 << '/' << images.size() << "] "
                      << images[index].filename()
                      << " algorithm_total=" << std::fixed
                      << std::setprecision(3) << algorithm_total_ms
                      << " ms\n";
            if (config.show) {
                // 可视化绘制和窗口显示明确放在所有阶段计时结束之后。
                // 它们不会进入 per_image_timing.csv 或 mean/P95。
                visualizer.run(frame, pose_solver);
                cv::imshow("Single Frame Pipeline", frame.visualization);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q') break;
            }
        }
        if (config.show) cv::destroyAllWindows();

        const fs::path timing_root = config.output_path / "timing";
        write_timing_tables(timing_root, timings);
        write_timing_charts(timing_root / "charts", timings);

        std::cout << "\nTiming summary (successful images):\n";
        for (size_t stage = 0; stage < kTimingStageCount; ++stage) {
            const TimingStatistics statistics =
                calculate_timing_statistics(
                    successful_stage_values(timings, stage));
            std::cout << "  " << std::setw(16) << std::left
                      << kTimingStageKeys[stage]
                      << " mean=" << std::setw(9) << std::right
                      << std::fixed << std::setprecision(3)
                      << statistics.mean << " ms"
                      << "  P95=" << std::setw(9) << statistics.p95
                      << " ms  n=" << statistics.count << '\n';
        }
        std::vector<double> algorithm_totals;
        for (const auto& timing : timings) {
            if (timing.status != "ok") continue;
            algorithm_totals.push_back(std::accumulate(
                timing.milliseconds.begin(),
                timing.milliseconds.end(),
                0.0));
        }
        const TimingStatistics total_statistics =
            calculate_timing_statistics(algorithm_totals);
        std::cout << "  " << std::setw(16) << std::left
                  << "algorithm_total"
                  << " mean=" << std::setw(9) << std::right
                  << std::fixed << std::setprecision(3)
                  << total_statistics.mean << " ms"
                  << "  P95=" << std::setw(9)
                  << total_statistics.p95 << " ms  n="
                  << total_statistics.count << '\n';
        std::cout << "Done. Results: "
                  << fs::absolute(config.output_path) << '\n'
                  << "Timing CSV: "
                  << fs::absolute(timing_root / "timing_summary.csv")
                  << '\n'
                  << "Timing charts: "
                  << fs::absolute(timing_root / "charts") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
