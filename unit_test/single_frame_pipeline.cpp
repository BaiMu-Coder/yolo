// 单帧串行示例：读取 -> RKNN推理 -> 椭圆拟合 -> 位姿解算 -> 可视化 -> 保存。
// 所有适配接口仅存在于本文件，不修改项目其他模块。

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
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
    fs::path image_path;
    fs::path output_path;
    bool show = false;
    bool display_fixed_pose = false;
    bool force_reference_box = false;
    bool fit_hole_from_mask = false;
    double fixed_distance_mm = 3000.0;
    float ellipse_center_deviation_ratio = 0.30f;
};

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

    bool run(FrameContext& frame) {
        cv::Mat inference_image = frame.image.clone();
        image_process preprocessor(inference_image);
        if (preprocessor.image_preprocessing(640, 640) != 0) return false;

        int input_bytes = 0;
        uint8_t* input = preprocessor.get_image_buffer(&input_bytes);
        if (input == nullptr) return false;
        if (model_.set_input_data(input, input_bytes) != RKNN_SUCC) return false;
        if (model_.rknn_model_inference() != RKNN_SUCC) return false;
        if (model_.get_output_data() != RKNN_SUCC) return false;

        letterbox transform = preprocessor.get_letterbox();
        const int status = model_.post_process(frame.detections, transform);
        model_.release_output_data();
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
        const RingPairSelection pair = ring_refiner_.SelectAndRefine(
            class_ids, detection_confidences, frame.ellipses);
        const int outer = pair.outer_index;
        const int middle = pair.middle_index;
        const int hole = best_detection(frame.detections, frame.ellipses, 2);
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
                EllipseObservationSigmaPx(frame.ellipses[outer]), true};
        if (middle >= 0)
            middle_observation = PoseEllipseObservation{frame.ellipses[middle].ellipse,
                EllipseObservationSigmaPx(frame.ellipses[middle]), true};
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
            const cv::Scalar fit_color = ellipse.source == EllipseSource::Mask
                                             ? cv::Scalar(0, 255, 0)
                                             : (ellipse.source == EllipseSource::Edge
                                                    ? cv::Scalar(255, 0, 255)
                                                    : cv::Scalar(0, 255, 255));
            cv::ellipse(frame.visualization, ellipse.ellipse, fit_color, 2);
            cv::putText(frame.visualization,
                        "cls=" + std::to_string(detection.cls_id) +
                            " " + cv::format("%s q=%.2f", EllipseSourceName(ellipse.source),
                                             ellipse.quality),
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

    bool run(const std::string& name, const FrameContext& frame) const {
        const fs::path visual_path = output_root_ / "visual" / (name + ".jpg");
        if (!cv::imwrite(visual_path.string(), frame.visualization)) return false;
        write_yolo(output_root_ / "labels" / (name + ".txt"), frame);
        write_ellipses(output_root_ / "ellipses" / (name + ".txt"), frame);
        write_pose(output_root_ / "poses" / (name + ".txt"), frame.pose);
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
        output << "# class_id center_x center_y major_axis minor_axis angle confidence source quality inlier_ratio inliers mean_error_px coverage_deg quadrants center_std major_std minor_std angle_std cov_condition geometry_ok temporal conic00 conic01 conic02 conic11 conic12 conic22\n"
               << std::fixed << std::setprecision(6);
        for (int i = 0; i < frame.detections.count; ++i) {
            const auto& d = frame.detections.results_box[i];
            const auto& e = frame.ellipses[i];
            output << d.cls_id << ' ' << e.ellipse.center.x << ' ' << e.ellipse.center.y << ' '
                   << std::max(e.ellipse.size.width, e.ellipse.size.height) << ' '
                   << std::min(e.ellipse.size.width, e.ellipse.size.height) << ' '
                   << e.ellipse.angle << ' ' << d.prop << ' '
                   << EllipseSourceName(e.source) << ' ' << e.quality << ' '
                   << e.inlier_ratio << ' ' << e.inliers << ' ' << e.mean_error_px << ' '
                   << e.angular_coverage_deg << ' ' << e.occupied_quadrants << ' '
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
        else if (key == "--image") config.image_path = next();
        else if (key == "--output") config.output_path = next();
        else if (key == "--fixed-distance") config.fixed_distance_mm = std::stod(next());
        else if (key == "--show") config.show = true;
        else if (key == "--display-fixed") config.display_fixed_pose = true;
        else if (key == "--force-reference-box") config.force_reference_box = true;
        else if (key == "--hole-mask") config.fit_hole_from_mask = true;
        else if (key == "--help" || key == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " --model best.rknn --image input.jpg --output result [--show]"
                         " [--display-fixed] [--fixed-distance 3000]"
                         " [--force-reference-box] [--hole-mask]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (config.model_path.empty() || config.image_path.empty() || config.output_path.empty()) {
        throw std::runtime_error("--model, --image and --output are required");
    }
    return config;
}

int main(int argc, char** argv) {
    try {
        const AppConfig config = parse_args(argc, argv);
        FrameContext frame;

        // 一帧图像严格按照下面六步串行执行，模块间只通过 FrameContext 传递数据。
        ImageLoader loader;
        RknnInference inference(config.model_path);
        EllipseStage ellipse_stage(config);
        PoseSolver pose_solver(config.fixed_distance_mm);
        Visualizer visualizer(config.display_fixed_pose);
        ResultWriter writer(config.output_path);

        if (!loader.run(config.image_path, frame)) throw std::runtime_error("Image loading failed");
        if (!inference.run(frame)) throw std::runtime_error("RKNN inference failed");
        ellipse_stage.run(frame);
        pose_solver.run(frame);
        visualizer.run(frame, pose_solver);
        if (!writer.run(config.image_path.stem().string(), frame)) {
            throw std::runtime_error("Result saving failed");
        }

        if (config.show) {
            cv::namedWindow("Single Frame Pipeline", cv::WINDOW_NORMAL);
            cv::resizeWindow("Single Frame Pipeline", 1280, 720);
            cv::imshow("Single Frame Pipeline", frame.visualization);
            cv::waitKey(0);
        }
        std::cout << "Done. Results: " << fs::absolute(config.output_path) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
