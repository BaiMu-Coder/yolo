#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "ellipse_fitter.hpp"
#include "image_process.hpp"
#include "pose_estimator_lm.hpp"
#include "yolov8seg.hpp"

namespace fs = std::filesystem;

#ifndef BATCH_RING_MASK_ONLY
#define BATCH_RING_MASK_ONLY 0
#endif

struct Args {
    std::string model_path;
    fs::path input_path;
    fs::path output_path;
    std::string video_name;
    std::string mode = "auto";  // auto / images / video
    bool show = false;
    bool save_video_frames = false;
    bool display_fixed_pose = false;
    bool fit_hole_from_mask = false;
    bool force_reference_box = false;
    // 独立的 batch_mask_ring_visualization 入口会把该默认值编译为 true。
    // 处理和保存格式保持一致，只改变最终画面叠加内容。
    bool ring_mask_only = BATCH_RING_MASK_ONLY != 0;
    double fixed_distance_mm = 3000.0;
    float ellipse_deviation_ratio = 0.30f;
    int start_frame = 0;
    int end_frame = -1;
    int frame_step = 1;
};

static void print_help(const char* app) {
    std::cout
        << "Usage:\n"
        << "  Images: " << app
        << " --model model.rknn --input-path /data/images --output-path /data/result\n"
        << "  Video : " << app
        << " --model model.rknn --input-path /data/video --video-name test.mp4"
           " --output-path /data/result\n\n"
        << "Options:\n"
        << "  --mode auto|images|video   Usually auto is sufficient\n"
        << "  --show                     Show the annotated frame in real time\n"
        << "  --save-video-frames        Also save every annotated video frame as JPG\n"
        << "  --display-fixed            Draw fixed-distance pose instead of auto pose\n"
        << "  --fixed-distance MM        Fixed distance, default 3000\n"
        << "  --hole-box                 Use box center/inscribed circle for cls2 (default)\n"
        << "  --hole-mask                Try Mask ellipse fitting for cls2\n"
        << "  --force-reference-box      Force cls0/cls1 to use box inscribed circles\n"
        << "  Visualization: "
        << (BATCH_RING_MASK_ONLY
                ? "Mask + detection boxes + cls0/cls1 fitted ellipses\n"
                : "Full detection + ellipse + pose\n")
        << "  --start-frame N --end-frame N --frame-step N\n";
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };
        if (key == "--model") args.model_path = value("--model");
        else if (key == "--input-path") args.input_path = value("--input-path");
        else if (key == "--output-path") args.output_path = value("--output-path");
        else if (key == "--video-name") args.video_name = value("--video-name");
        else if (key == "--mode") args.mode = value("--mode");
        else if (key == "--fixed-distance") args.fixed_distance_mm = std::stod(value("--fixed-distance"));
        else if (key == "--start-frame") args.start_frame = std::stoi(value("--start-frame"));
        else if (key == "--end-frame") args.end_frame = std::stoi(value("--end-frame"));
        else if (key == "--frame-step") args.frame_step = std::max(1, std::stoi(value("--frame-step")));
        else if (key == "--show") args.show = true;
        else if (key == "--save-video-frames") args.save_video_frames = true;
        else if (key == "--display-fixed") args.display_fixed_pose = true;
        else if (key == "--hole-box") args.fit_hole_from_mask = false;
        else if (key == "--hole-mask") args.fit_hole_from_mask = true;
        else if (key == "--force-reference-box") args.force_reference_box = true;
        else if (key == "--help" || key == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (args.model_path.empty() || args.input_path.empty() || args.output_path.empty()) {
        throw std::runtime_error("--model, --input-path and --output-path are required");
    }
    if (!args.video_name.empty()) args.input_path /= args.video_name;
    if (args.mode == "auto") {
        args.mode = fs::is_directory(args.input_path) ? "images" : "video";
    }
    if (args.mode != "images" && args.mode != "video") {
        throw std::runtime_error("--mode must be auto, images or video");
    }
    return args;
}


static void draw_text(cv::Mat& image, const std::string& text, int y,
                      const cv::Scalar& color, double scale = 0.8) {
    cv::putText(image, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 5);
    cv::putText(image, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, 2);
}

static int best_index(const object_detect_result_list& result,
                      const std::vector<EllipseFitResult>& ellipses,
                      int class_id) {
    int best = -1;
    float score = -1.0f;
    for (int i = 0; i < result.count; ++i) {
        if (result.results_box[i].cls_id != class_id || i >= static_cast<int>(ellipses.size())) continue;
        if (!ellipses[i].valid) continue;
        const float candidate = EllipseSelectionScore(ellipses[i], result.results_box[i].prop);
        if (candidate > score) { best = i; score = candidate; }
    }
    return best;
}

struct OutputDirs {
    fs::path visual;
    fs::path labels;
    fs::path ellipses;
    fs::path poses;
};

class SequentialProcessor {
public:
    explicit SequentialProcessor(const Args& args) : args_(args), model_(args.model_path) {
        const int status = model_.init();
        if (status != RKNN_SUCC) throw std::runtime_error("RKNN model init failed: " + std::to_string(status));
        K_ = (cv::Mat_<double>(3, 3) << 1639.6, 0, 960, 0, 2165.4, 540, 0, 0, 1);
        D_ = cv::Mat::zeros(4, 1, CV_64F);
        const double scale = 20.0 / 45.5;
        drogue_.radius_cls0_mm = 1200.0 * scale;
        drogue_.radius_cls1_mm = 980.0 * scale;
        drogue_.radius_hole_mm = 120.0 * scale;
        drogue_.length_L_mm = 920.0 * scale;
        pose_estimator_.Reset(K_, D_, drogue_);
    }

    bool process(const cv::Mat& input, const std::string& key, const OutputDirs& dirs,
                 cv::Mat& visualization) {
        if (input.empty()) return false;
        if (args_.mode == "images") temporal_filter_.Reset();
        cv::Mat inference_frame = input.clone();
        image_process preprocessor(inference_frame);
        const cv::Size model_input = model_.model_input_size();
        if (preprocessor.image_preprocessing(model_input.width,
                                             model_input.height) != 0)
            return fail(key, "preprocess failed");
        int input_size = 0;
        uint8_t* input_data = preprocessor.get_image_buffer(&input_size);
        if (input_data == nullptr) return fail(key, "empty model input");
        if (model_.set_input_data(input_data, input_size) != RKNN_SUCC) return fail(key, "set input failed");
        if (model_.rknn_model_inference() != RKNN_SUCC) return fail(key, "inference failed");
        if (model_.get_output_data() != RKNN_SUCC) return fail(key, "get outputs failed");

        object_detect_result_list result;
        letterbox letter_box = preprocessor.get_letterbox();
        const int post_status = model_.post_process(result, letter_box);
        model_.release_output_data();
        if (post_status != RKNN_SUCC) return fail(key, "postprocess failed");

        visualization = input.clone();
        EllipseFitConfig fit_config;
        fit_config.center_deviation_ratio = args_.ellipse_deviation_ratio;
        const EllipseFitter ellipse_fitter(fit_config);
        std::vector<EllipseFitResult> ellipse_results;
        ellipse_results.reserve(result.count);
        const cv::Scalar colors[3] = {{0, 0, 255}, {0, 255, 0}, {255, 255, 0}};

        std::ofstream yolo_file(dirs.labels / (key + ".txt"));
        std::ofstream ellipse_file(dirs.ellipses / (key + ".txt"));
        std::ofstream pose_file(dirs.poses / (key + ".txt"));
        if (!yolo_file || !ellipse_file || !pose_file) return fail(key, "cannot create txt output");
        yolo_file << std::fixed << std::setprecision(8);
        ellipse_file << "# class_id center_x_px center_y_px major_axis_px minor_axis_px angle_deg confidence fit_source valid quality inlier_ratio inliers mean_error_px coverage_deg quadrants border_truncated partial visible_arc_ratio removed_border_points support_points center_std_px major_std_px minor_std_px angle_std_deg cov_condition geometry_ok temporal conic00 conic01 conic02 conic11 conic12 conic22\n"
                     << std::fixed << std::setprecision(6);

        for (int i = 0; i < result.count; ++i) {
            const auto& detection = result.results_box[i];
            const uint8_t* mask = nullptr;
            const uint8_t* mask_probability = nullptr;
            if (i < static_cast<int>(result.results_mask[0].each_of_mask.size()) &&
                result.results_mask[0].each_of_mask[i]) {
                mask = result.results_mask[0].each_of_mask[i].get();
            }
            if (i < static_cast<int>(result.results_mask[0].each_of_mask_probability.size()) &&
                result.results_mask[0].each_of_mask_probability[i]) {
                mask_probability = result.results_mask[0].each_of_mask_probability[i].get();
            }
            // 拟合模式与在线推理一致：
            // 外/中参考圈只走“Mask -> Box”，不尝试容易受反光影响的灰度 Edge；
            // 内孔默认 Box，--hole-mask 才允许“Mask -> Edge -> Box”；
            // --force-reference-box 的优先级最高，强制外/中圈使用框内切圆。
            const bool is_reference = detection.cls_id == 0 || detection.cls_id == 1;
            const bool force_box = (is_reference && args_.force_reference_box) ||
                                   (detection.cls_id == 2 && !args_.fit_hole_from_mask);
            const cv::Rect detection_rect(detection.x, detection.y,
                                          detection.w, detection.h);
            const EllipseFitMode fit_mode = force_box
                                                ? EllipseFitMode::ForceBox
                                                : (is_reference
                                                       ? EllipseFitMode::PreferMaskNoEdge
                                                       : EllipseFitMode::PreferMask);
            EllipseFitResult ellipse = ellipse_fitter.Fit(
                input, detection_rect, mask, fit_mode, mask_probability);
            ellipse_results.push_back(ellipse);

            const double center_x = (detection.x + detection.w * 0.5) / input.cols;
            const double center_y = (detection.y + detection.h * 0.5) / input.rows;
            const double width = static_cast<double>(detection.w) / input.cols;
            const double height = static_cast<double>(detection.h) / input.rows;
            yolo_file << detection.cls_id << ' ' << center_x << ' ' << center_y << ' '
                      << width << ' ' << height << ' ' << detection.prop << '\n';

            const int class_id = detection.cls_id;
            if (mask != nullptr && class_id >= 0 && class_id < 3) {
                cv::Mat full_mask(input.rows, input.cols, CV_8UC1, const_cast<uint8_t*>(mask));
                cv::Mat active = full_mask > 0;
                cv::Mat overlay = visualization.clone();
                overlay.setTo(colors[class_id], active);
                cv::addWeighted(overlay, 0.5, visualization, 0.5, 0.0, visualization);
            }
            const cv::Scalar fit_color = !ellipse.valid
                                             ? cv::Scalar(0, 0, 255)
                                         : ellipse.partial_visibility
                                             ? cv::Scalar(0, 165, 255)
                                         : ellipse.source == EllipseSource::Mask
                                             ? cv::Scalar(0, 255, 0)
                                             : (ellipse.source == EllipseSource::Edge
                                                    ? cv::Scalar(255, 0, 255)
                                                    : cv::Scalar(0, 255, 255));
            if (args_.ring_mask_only) {
                // 演示入口画所有检测框，并只画外圈(cls0)和中圈(cls1)的最终拟合椭圆：
                // 检测框红色、外圈青色、中圈紫色。中心点、弧点、文字和内孔不叠加。
                cv::rectangle(visualization,
                              {detection.x, detection.y, detection.w, detection.h},
                              {0, 0, 255}, 2);
                if (is_reference && ellipse.valid) {
                    const cv::Scalar ring_color = class_id == 0
                                                      ? cv::Scalar(255, 255, 0)
                                                      : cv::Scalar(255, 0, 255);
                    cv::ellipse(visualization, ellipse.ellipse, ring_color, 3);
                }
            } else {
                cv::rectangle(visualization,
                              {detection.x, detection.y, detection.w, detection.h},
                              {0, 0, 255}, 2);
                cv::ellipse(visualization, ellipse.ellipse, fit_color, 2);
                cv::circle(visualization, ellipse.ellipse.center, 2,
                           {0, 0, 255}, -1);
                if (ellipse.partial_visibility) {
                    for (size_t p = 0;
                         p < ellipse.visible_arc_points.size(); p += 4)
                        cv::circle(visualization,
                                   ellipse.visible_arc_points[p], 1,
                                   {0, 165, 255}, -1);
                }
                const std::string label =
                    "cls=" + std::to_string(class_id) + " conf=" +
                    cv::format("%.2f q=%.2f %s%s", detection.prop,
                               ellipse.quality,
                               EllipseSourceName(ellipse.source),
                               ellipse.partial_visibility ? " PARTIAL" : "");
                cv::putText(visualization, label,
                            {detection.x, std::max(18, detection.y - 5)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.55,
                            {255, 255, 255}, 2);
            }
        }
        std::vector<int> class_ids(result.count);
        std::vector<float> detection_confidences(result.count);
        for (int i = 0; i < result.count; ++i) {
            class_ids[i] = result.results_box[i].cls_id;
            detection_confidences[i] = result.results_box[i].prop;
        }
        const int hole = best_index(result, ellipse_results, 2);
        int outer_count = 0, middle_count = 0;
        for (int i = 0; i < result.count; ++i) {
            if (!ellipse_results[i].valid) continue;
            outer_count += class_ids[i] == 0;
            middle_count += class_ids[i] == 1;
        }
        std::function<float(int, int)> pose_pair_score;
        if (hole >= 0 && outer_count * middle_count > 1 &&
            outer_count * middle_count <= 4) {
            pose_pair_score = [&](int outer_index, int middle_index) {
                const auto& outer_fit = ellipse_results[outer_index];
                const auto& middle_fit = ellipse_results[middle_index];
                const PoseEllipseObservation outer_obs{
                    outer_fit.ellipse, EllipseObservationSigmaPx(outer_fit), true,
                    outer_fit.visible_arc_points, outer_fit.partial_visibility,
                    outer_fit.visible_arc_ratio};
                const PoseEllipseObservation middle_obs{
                    middle_fit.ellipse, EllipseObservationSigmaPx(middle_fit), true,
                    middle_fit.visible_arc_points, middle_fit.partial_visibility,
                    middle_fit.visible_arc_ratio};
                const float reprojection = static_cast<float>(
                    pose_estimator_.EvaluateDualReprojectionScore(
                    outer_obs, middle_obs, ellipse_results[hole].ellipse.center,
                    EllipseObservationSigmaPx(ellipse_results[hole])));
                return reprojection * std::sqrt(
                    std::max(0.0f, outer_fit.quality * middle_fit.quality));
            };
        }
        const RingPairSelection pair = ring_refiner_.SelectAndRefine(
            class_ids, detection_confidences, ellipse_results, 0, 1,
            pose_pair_score);
        const int outer = pair.outer_index;
        const int middle = pair.middle_index;
        if (args_.mode == "video") {
            if (outer >= 0) ellipse_results[outer] = temporal_filter_.Update(0, ellipse_results[outer]);
            if (middle >= 0) ellipse_results[middle] = temporal_filter_.Update(1, ellipse_results[middle]);
            if (hole >= 0) ellipse_results[hole] = temporal_filter_.Update(2, ellipse_results[hole]);
        }
        for (int i = 0; i < result.count; ++i) {
            const auto& detection = result.results_box[i];
            const auto& ellipse = ellipse_results[i];
            ellipse_file << detection.cls_id << ' ' << ellipse.ellipse.center.x << ' '
                         << ellipse.ellipse.center.y << ' '
                         << std::max(ellipse.ellipse.size.width, ellipse.ellipse.size.height) << ' '
                         << std::min(ellipse.ellipse.size.width, ellipse.ellipse.size.height) << ' '
                         << ellipse.ellipse.angle << ' ' << detection.prop << ' '
                         << EllipseSourceName(ellipse.source) << ' ' << ellipse.valid << ' '
                         << ellipse.quality << ' '
                         << ellipse.inlier_ratio << ' ' << ellipse.inliers << ' '
                         << ellipse.mean_error_px << ' ' << ellipse.angular_coverage_deg << ' '
                         << ellipse.occupied_quadrants << ' ' << ellipse.border_truncated << ' '
                         << ellipse.partial_visibility << ' ' << ellipse.visible_arc_ratio << ' '
                         << ellipse.removed_border_points << ' '
                         << ellipse.visible_arc_points.size() << ' ' << ellipse.center_std_px << ' '
                         << ellipse.major_axis_std_px << ' ' << ellipse.minor_axis_std_px << ' '
                         << ellipse.angle_std_deg << ' ' << ellipse.covariance_condition << ' '
                         << ellipse.geometry_consistent << ' ' << ellipse.temporally_filtered << ' '
                         << ellipse.conic(0, 0) << ' ' << ellipse.conic(0, 1) << ' '
                         << ellipse.conic(0, 2) << ' ' << ellipse.conic(1, 1) << ' '
                         << ellipse.conic(1, 2) << ' ' << ellipse.conic(2, 2) << '\n';
        }
        save_and_draw_pose(result, ellipse_results, outer, middle, hole,
                           visualization, pose_file, !args_.ring_mask_only);
        return true;
    }

private:
    bool fail(const std::string& key, const std::string& message) const {
        std::cerr << "[ERROR] " << key << ": " << message << '\n';
        return false;
    }

    static void write_pose_values(std::ostream& stream, const Pose6D& pose) {
        stream << pose.yaw_deg << ' ' << pose.pitch_deg << ' ' << pose.roll_deg << ' '
               << pose.tx_mm << ' ' << pose.ty_mm << ' ' << pose.tz_mm;
    }

    void save_and_draw_pose(const object_detect_result_list& result,
                            const std::vector<EllipseFitResult>& ellipses,
                            int outer, int middle, int hole,
                            cv::Mat& visualization, std::ofstream& pose_file,
                            bool draw_pose_visualization) {
        pose_file << "# valid reference_class ref_center_x_px ref_center_y_px hole_center_x_px hole_center_y_px "
                     "auto_yaw_deg auto_pitch_deg auto_roll_deg auto_tx_mm auto_ty_mm auto_tz_mm "
                     "fixed_yaw_deg fixed_pitch_deg fixed_roll_deg fixed_tx_mm fixed_ty_mm fixed_tz_mm "
                     "display_mode\n"
                  << std::fixed << std::setprecision(9);
        if (hole < 0 || (outer < 0 && middle < 0) || ellipses.size() < static_cast<size_t>(result.count)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            pose_file << "0 -1 " << nan << ' ' << nan << ' ' << nan << ' ' << nan;
            for (int i = 0; i < 12; ++i) pose_file << ' ' << nan;
            pose_file << " invalid\n";
            if (draw_pose_visualization) {
                draw_text(visualization, "Pose: --", 190, {255, 255, 0}, 1.2);
                draw_text(visualization, "Dist: --", 240, {255, 255, 0}, 1.2);
            }
            return;
        }
        int reference = -1;
        if (outer >= 0 && middle >= 0) {
            const float outer_score = EllipseSelectionScore(ellipses[outer], result.results_box[outer].prop);
            const float middle_score = EllipseSelectionScore(ellipses[middle], result.results_box[middle].prop);
            reference = outer_score >= middle_score ? outer : middle;
        } else {
            reference = outer >= 0 ? outer : middle;
        }
        const bool use_middle = result.results_box[reference].cls_id == 1;
        const cv::RotatedRect& target = ellipses[reference].ellipse;
        const cv::Point2f hole_center = ellipses[hole].ellipse.center;
        std::optional<PoseEllipseObservation> outer_observation;
        std::optional<PoseEllipseObservation> middle_observation;
        if (outer >= 0)
            outer_observation = PoseEllipseObservation{ellipses[outer].ellipse,
                EllipseObservationSigmaPx(ellipses[outer]), true,
                ellipses[outer].visible_arc_points, ellipses[outer].partial_visibility,
                ellipses[outer].visible_arc_ratio};
        if (middle >= 0)
            middle_observation = PoseEllipseObservation{ellipses[middle].ellipse,
                EllipseObservationSigmaPx(ellipses[middle]), true,
                ellipses[middle].visible_arc_points, ellipses[middle].partial_visibility,
                ellipses[middle].visible_arc_ratio};
        const double hole_sigma = EllipseObservationSigmaPx(ellipses[hole]);
        const Pose6D pose_auto = pose_estimator_.SolveDual(
            outer_observation, middle_observation, hole_center, hole_sigma, std::nullopt);
        const Pose6D pose_fixed = pose_estimator_.SolveDual(
            outer_observation, middle_observation, hole_center, hole_sigma,
            args_.fixed_distance_mm);
        const Pose6D& displayed = args_.display_fixed_pose ? pose_fixed : pose_auto;

        pose_file << "1 " << result.results_box[reference].cls_id << ' ' << target.center.x << ' '
                  << target.center.y << ' ' << hole_center.x << ' ' << hole_center.y << ' ';
        write_pose_values(pose_file, pose_auto);
        pose_file << ' ';
        write_pose_values(pose_file, pose_fixed);
        pose_file << ' ' << (args_.display_fixed_pose ? "fixed" : "auto") << '\n';

        if (!draw_pose_visualization) return;
        cv::ellipse(visualization, target, {255, 255, 0}, 4);
        cv::ellipse(visualization, ellipses[hole].ellipse, {255, 255, 0}, 4);
        cv::circle(visualization, hole_center, 6, {255, 255, 0}, -1);
        cv::line(visualization, target.center, hole_center, {255, 255, 0}, 3);
        pose_estimator_.DrawAxis(visualization, displayed, use_middle);
        draw_text(visualization, "Ref(" + std::string(use_middle ? "Mid" : "Out") + "): (" +
                  cv::format("%.0f, %.0f", target.center.x, target.center.y) + ")", 100, {255, 255, 255});
        draw_text(visualization, "Hole: (" + cv::format("%.0f, %.0f", hole_center.x, hole_center.y) + ")",
                  140, {255, 255, 255});
        draw_text(visualization, cv::format("Yaw:%.1f Pit:%.1f", displayed.yaw_deg, displayed.pitch_deg),
                  190, args_.display_fixed_pose ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0), 1.2);
        draw_text(visualization, cv::format("Dist: %.2fm (%s)", displayed.tz_mm / 1000.0,
                                           args_.display_fixed_pose ? "Fixed" : "Auto"),
                  240, args_.display_fixed_pose ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0), 1.2);
    }

    Args args_;
    yolov8seg model_;
    cv::Mat K_, D_;
    DrogueModel drogue_;
    PoseEstimatorLM pose_estimator_;
    RingPairRefiner ring_refiner_;
    EllipseTemporalFilter temporal_filter_;
};

static bool is_image_file(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
           extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

static OutputDirs create_output_dirs(const fs::path& root) {
    OutputDirs dirs{root / "visual", root / "labels", root / "ellipses", root / "poses"};
    fs::create_directories(dirs.visual);
    fs::create_directories(dirs.labels);
    fs::create_directories(dirs.ellipses);
    fs::create_directories(dirs.poses);
    return dirs;
}

static int process_images(const Args& args, SequentialProcessor& processor, const OutputDirs& dirs) {
    if (!fs::is_directory(args.input_path)) throw std::runtime_error("Image input is not a directory");
    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(args.input_path)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) throw std::runtime_error("No supported images found in: " + args.input_path.string());
    int success = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        const cv::Mat image = cv::imread(images[i].string(), cv::IMREAD_COLOR);
        cv::Mat visualization;
        const std::string key = images[i].stem().string();
        if (processor.process(image, key, dirs, visualization)) {
            const fs::path output_image = dirs.visual / (key + images[i].extension().string());
            if (!cv::imwrite(output_image.string(), visualization)) {
                std::cerr << "[ERROR] Cannot save " << output_image << '\n';
            } else {
                ++success;
            }
            if (args.show) {
                cv::imshow("Batch RKNN Result", visualization);
                const int key_code = cv::waitKey(1);
                if (key_code == 27 || key_code == 'q') break;
            }
        }
        std::cout << "\r[images] " << (i + 1) << '/' << images.size() << std::flush;
    }
    std::cout << "\nProcessed " << success << '/' << images.size() << " images\n";
    return success == static_cast<int>(images.size()) ? 0 : 2;
}

static int process_video(const Args& args, SequentialProcessor& processor, const OutputDirs& dirs) {
    cv::VideoCapture capture(args.input_path.string());
    if (!capture.isOpened()) throw std::runtime_error("Cannot open video: " + args.input_path.string());
    capture.set(cv::CAP_PROP_POS_FRAMES, args.start_frame);
    const double source_fps = capture.get(cv::CAP_PROP_FPS) > 0.0 ? capture.get(cv::CAP_PROP_FPS) : 25.0;
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const std::string video_stem = args.input_path.stem().string();
    cv::VideoWriter writer((args.output_path / (video_stem + "_result.mp4")).string(),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           source_fps / args.frame_step, {width, height});
    if (!writer.isOpened()) {
        writer.open((args.output_path / (video_stem + "_result.avi")).string(),
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    source_fps / args.frame_step, {width, height});
    }
    if (!writer.isOpened()) throw std::runtime_error("Cannot create MP4 or AVI annotated output video");

    int frame_id = args.start_frame;
    int processed = 0;
    while (true) {
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) break;
        const int current = frame_id++;
        if (args.end_frame >= 0 && current > args.end_frame) break;
        if ((current - args.start_frame) % args.frame_step != 0) continue;
        const std::string key = video_stem + "_frame_" + cv::format("%08d", current);
        cv::Mat visualization;
        if (!processor.process(frame, key, dirs, visualization)) continue;
        writer.write(visualization);
        if (args.save_video_frames) cv::imwrite((dirs.visual / (key + ".jpg")).string(), visualization);
        if (args.show) {
            cv::imshow("Batch RKNN Result", visualization);
            const int key_code = cv::waitKey(1);
            if (key_code == 27 || key_code == 'q') break;
        }
        ++processed;
        std::cout << "\r[video] frame=" << current << std::flush;
    }
    std::cout << "\nProcessed " << processed << " video frames\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        fs::create_directories(args.output_path);
        const OutputDirs dirs = create_output_dirs(args.output_path);
        if (args.show) {
            cv::namedWindow("Batch RKNN Result", cv::WINDOW_NORMAL);
            cv::resizeWindow("Batch RKNN Result", 1280, 720);
        }
        SequentialProcessor processor(args);
        const int status = args.mode == "images" ? process_images(args, processor, dirs)
                                                   : process_video(args, processor, dirs);
        cv::destroyAllWindows();
        return status;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        print_help(argv[0]);
        return 1;
    }
}
