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
#include "image_process.hpp"
#include "pose_estimator_lm.hpp"
#include "yolov8seg.hpp"

namespace fs = std::filesystem;

struct Args {
    std::string model_path;
    fs::path input_path;
    fs::path output_path;
    std::string video_name;
    std::string mode = "auto";  // auto / images / video
    bool show = false;
    bool save_video_frames = false;
    bool display_fixed_pose = false;
    bool fit_hole_from_mask = true;
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
        << "  --hole-box                 Force class-2 ellipse to use box inscribed circle\n"
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

struct EllipseResult {
    cv::RotatedRect ellipse;
    bool from_mask = false;
    int inliers = 0;
    float mean_error_px = std::numeric_limits<float>::infinity();
};

static float radial_error(const cv::RotatedRect& e, const cv::Point2f& p) {
    const float a = e.size.width * 0.5f;
    const float b = e.size.height * 0.5f;
    if (a < 1e-3f || b < 1e-3f) return 1e9f;
    const float x = p.x - e.center.x;
    const float y = p.y - e.center.y;
    const float angle = -e.angle * static_cast<float>(CV_PI) / 180.0f;
    const float c = std::cos(angle), s = std::sin(angle);
    const float xr = c * x - s * y;
    const float yr = s * x + c * y;
    const float r = std::sqrt(xr * xr / (a * a) + yr * yr / (b * b));
    return std::abs(r - 1.0f) * std::min(a, b);
}

static bool fit_ellipse_ransac(const std::vector<cv::Point>& points, EllipseResult& result,
                               int iterations = 120, float inlier_threshold = 3.0f,
                               float min_inlier_ratio = 0.45f, float max_axis_ratio = 6.0f) {
    const int count = static_cast<int>(points.size());
    if (count < 20) return false;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> pick(0, count - 1);
    cv::RotatedRect best;
    int best_inliers = 0;
    float best_error = 1e9f;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<cv::Point> sample;
        std::vector<int> indices;
        while (sample.size() < 5) {
            const int index = pick(rng);
            if (std::find(indices.begin(), indices.end(), index) != indices.end()) continue;
            indices.push_back(index);
            sample.push_back(points[index]);
        }
        cv::RotatedRect candidate;
        try { candidate = cv::fitEllipse(sample); }
        catch (...) { continue; }
        const float a = candidate.size.width * 0.5f;
        const float b = candidate.size.height * 0.5f;
        if (std::min(a, b) < 2.0f || std::max(a, b) / std::max(1e-3f, std::min(a, b)) > max_axis_ratio) continue;
        int inliers = 0;
        float error_sum = 0.0f;
        for (const auto& point : points) {
            const float error = radial_error(candidate, point);
            if (error <= inlier_threshold) { ++inliers; error_sum += error; }
        }
        const float mean_error = inliers > 0 ? error_sum / inliers : 1e9f;
        if (inliers > best_inliers || (inliers == best_inliers && mean_error < best_error)) {
            best = candidate;
            best_inliers = inliers;
            best_error = mean_error;
        }
    }
    const int required = std::max(20, static_cast<int>(std::ceil(min_inlier_ratio * count)));
    if (best_inliers < required) return false;
    std::vector<cv::Point> inlier_points;
    for (const auto& point : points) {
        if (radial_error(best, point) <= inlier_threshold) inlier_points.push_back(point);
    }
    if (inlier_points.size() < 5) return false;
    try { best = cv::fitEllipse(inlier_points); }
    catch (...) { return false; }
    float error_sum = 0.0f;
    for (const auto& point : inlier_points) error_sum += radial_error(best, point);
    result.ellipse = best;
    result.from_mask = true;
    result.inliers = static_cast<int>(inlier_points.size());
    result.mean_error_px = error_sum / inlier_points.size();
    return true;
}

static EllipseResult calculate_ellipse(const cv::Mat& frame, const object_detect_result& detection,
                                       const uint8_t* mask_data, bool force_box,
                                       float deviation_ratio) {
    EllipseResult result;
    const cv::Point2f box_center(detection.x + detection.w * 0.5f,
                                 detection.y + detection.h * 0.5f);
    if (mask_data != nullptr && !force_box) {
        const int x = std::max(0, detection.x);
        const int y = std::max(0, detection.y);
        const int width = std::min(detection.w, frame.cols - x);
        const int height = std::min(detection.h, frame.rows - y);
        if (width > 0 && height > 0) {
            cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, const_cast<uint8_t*>(mask_data));
            cv::Mat roi = full_mask(cv::Rect(x, y, width, height)).clone();
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<cv::Point> all_points;
            const double area_min = 0.005 * width * height;
            const float distance_limit = 0.80f * std::min(width, height);
            for (const auto& contour : contours) {
                if (contour.size() < 5 || std::abs(cv::contourArea(contour)) < area_min) continue;
                const cv::Moments moments = cv::moments(contour);
                if (std::abs(moments.m00) < 1e-6) continue;
                const float cx = static_cast<float>(moments.m10 / moments.m00);
                const float cy = static_cast<float>(moments.m01 / moments.m00);
                if (std::hypot(cx - (box_center.x - x), cy - (box_center.y - y)) <= distance_limit) {
                    all_points.insert(all_points.end(), contour.begin(), contour.end());
                }
            }
            if (all_points.size() >= 20) {
                std::vector<cv::Point> sampled;
                const int step = std::max(1, static_cast<int>(all_points.size()) / 150);
                for (int i = 0; i < static_cast<int>(all_points.size()) && sampled.size() < 150; i += step) {
                    sampled.push_back(all_points[i]);
                }
                if (fit_ellipse_ransac(sampled, result)) {
                    result.ellipse.center.x += x;
                    result.ellipse.center.y += y;
                    const float deviation = std::hypot(result.ellipse.center.x - box_center.x,
                                                       result.ellipse.center.y - box_center.y);
                    if (deviation <= std::min(detection.w, detection.h) * deviation_ratio) return result;
                }
            }
        }
    }
    const float side = static_cast<float>(std::min(detection.w, detection.h));
    result.ellipse = cv::RotatedRect(box_center, cv::Size2f(side, side), 0.0f);
    result.from_mask = false;
    result.inliers = 0;
    result.mean_error_px = std::numeric_limits<float>::quiet_NaN();
    return result;
}

static void draw_text(cv::Mat& image, const std::string& text, int y,
                      const cv::Scalar& color, double scale = 0.8) {
    cv::putText(image, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 5);
    cv::putText(image, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, 2);
}

static int best_index(const object_detect_result_list& result, int class_id) {
    int best = -1;
    float score = -1.0f;
    for (int i = 0; i < result.count; ++i) {
        if (result.results_box[i].cls_id == class_id && result.results_box[i].prop > score) {
            best = i;
            score = result.results_box[i].prop;
        }
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
        cv::Mat inference_frame = input.clone();
        image_process preprocessor(inference_frame);
        if (preprocessor.image_preprocessing(640, 640) != 0) return fail(key, "preprocess failed");
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
        std::vector<EllipseResult> ellipse_results;
        ellipse_results.reserve(result.count);
        const cv::Scalar colors[3] = {{0, 0, 255}, {0, 255, 0}, {255, 255, 0}};

        std::ofstream yolo_file(dirs.labels / (key + ".txt"));
        std::ofstream ellipse_file(dirs.ellipses / (key + ".txt"));
        std::ofstream pose_file(dirs.poses / (key + ".txt"));
        if (!yolo_file || !ellipse_file || !pose_file) return fail(key, "cannot create txt output");
        yolo_file << std::fixed << std::setprecision(8);
        ellipse_file << "# class_id center_x_px center_y_px major_axis_px minor_axis_px angle_deg confidence fit_source inliers mean_error_px\n"
                     << std::fixed << std::setprecision(6);

        for (int i = 0; i < result.count; ++i) {
            const auto& detection = result.results_box[i];
            const uint8_t* mask = nullptr;
            if (i < static_cast<int>(result.results_mask[0].each_of_mask.size()) &&
                result.results_mask[0].each_of_mask[i]) {
                mask = result.results_mask[0].each_of_mask[i].get();
            }
            const bool force_box = detection.cls_id == 2 && !args_.fit_hole_from_mask;
            EllipseResult ellipse = calculate_ellipse(input, detection, mask, force_box,
                                                       args_.ellipse_deviation_ratio);
            ellipse_results.push_back(ellipse);

            const double center_x = (detection.x + detection.w * 0.5) / input.cols;
            const double center_y = (detection.y + detection.h * 0.5) / input.rows;
            const double width = static_cast<double>(detection.w) / input.cols;
            const double height = static_cast<double>(detection.h) / input.rows;
            yolo_file << detection.cls_id << ' ' << center_x << ' ' << center_y << ' '
                      << width << ' ' << height << ' ' << detection.prop << '\n';

            const float major_axis = std::max(ellipse.ellipse.size.width, ellipse.ellipse.size.height);
            const float minor_axis = std::min(ellipse.ellipse.size.width, ellipse.ellipse.size.height);
            ellipse_file << detection.cls_id << ' ' << ellipse.ellipse.center.x << ' '
                         << ellipse.ellipse.center.y << ' ' << major_axis << ' ' << minor_axis << ' '
                         << ellipse.ellipse.angle << ' ' << detection.prop << ' '
                         << (ellipse.from_mask ? "mask" : "box") << ' ' << ellipse.inliers << ' '
                         << ellipse.mean_error_px << '\n';

            const int class_id = detection.cls_id;
            if (mask != nullptr && class_id >= 0 && class_id < 3) {
                cv::Mat full_mask(input.rows, input.cols, CV_8UC1, const_cast<uint8_t*>(mask));
                cv::Mat active = full_mask > 0;
                cv::Mat overlay = visualization.clone();
                overlay.setTo(colors[class_id], active);
                cv::addWeighted(overlay, 0.5, visualization, 0.5, 0.0, visualization);
            }
            cv::rectangle(visualization, {detection.x, detection.y, detection.w, detection.h}, {0, 0, 255}, 2);
            cv::ellipse(visualization, ellipse.ellipse, ellipse.from_mask ? cv::Scalar(0, 255, 0)
                                                                         : cv::Scalar(0, 255, 255), 2);
            cv::circle(visualization, ellipse.ellipse.center, 2, {0, 0, 255}, -1);
            const std::string label = "cls=" + std::to_string(class_id) + " conf=" +
                                      cv::format("%.3f", detection.prop);
            cv::putText(visualization, label, {detection.x, std::max(18, detection.y - 5)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 2);
        }
        save_and_draw_pose(result, ellipse_results, visualization, pose_file);
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
                            const std::vector<EllipseResult>& ellipses,
                            cv::Mat& visualization, std::ofstream& pose_file) {
        pose_file << "# valid reference_class ref_center_x_px ref_center_y_px hole_center_x_px hole_center_y_px "
                     "auto_yaw_deg auto_pitch_deg auto_roll_deg auto_tx_mm auto_ty_mm auto_tz_mm "
                     "fixed_yaw_deg fixed_pitch_deg fixed_roll_deg fixed_tx_mm fixed_ty_mm fixed_tz_mm "
                     "display_mode\n"
                  << std::fixed << std::setprecision(9);
        const int outer = best_index(result, 0);
        const int middle = best_index(result, 1);
        const int hole = best_index(result, 2);
        if (hole < 0 || (outer < 0 && middle < 0) || ellipses.size() < static_cast<size_t>(result.count)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            pose_file << "0 -1 " << nan << ' ' << nan << ' ' << nan << ' ' << nan;
            for (int i = 0; i < 12; ++i) pose_file << ' ' << nan;
            pose_file << " invalid\n";
            draw_text(visualization, "Pose: --", 190, {255, 255, 0}, 1.2);
            draw_text(visualization, "Dist: --", 240, {255, 255, 0}, 1.2);
            return;
        }
        int reference = -1;
        if (outer >= 0 && middle >= 0) {
            reference = ellipses[outer].from_mask ? outer : (ellipses[middle].from_mask ? middle : outer);
        } else {
            reference = outer >= 0 ? outer : middle;
        }
        const bool use_middle = result.results_box[reference].cls_id == 1;
        const cv::RotatedRect& target = ellipses[reference].ellipse;
        const cv::Point2f hole_center = ellipses[hole].ellipse.center;
        const Pose6D pose_auto = pose_estimator_.Solve(target, hole_center, use_middle, std::nullopt);
        const Pose6D pose_fixed = pose_estimator_.Solve(target, hole_center, use_middle, args_.fixed_distance_mm);
        const Pose6D& displayed = args_.display_fixed_pose ? pose_fixed : pose_auto;

        pose_file << "1 " << result.results_box[reference].cls_id << ' ' << target.center.x << ' '
                  << target.center.y << ' ' << hole_center.x << ' ' << hole_center.y << ' ';
        write_pose_values(pose_file, pose_auto);
        pose_file << ' ';
        write_pose_values(pose_file, pose_fixed);
        pose_file << ' ' << (args_.display_fixed_pose ? "fixed" : "auto") << '\n';

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
