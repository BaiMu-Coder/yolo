// 板端 RKNN 版：检测框七项误差 + 最外圈拟合椭圆 IoU 批量评估。
//
// 与 yolo_mask_bbox_error.py 保持相同核心口径：
//   1. YOLO 分割标签多边形的外接框作为检测误差真值；
//   2. 标签侧与预测侧分别选择外接框面积最大的目标，类别不参与选择；
//   3. 两个最大目标类别不一致时单独留图，并排除所有指标；
//   4. 检测误差和椭圆 IoU 共用上述唯一一对最外圈目标；
//   5. 最大预测实例复用生产 EllipseFitter 的 Mask -> Box 管线；
//   6. 按拟合外圈完整长轴 <77、77~311、>311 px 分组；
//   7. 保存全部检测/IoU 可视化、超阈值、总体误差 Top10、IoU 最低50；
//   8. 优先写 Excel 2003 XML 多工作表（Excel/WPS 可直接打开），失败时回退 TXT。

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common.hpp"
#include "ellipse_fitter.hpp"
#include "image_process.hpp"
#include "yolov8seg.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kMetricCount = 7;
const std::array<const char*, kMetricCount> kMetricNames = {
    "left_edge_error_px", "top_edge_error_px", "right_edge_error_px",
    "bottom_edge_error_px", "x_axis_mean_error_px",
    "y_axis_mean_error_px", "long_side_length_error_px"};

struct Args {
    fs::path model;
    fs::path images;
    fs::path labels;
    fs::path output;
    double threshold_px = 3.0;
    int class_id = -1;  // 兼容旧命令，当前选择逻辑忽略。
    bool match_prediction_class = false;  // 兼容旧命令，当前忽略。
    bool recursive = false;
    bool save_visualizations = true;
    bool force_outer_box = false;
    int outer_class_id = 0;  // 兼容旧命令，当前忽略。
    double small_major_px = 77.0;
    double large_major_px = 311.0;
    int top_k = 10;
    int lowest_iou_count = 50;
};

struct GroundTruth {
    int class_id = -1;
    std::vector<cv::Point2f> polygon;
    cv::Rect2f box;
};

struct EvaluationRow {
    fs::path image;
    std::string status = "not_evaluated";
    int gt_class = -1;
    int pred_class = -1;
    float confidence = 0.0f;
    float detection_iou = 0.0f;
    std::array<double, kMetricCount> errors{};
    bool all_within_threshold = false;

    std::string outer_status = "not_evaluated";
    std::string outer_source;
    std::string size_group;
    double outer_major_px = std::numeric_limits<double>::quiet_NaN();
    double ellipse_iou = std::numeric_limits<double>::quiet_NaN();
    cv::RotatedRect outer_ellipse;
};

struct MetricStat {
    std::string group;
    std::string range;
    std::string metric;
    int count = 0;
    double sample_percent = 0.0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double minimum = std::numeric_limits<double>::quiet_NaN();
    double maximum = std::numeric_limits<double>::quiet_NaN();
    double p95 = std::numeric_limits<double>::quiet_NaN();
    int pass_count = 0;
    double pass_percent = 0.0;
};

struct IouStat {
    std::string group;
    std::string range;
    int count = 0;
    double sample_percent = 0.0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double minimum = std::numeric_limits<double>::quiet_NaN();
    double maximum = std::numeric_limits<double>::quiet_NaN();
    double p95 = std::numeric_limits<double>::quiet_NaN();
};

using Table = std::vector<std::vector<std::string>>;

std::string value_after(int& index, int argc, char** argv, const char* name) {
    if (index + 1 >= argc)
        throw std::runtime_error(std::string("Missing value for ") + name);
    return argv[++index];
}

void print_help(const char* app) {
    std::cout
        << "Usage:\n  " << app
        << " --model best.rknn --images /data/images --labels /data/labels"
           " --output /data/evaluation\n\n"
        << "Options:\n"
        << "  --threshold PX             Detection pass threshold, default 3\n"
        << "  --class-id ID              Deprecated compatibility option; ignored\n"
        << "  --match-pred-class         Deprecated compatibility option; ignored\n"
        << "  --outer-class-id ID        Deprecated compatibility option; ignored\n"
        << "  --force-outer-box          Force outer box-inscribed circle\n"
        << "  --small-major PX           Small/medium boundary, default 77\n"
        << "  --large-major PX           Medium/large boundary, default 311\n"
        << "  --top-k N                  Largest detection-error images, default 10\n"
        << "  --lowest-iou N             Lowest ellipse-IoU images, default 50\n"
        << "  --recursive                Recursively search image directory\n"
        << "  --no-save-vis              Do not save per-image visualizations\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--model") args.model = value_after(i, argc, argv, "--model");
        else if (key == "--images") args.images = value_after(i, argc, argv, "--images");
        else if (key == "--labels") args.labels = value_after(i, argc, argv, "--labels");
        else if (key == "--output") args.output = value_after(i, argc, argv, "--output");
        else if (key == "--threshold")
            args.threshold_px = std::stod(value_after(i, argc, argv, "--threshold"));
        else if (key == "--class-id")
            args.class_id = std::stoi(value_after(i, argc, argv, "--class-id"));
        else if (key == "--outer-class-id")
            args.outer_class_id = std::stoi(value_after(i, argc, argv, "--outer-class-id"));
        else if (key == "--small-major")
            args.small_major_px = std::stod(value_after(i, argc, argv, "--small-major"));
        else if (key == "--large-major")
            args.large_major_px = std::stod(value_after(i, argc, argv, "--large-major"));
        else if (key == "--top-k")
            args.top_k = std::stoi(value_after(i, argc, argv, "--top-k"));
        else if (key == "--lowest-iou")
            args.lowest_iou_count = std::stoi(value_after(i, argc, argv, "--lowest-iou"));
        else if (key == "--match-pred-class") args.match_prediction_class = true;
        else if (key == "--force-outer-box") args.force_outer_box = true;
        else if (key == "--recursive") args.recursive = true;
        else if (key == "--no-save-vis") args.save_visualizations = false;
        else if (key == "--help" || key == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (args.model.empty() || args.images.empty() ||
        args.labels.empty() || args.output.empty())
        throw std::runtime_error("--model, --images, --labels and --output are required");
    if (args.threshold_px < 0.0 || args.small_major_px <= 0.0 ||
        args.large_major_px <= args.small_major_px ||
        args.top_k <= 0 || args.lowest_iou_count <= 0)
        throw std::runtime_error("Invalid threshold/group/ranking argument");
    return args;
}

bool is_image(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
           extension == ".bmp" || extension == ".tif" || extension == ".tiff" ||
           extension == ".webp";
}

std::vector<fs::path> list_images(const Args& args) {
    if (fs::is_regular_file(args.images)) return {args.images};
    if (!fs::is_directory(args.images))
        throw std::runtime_error("Image path does not exist: " + args.images.string());
    std::vector<fs::path> images;
    if (args.recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(args.images))
            if (entry.is_regular_file() && is_image(entry.path()))
                images.push_back(entry.path());
    } else {
        for (const auto& entry : fs::directory_iterator(args.images))
            if (entry.is_regular_file() && is_image(entry.path()))
                images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    return images;
}

fs::path relative_image_path(const fs::path& image, const Args& args) {
    if (fs::is_regular_file(args.images)) return image.filename();
    std::error_code error;
    fs::path relative = fs::relative(image, args.images, error);
    return error ? image.filename() : relative;
}

fs::path label_path_for(const fs::path& image, const Args& args) {
    fs::path relative = relative_image_path(image, args);
    relative.replace_extension(".txt");
    return args.labels / relative;
}

cv::Rect2f polygon_box(const std::vector<cv::Point2f>& polygon) {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& point : polygon) {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }
    return {min_x, min_y, std::max(0.0f, max_x - min_x),
            std::max(0.0f, max_y - min_y)};
}

std::vector<GroundTruth> read_labels(const fs::path& path, cv::Size size) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Missing label: " + path.string());
    std::vector<GroundTruth> targets;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream stream(line);
        GroundTruth target;
        if (!(stream >> target.class_id)) continue;
        std::vector<double> coordinates;
        double value = 0.0;
        while (stream >> value) coordinates.push_back(value);
        if (coordinates.size() < 6 || coordinates.size() % 2 != 0)
            throw std::runtime_error("Invalid polygon at " + path.string() +
                                     ":" + std::to_string(line_number));
        for (size_t i = 0; i < coordinates.size(); i += 2) {
            if (coordinates[i] < 0.0 || coordinates[i] > 1.0 ||
                coordinates[i + 1] < 0.0 || coordinates[i + 1] > 1.0)
                throw std::runtime_error("Normalized polygon coordinate out of range");
            target.polygon.emplace_back(
                static_cast<float>(coordinates[i] * size.width),
                static_cast<float>(coordinates[i + 1] * size.height));
        }
        target.box = polygon_box(target.polygon);
        targets.push_back(std::move(target));
    }
    if (targets.empty()) throw std::runtime_error("No valid target: " + path.string());
    return targets;
}

double box_iou(const cv::Rect2f& lhs, const cv::Rect2f& rhs) {
    const float x1 = std::max(lhs.x, rhs.x);
    const float y1 = std::max(lhs.y, rhs.y);
    const float x2 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float y2 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    const double intersection =
        std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const double union_area =
        static_cast<double>(lhs.area()) + rhs.area() - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

cv::Rect2f detection_box(const object_detect_result& detection) {
    return {static_cast<float>(detection.x), static_cast<float>(detection.y),
            static_cast<float>(detection.w), static_cast<float>(detection.h)};
}

struct Match {
    int gt = -1;
    int pred = -1;
    double iou = -1.0;
};

Match choose_largest_outer_pair(
    const std::vector<GroundTruth>& targets,
    const object_detect_result_list& detections) {
    Match selected;
    double largest_gt_area = -1.0;
    for (int index = 0; index < static_cast<int>(targets.size()); ++index) {
        const double area = std::max(0.0f, targets[index].box.area());
        if (area > largest_gt_area) {
            largest_gt_area = area;
            selected.gt = index;
        }
    }

    double largest_prediction_area = -1.0;
    float best_confidence = -1.0f;
    for (int index = 0; index < detections.count; ++index) {
        const auto& prediction = detections.results_box[index];
        const cv::Rect2f box = detection_box(prediction);
        const double area = std::max(0.0f, box.area());
        if (area > largest_prediction_area ||
            (std::abs(area - largest_prediction_area) < 1e-12 &&
             prediction.prop > best_confidence)) {
            largest_prediction_area = area;
            best_confidence = prediction.prop;
            selected.pred = index;
        }
    }

    if (selected.gt >= 0 && selected.pred >= 0)
        selected.iou = box_iou(
            targets[selected.gt].box,
            detection_box(detections.results_box[selected.pred]));
    return selected;
}

std::string size_group(double major, const Args& args) {
    if (major < args.small_major_px) return "small";
    if (major <= args.large_major_px) return "medium";
    return "large";
}

std::string group_range(const std::string& group, const Args& args) {
    std::ostringstream stream;
    if (group == "overall") return "all valid samples";
    if (group == "small") stream << "major < " << args.small_major_px << " px";
    else if (group == "medium")
        stream << args.small_major_px << " <= major <= "
               << args.large_major_px << " px";
    else stream << "major > " << args.large_major_px << " px";
    return stream.str();
}

std::array<double, kMetricCount> calculate_errors(
    const cv::Rect2f& gt, const cv::Rect2f& pred) {
    const double left = std::abs(pred.x - gt.x);
    const double top = std::abs(pred.y - gt.y);
    const double right = std::abs((pred.x + pred.width) - (gt.x + gt.width));
    const double bottom = std::abs((pred.y + pred.height) - (gt.y + gt.height));
    const double long_side = gt.width >= gt.height
                                 ? std::abs(pred.width - gt.width)
                                 : std::abs(pred.height - gt.height);
    return {left, top, right, bottom, (left + right) * 0.5,
            (top + bottom) * 0.5, long_side};
}

double ellipse_polygon_iou(cv::Size size,
                           const std::vector<cv::Point2f>& polygon,
                           const cv::RotatedRect& ellipse,
                           cv::Mat* gt_mask_out = nullptr,
                           cv::Mat* ellipse_mask_out = nullptr) {
    cv::Mat gt_mask = cv::Mat::zeros(size, CV_8UC1);
    cv::Mat ellipse_mask = cv::Mat::zeros(size, CV_8UC1);
    std::vector<cv::Point> integer_polygon;
    integer_polygon.reserve(polygon.size());
    for (const auto& point : polygon)
        integer_polygon.emplace_back(cvRound(point.x), cvRound(point.y));
    cv::fillPoly(gt_mask, std::vector<std::vector<cv::Point>>{integer_polygon},
                 cv::Scalar(255));
    cv::ellipse(ellipse_mask, ellipse, cv::Scalar(255), cv::FILLED, cv::LINE_8);
    cv::Mat intersection, union_mask;
    cv::bitwise_and(gt_mask, ellipse_mask, intersection);
    cv::bitwise_or(gt_mask, ellipse_mask, union_mask);
    const int union_area = cv::countNonZero(union_mask);
    if (gt_mask_out) *gt_mask_out = gt_mask;
    if (ellipse_mask_out) *ellipse_mask_out = ellipse_mask;
    return union_area > 0
               ? static_cast<double>(cv::countNonZero(intersection)) / union_area
               : 0.0;
}

bool run_inference(yolov8seg& model, const cv::Mat& image,
                   object_detect_result_list& result, std::string& error) {
    cv::Mat input = image.clone();
    image_process preprocessor(input);
    const cv::Size model_size = model.model_input_size();
    if (preprocessor.image_preprocessing(model_size.width, model_size.height) != 0) {
        error = "preprocess_error";
        return false;
    }
    int bytes = 0;
    uint8_t* data = preprocessor.get_image_buffer(&bytes);
    if (!data || model.set_input_data(data, bytes) != RKNN_SUCC ||
        model.rknn_model_inference() != RKNN_SUCC ||
        model.get_output_data() != RKNN_SUCC) {
        error = "inference_error";
        return false;
    }
    letterbox transform = preprocessor.get_letterbox();
    const int status = model.post_process(result, transform);
    model.release_output_data();
    if (status != RKNN_SUCC) {
        error = "postprocess_error";
        return false;
    }
    return true;
}

fs::path output_image_path(const fs::path& root, const fs::path& image,
                           const Args& args) {
    return root / relative_image_path(image, args);
}

bool save_image(const fs::path& path, const cv::Mat& image) {
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    return !image.empty() && cv::imwrite(path.string(), image);
}

void draw_panel(cv::Mat& image, const std::vector<std::string>& lines,
                const cv::Scalar& color = {255, 255, 255}) {
    if (lines.empty()) return;
    const int panel_height = std::min(
        image.rows, 16 + static_cast<int>(lines.size()) * 25);
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, {5, 5}, {std::min(image.cols - 5, 720), panel_height},
                  {0, 0, 0}, cv::FILLED);
    cv::addWeighted(overlay, 0.70, image, 0.30, 0.0, image);
    for (size_t i = 0; i < lines.size(); ++i)
        cv::putText(image, lines[i], {14, 27 + static_cast<int>(i) * 25},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv::LINE_AA);
}

cv::Mat draw_detection_visual(const cv::Mat& image, const GroundTruth* gt,
                              const object_detect_result* pred,
                              const EvaluationRow& row, const Args& args) {
    cv::Mat visual = image.clone();
    if (gt) {
        std::vector<cv::Point> polygon;
        for (const auto& point : gt->polygon)
            polygon.emplace_back(cvRound(point.x), cvRound(point.y));
        cv::polylines(visual, std::vector<std::vector<cv::Point>>{polygon},
                      true, {0, 255, 255}, 2);
        cv::rectangle(visual, gt->box, {0, 255, 0}, 2);
    }
    if (pred) cv::rectangle(visual, detection_box(*pred), {0, 0, 255}, 2);
    std::vector<std::string> lines{"status=" + row.status};
    if (pred) {
        std::ostringstream header;
        header << std::fixed << std::setprecision(3)
               << "GT cls=" << row.gt_class << " pred cls=" << row.pred_class
               << " conf=" << row.confidence << " box IoU=" << row.detection_iou;
        lines.push_back(header.str());
    }
    if (row.status == "ok") {
        for (int i = 0; i < kMetricCount; ++i) {
            std::ostringstream item;
            item << kMetricNames[i] << '=' << std::fixed << std::setprecision(2)
                 << row.errors[i] << " px"
                 << (row.errors[i] <= args.threshold_px ? " PASS" : " FAIL");
            lines.push_back(item.str());
        }
    }
    draw_panel(visual, lines, row.status == "ok" ? cv::Scalar(255, 255, 255)
                                                  : cv::Scalar(0, 165, 255));
    return visual;
}

cv::Mat draw_iou_visual(const cv::Mat& image,
                        const std::vector<cv::Point2f>* polygon,
                        const EvaluationRow& row) {
    cv::Mat visual = image.clone();
    if (row.outer_status != "ok" || !polygon) {
        draw_panel(visual, {"outer ellipse status=" + row.outer_status},
                   {0, 165, 255});
        return visual;
    }
    cv::Mat gt_mask, ellipse_mask;
    ellipse_polygon_iou(image.size(), *polygon, row.outer_ellipse,
                        &gt_mask, &ellipse_mask);
    cv::Mat intersection, gt_only, ellipse_only;
    cv::bitwise_and(gt_mask, ellipse_mask, intersection);
    cv::subtract(gt_mask, intersection, gt_only);
    cv::subtract(ellipse_mask, intersection, ellipse_only);
    cv::Mat overlay = visual.clone();
    overlay.setTo(cv::Scalar(0, 180, 0), gt_only);
    overlay.setTo(cv::Scalar(0, 0, 220), ellipse_only);
    overlay.setTo(cv::Scalar(0, 220, 220), intersection);
    cv::addWeighted(overlay, 0.36, visual, 0.64, 0.0, visual);
    std::vector<cv::Point> integer_polygon;
    for (const auto& point : *polygon)
        integer_polygon.emplace_back(cvRound(point.x), cvRound(point.y));
    cv::polylines(visual, std::vector<std::vector<cv::Point>>{integer_polygon},
                  true, {0, 255, 0}, 3);
    cv::ellipse(visual, row.outer_ellipse, {0, 0, 255}, 3);
    std::ostringstream line;
    line << std::fixed << std::setprecision(2)
         << "IoU=" << row.ellipse_iou * 100.0 << "% major="
         << row.outer_major_px << " px group=" << row.size_group
         << " source=" << row.outer_source;
    draw_panel(visual, {line.str()});
    return visual;
}

double percentile(std::vector<double> values, double percent) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * percent / 100.0;
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper) return values[lower];
    return values[lower] * (upper - position) +
           values[upper] * (position - lower);
}

std::vector<MetricStat> detection_statistics(
    const std::vector<EvaluationRow>& rows, const Args& args) {
    const int valid_detection = static_cast<int>(std::count_if(
        rows.begin(), rows.end(), [](const auto& row) { return row.status == "ok"; }));
    const int groupable = static_cast<int>(std::count_if(
        rows.begin(), rows.end(), [](const auto& row) {
            return row.status == "ok" && row.outer_status == "ok";
        }));
    std::vector<MetricStat> output;
    for (const std::string group : {"overall", "small", "medium", "large"}) {
        for (int metric = 0; metric < kMetricCount; ++metric) {
            std::vector<double> values;
            for (const auto& row : rows) {
                if (row.status != "ok") continue;
                if (group != "overall" &&
                    (row.outer_status != "ok" || row.size_group != group))
                    continue;
                values.push_back(row.errors[metric]);
            }
            MetricStat stat;
            stat.group = group;
            stat.range = group_range(group, args);
            stat.metric = kMetricNames[metric];
            stat.count = static_cast<int>(values.size());
            const int denominator = group == "overall" ? valid_detection : groupable;
            stat.sample_percent =
                denominator > 0 ? stat.count * 100.0 / denominator : 0.0;
            if (!values.empty()) {
                stat.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                            values.size();
                stat.minimum = *std::min_element(values.begin(), values.end());
                stat.maximum = *std::max_element(values.begin(), values.end());
                stat.p95 = percentile(values, 95.0);
                stat.pass_count = static_cast<int>(std::count_if(
                    values.begin(), values.end(),
                    [&](double value) { return value <= args.threshold_px; }));
                stat.pass_percent = stat.pass_count * 100.0 / values.size();
            }
            output.push_back(stat);
        }
    }
    return output;
}

std::vector<IouStat> iou_statistics(
    const std::vector<EvaluationRow>& rows, const Args& args) {
    const int valid = static_cast<int>(std::count_if(
        rows.begin(), rows.end(),
        [](const auto& row) { return row.outer_status == "ok"; }));
    std::vector<IouStat> output;
    for (const std::string group : {"overall", "small", "medium", "large"}) {
        std::vector<double> values;
        for (const auto& row : rows) {
            if (row.outer_status != "ok") continue;
            if (group != "overall" && row.size_group != group) continue;
            values.push_back(row.ellipse_iou);
        }
        IouStat stat;
        stat.group = group;
        stat.range = group_range(group, args);
        stat.count = static_cast<int>(values.size());
        stat.sample_percent = valid > 0 ? stat.count * 100.0 / valid : 0.0;
        if (!values.empty()) {
            stat.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        values.size();
            stat.minimum = *std::min_element(values.begin(), values.end());
            stat.maximum = *std::max_element(values.begin(), values.end());
            stat.p95 = percentile(values, 95.0);
        }
        output.push_back(stat);
    }
    return output;
}

std::string number(double value, int precision = 6) {
    if (!std::isfinite(value)) return "";
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string xml_escape(const std::string& text) {
    std::string output;
    output.reserve(text.size());
    for (char value : text) {
        if (value == '&') output += "&amp;";
        else if (value == '<') output += "&lt;";
        else if (value == '>') output += "&gt;";
        else if (value == '"') output += "&quot;";
        else if (value == '\'') output += "&apos;";
        else output += value;
    }
    return output;
}

bool is_numeric_cell(const std::string& text) {
    if (text.empty()) return false;
    char* end = nullptr;
    std::strtod(text.c_str(), &end);
    return end != text.c_str() && end != nullptr && *end == '\0';
}

bool write_excel_xml(
    const fs::path& path,
    const std::vector<std::pair<std::string, Table>>& sheets) {
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path);
    if (!output) return false;
    output << "<?xml version=\"1.0\"?>\n"
           << "<?mso-application progid=\"Excel.Sheet\"?>\n"
           << "<Workbook xmlns=\"urn:schemas-microsoft-com:office:spreadsheet\" "
              "xmlns:ss=\"urn:schemas-microsoft-com:office:spreadsheet\">\n"
           << "<Styles><Style ss:ID=\"Header\"><Font ss:Bold=\"1\" "
              "ss:Color=\"#FFFFFF\"/><Interior ss:Color=\"#1F4E78\" "
              "ss:Pattern=\"Solid\"/></Style></Styles>\n";
    for (const auto& [name, table] : sheets) {
        output << "<Worksheet ss:Name=\"" << xml_escape(name)
               << "\"><Table>\n";
        for (size_t row_index = 0; row_index < table.size(); ++row_index) {
            output << "<Row>\n";
            for (const auto& cell : table[row_index]) {
                const bool numeric = row_index > 0 && is_numeric_cell(cell);
                output << "<Cell"
                       << (row_index == 0 ? " ss:StyleID=\"Header\"" : "")
                       << "><Data ss:Type=\""
                       << (numeric ? "Number" : "String") << "\">"
                       << xml_escape(cell)
                       << "</Data></Cell>\n";
            }
            output << "</Row>\n";
        }
        output << "</Table></Worksheet>\n";
    }
    output << "</Workbook>\n";
    return static_cast<bool>(output);
}

bool write_txt_table(const fs::path& path, const Table& table) {
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path);
    if (!output) return false;
    for (const auto& row : table) {
        for (size_t index = 0; index < row.size(); ++index) {
            if (index) output << '\t';
            output << row[index];
        }
        output << '\n';
    }
    return static_cast<bool>(output);
}

Table detection_stat_table(const std::vector<MetricStat>& stats,
                           const std::string& group_filter) {
    Table table{{"group", "major_axis_range", "count", "sample_percent",
                 "metric", "mean_px", "min_px", "max_px", "p95_px",
                 "pass_count", "pass_percent"}};
    for (const auto& stat : stats) {
        if (group_filter == "overall" && stat.group != "overall") continue;
        if (group_filter == "grouped" && stat.group == "overall") continue;
        table.push_back(
            {stat.group, stat.range, std::to_string(stat.count),
             number(stat.sample_percent), stat.metric, number(stat.mean),
             number(stat.minimum), number(stat.maximum), number(stat.p95),
             std::to_string(stat.pass_count), number(stat.pass_percent)});
    }
    return table;
}

Table per_image_detection_table(const std::vector<EvaluationRow>& rows) {
    Table table{{"image", "status", "gt_class", "pred_class",
                 "outer_group", "outer_major_px",
                 kMetricNames[0], kMetricNames[1], kMetricNames[2],
                 kMetricNames[3], kMetricNames[4], kMetricNames[5],
                 kMetricNames[6], "max_error_px", "max_error_metric",
                 "all_within_threshold"}};
    for (const auto& row : rows) {
        std::vector<std::string> record{
            row.image.string(), row.status, std::to_string(row.gt_class),
            row.pred_class >= 0 ? std::to_string(row.pred_class) : "",
            row.size_group,
            number(row.outer_major_px)};
        int maximum_index = 0;
        for (int i = 0; i < kMetricCount; ++i) {
            record.push_back(row.status == "ok" ? number(row.errors[i]) : "");
            if (row.errors[i] > row.errors[maximum_index]) maximum_index = i;
        }
        record.push_back(row.status == "ok" ? number(row.errors[maximum_index]) : "");
        record.push_back(row.status == "ok" ? kMetricNames[maximum_index] : "");
        record.push_back(row.status == "ok"
                             ? (row.all_within_threshold ? "1" : "0")
                             : "");
        table.push_back(std::move(record));
    }
    return table;
}

Table iou_stat_table(const std::vector<IouStat>& stats) {
    Table table{{"group", "major_axis_range", "count", "sample_percent",
                 "mean_iou_percent", "min_iou_percent",
                 "max_iou_percent", "p95_iou_percent"}};
    for (const auto& stat : stats)
        table.push_back(
            {stat.group, stat.range, std::to_string(stat.count),
             number(stat.sample_percent), number(stat.mean * 100.0),
             number(stat.minimum * 100.0), number(stat.maximum * 100.0),
             number(stat.p95 * 100.0)});
    return table;
}

Table per_image_iou_table(const std::vector<EvaluationRow>& rows) {
    Table table{{"image", "status", "gt_class", "pred_class",
                 "major_axis_px", "group", "source",
                 "iou", "iou_percent"}};
    for (const auto& row : rows)
        table.push_back({row.image.string(), row.outer_status,
                         std::to_string(row.gt_class),
                         row.pred_class >= 0 ? std::to_string(row.pred_class) : "",
                         number(row.outer_major_px), row.size_group,
                         row.outer_source, number(row.ellipse_iou),
                         number(row.ellipse_iou * 100.0)});
    return table;
}

Table outer_status_table(const std::vector<EvaluationRow>& rows) {
    std::vector<std::pair<std::string, int>> counts;
    for (const auto& row : rows) {
        auto iterator = std::find_if(
            counts.begin(), counts.end(),
            [&](const auto& item) { return item.first == row.outer_status; });
        if (iterator == counts.end()) counts.emplace_back(row.outer_status, 1);
        else ++iterator->second;
    }
    std::sort(counts.begin(), counts.end());
    Table table{{"outer_ellipse_status", "count", "all_image_percent"}};
    for (const auto& [status, count] : counts)
        table.push_back(
            {status, std::to_string(count),
             number(rows.empty() ? 0.0 : count * 100.0 / rows.size())});
    return table;
}

void draw_detection_chart(const fs::path& path,
                          const std::vector<MetricStat>& stats) {
    const std::array<std::string, 4> groups{"overall", "small", "medium", "large"};
    cv::Mat image(620, 1500, CV_8UC3, cv::Scalar(248, 248, 248));
    cv::putText(image, "Detection errors by fitted outer-ellipse major axis",
                {50, 48}, cv::FONT_HERSHEY_SIMPLEX, 0.9, {20, 20, 20}, 2);
    const int left = 410, top = 90, row_height = 65, column_width = 260;
    for (int group = 0; group < 4; ++group)
        cv::putText(image, groups[group], {left + group * column_width + 20, 78},
                    cv::FONT_HERSHEY_SIMPLEX, 0.58, {30, 30, 30}, 2);
    for (int metric = 0; metric < kMetricCount; ++metric) {
        const int y = top + metric * row_height;
        cv::rectangle(image, {30, y}, {1470, y + row_height},
                      metric % 2 ? cv::Scalar(250, 250, 250)
                                 : cv::Scalar(232, 238, 242),
                      cv::FILLED);
        cv::putText(image, kMetricNames[metric], {45, y + 39},
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, {30, 30, 30}, 1);
        for (int group = 0; group < 4; ++group) {
            const auto& stat = stats[group * kMetricCount + metric];
            const std::string text = stat.count == 0
                ? "-- (n=0)"
                : number(stat.mean, 3) + "px (n=" +
                      std::to_string(stat.count) + ")";
            cv::putText(image, text,
                        {left + group * column_width + 18, y + 39},
                        cv::FONT_HERSHEY_SIMPLEX, 0.48, {30, 30, 30}, 1);
        }
    }
    save_image(path, image);
}

void draw_iou_chart(const fs::path& path, const std::vector<IouStat>& stats) {
    cv::Mat image(860, 1280, CV_8UC3, cv::Scalar(248, 248, 248));
    const int left = 120, right = 1210, top = 110, bottom = 600;
    cv::putText(image, "Outer label vs fitted ellipse IoU",
                {250, 55}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {20, 20, 20}, 2);
    for (int tick = 0; tick <= 100; tick += 10) {
        const int y = bottom - (bottom - top) * tick / 100;
        cv::line(image, {left, y}, {right, y}, {215, 215, 215}, 1);
        cv::putText(image, std::to_string(tick) + "%", {55, y + 5},
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, {60, 60, 60}, 1);
    }
    const std::array<cv::Scalar, 4> colors{
        cv::Scalar(150, 120, 210), cv::Scalar(90, 170, 255),
        cv::Scalar(80, 190, 80), cv::Scalar(210, 130, 70)};
    const int slot = (right - left) / 4;
    for (int index = 0; index < 4; ++index) {
        const auto& stat = stats[index];
        const int center = left + slot * index + slot / 2;
        const int width = slot * 42 / 100;
        const double mean = std::isfinite(stat.mean) ? stat.mean * 100.0 : 0.0;
        const double minimum =
            std::isfinite(stat.minimum) ? stat.minimum * 100.0 : 0.0;
        const double maximum =
            std::isfinite(stat.maximum) ? stat.maximum * 100.0 : 0.0;
        const double p95 = std::isfinite(stat.p95) ? stat.p95 * 100.0 : 0.0;
        const auto y_for = [&](double value) {
            return bottom - cvRound((bottom - top) * value / 100.0);
        };
        cv::rectangle(image, {center - width / 2, y_for(mean)},
                      {center + width / 2, bottom}, colors[index], cv::FILLED);
        const int whisker_x = center + width / 2 + 18;
        cv::line(image, {whisker_x, y_for(minimum)},
                 {whisker_x, y_for(maximum)}, {70, 70, 70}, 2);
        cv::line(image, {center - width / 2 - 8, y_for(p95)},
                 {center + width / 2 + 8, y_for(p95)}, {0, 120, 255}, 3);
        cv::putText(image, "mean " + number(mean, 2) + "%",
                    {center - 90, std::max(top + 22, y_for(mean) - 15)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {30, 30, 30}, 1);
        cv::putText(image, stat.group + " n=" + std::to_string(stat.count),
                    {center - 105, bottom + 36}, cv::FONT_HERSHEY_SIMPLEX,
                    0.52, {30, 30, 30}, 1);
        cv::putText(image, "P95 " + number(p95, 2) + "%",
                    {center - 105, bottom + 70}, cv::FONT_HERSHEY_SIMPLEX,
                    0.48, {30, 30, 30}, 1);
        cv::putText(image, "min/max " + number(minimum, 1) + "/" +
                                 number(maximum, 1) + "%",
                    {center - 105, bottom + 104}, cv::FONT_HERSHEY_SIMPLEX,
                    0.43, {30, 30, 30}, 1);
        cv::putText(image, "share " + number(stat.sample_percent, 1) + "%",
                    {center - 105, bottom + 138}, cv::FONT_HERSHEY_SIMPLEX,
                    0.43, {30, 30, 30}, 1);
    }
    cv::putText(image, "Bar=mean | orange=P95 | whisker=min-max",
                {left, 825}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {40, 40, 40}, 1);
    save_image(path, image);
}

double row_max_error(const EvaluationRow& row) {
    return *std::max_element(row.errors.begin(), row.errors.end());
}

int row_max_error_index(const EvaluationRow& row) {
    return static_cast<int>(
        std::distance(row.errors.begin(),
                      std::max_element(row.errors.begin(), row.errors.end())));
}

void save_ranked_visualizations(const std::vector<EvaluationRow>& rows,
                                const Args& args,
                                const fs::path& detection_visual_dir,
                                const fs::path& iou_visual_dir,
                                const fs::path& detection_top_dir,
                                const fs::path& lowest_iou_dir,
                                Table& top_table, Table& lowest_table) {
    std::vector<const EvaluationRow*> detection_rows;
    std::vector<const EvaluationRow*> iou_rows;
    for (const auto& row : rows) {
        if (row.status == "ok") detection_rows.push_back(&row);
        if (row.outer_status == "ok") iou_rows.push_back(&row);
    }
    std::sort(detection_rows.begin(), detection_rows.end(),
              [](const auto* lhs, const auto* rhs) {
                  return row_max_error(*lhs) > row_max_error(*rhs);
              });
    std::sort(iou_rows.begin(), iou_rows.end(),
              [](const auto* lhs, const auto* rhs) {
                  return lhs->ellipse_iou < rhs->ellipse_iou;
              });
    top_table = {{"rank", "image", "max_error_px",
                  "max_error_metric", "visualization"}};
    lowest_table = {{"rank", "image", "iou_percent", "major_axis_px",
                     "group", "source", "visualization"}};
    for (int rank = 0;
         rank < std::min(args.top_k, static_cast<int>(detection_rows.size())); ++rank) {
        const auto& row = *detection_rows[rank];
        const fs::path source = output_image_path(
            detection_visual_dir, row.image, args);
        const fs::path destination =
            detection_top_dir / ("rank_" + cv::format("%02d", rank + 1) + ".jpg");
        if (args.save_visualizations) {
            cv::Mat visual = cv::imread(source.string());
            if (!visual.empty()) {
                draw_panel(visual,
                           {"TOP MAX ERROR rank " + std::to_string(rank + 1) +
                            "/" + std::to_string(args.top_k) + " = " +
                            number(row_max_error(row), 2) + " px"},
                           {0, 165, 255});
                save_image(destination, visual);
            }
        }
        top_table.push_back({std::to_string(rank + 1), row.image.string(),
                             number(row_max_error(row)),
                             kMetricNames[row_max_error_index(row)],
                             args.save_visualizations
                                 ? destination.string() : ""});
    }
    for (int rank = 0;
         rank < std::min(args.lowest_iou_count, static_cast<int>(iou_rows.size()));
         ++rank) {
        const auto& row = *iou_rows[rank];
        const fs::path source = output_image_path(iou_visual_dir, row.image, args);
        const fs::path destination =
            lowest_iou_dir / ("rank_" + cv::format("%02d", rank + 1) + ".jpg");
        if (args.save_visualizations) {
            cv::Mat visual = cv::imread(source.string());
            if (!visual.empty()) {
                draw_panel(visual,
                           {"LOWEST IoU rank " + std::to_string(rank + 1) +
                            "/" + std::to_string(args.lowest_iou_count) + " = " +
                            number(row.ellipse_iou * 100.0, 2) + "%"},
                           {0, 165, 255});
                save_image(destination, visual);
            }
        }
        lowest_table.push_back(
            {std::to_string(rank + 1), row.image.string(),
             number(row.ellipse_iou * 100.0), number(row.outer_major_px),
             row.size_group, row.outer_source,
             args.save_visualizations ? destination.string() : ""});
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::vector<fs::path> images = list_images(args);
        if (images.empty()) throw std::runtime_error("No supported images found");
        fs::create_directories(args.output);

        const fs::path detection_dir = args.output / "detection";
        const fs::path iou_dir = args.output / "ellipse_iou";
        const fs::path class_mismatch_dir = args.output / "class_mismatch";
        const fs::path detection_visual_dir = detection_dir / "visualizations";
        const fs::path over_threshold_dir = detection_dir / "over_threshold";
        const fs::path detection_top_dir = detection_dir / "top10_max_error";
        const fs::path iou_visual_dir = iou_dir / "visualizations";
        const fs::path lowest_iou_dir = iou_dir / "lowest_50";
        fs::create_directories(detection_dir);
        fs::create_directories(iou_dir);
        if (args.save_visualizations)
            fs::create_directories(class_mismatch_dir);

        yolov8seg model(args.model.string());
        const int init_status = model.init();
        if (init_status != RKNN_SUCC)
            throw std::runtime_error("RKNN model init failed: " +
                                     std::to_string(init_status));
        EllipseFitConfig fit_config;
        fit_config.enable_edge_fallback = false;
        const EllipseFitter fitter(fit_config);
        std::vector<EvaluationRow> rows;
        rows.reserve(images.size());

        for (size_t index = 0; index < images.size(); ++index) {
            const fs::path& image_path = images[index];
            std::cout << '[' << index + 1 << '/' << images.size() << "] "
                      << image_path.filename() << '\n';
            EvaluationRow row;
            row.image = image_path;
            cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                row.status = row.outer_status = "image_read_error";
                rows.push_back(row);
                continue;
            }

            std::vector<GroundTruth> targets;
            try {
                targets = read_labels(label_path_for(image_path, args), image.size());
            } catch (const std::exception& error) {
                row.status = row.outer_status = "label_error";
                if (args.save_visualizations) {
                    cv::Mat visual = image.clone();
                    draw_panel(visual, {error.what()}, {0, 165, 255});
                    save_image(output_image_path(
                        detection_visual_dir, image_path, args), visual);
                    save_image(output_image_path(
                        iou_visual_dir, image_path, args), visual);
                }
                rows.push_back(row);
                continue;
            }

            object_detect_result_list detections;
            std::string inference_error;
            if (!run_inference(model, image, detections, inference_error)) {
                row.status = row.outer_status = inference_error;
                if (args.save_visualizations) {
                    cv::Mat visual = image.clone();
                    draw_panel(visual, {inference_error}, {0, 165, 255});
                    save_image(output_image_path(
                        detection_visual_dir, image_path, args), visual);
                    save_image(output_image_path(
                        iou_visual_dir, image_path, args), visual);
                }
                rows.push_back(row);
                continue;
            }

            // 只做一次几何选择：标签最大外框 + 预测最大外框。
            // 检测误差、椭圆拟合及 IoU 必须共用这一对；类别随后用于一致性门禁。
            const Match largest_outer =
                choose_largest_outer_pair(targets, detections);
            const GroundTruth* selected_gt =
                largest_outer.gt >= 0 ? &targets[largest_outer.gt] : nullptr;
            const object_detect_result* selected_prediction =
                largest_outer.pred >= 0
                    ? &detections.results_box[largest_outer.pred]
                    : nullptr;
            // 标签存在但模型漏检时也保留最大标签的类别，便于逐图结果追溯。
            row.gt_class = selected_gt->class_id;
            if (selected_prediction &&
                selected_prediction->cls_id != selected_gt->class_id) {
                // 最大面积目标类别不一致通常意味着选错实例。只保存异常图和
                // 状态记录，不计算检测误差、椭圆 IoU、分组或排行。
                row.status = row.outer_status = "class_mismatch";
                row.pred_class = selected_prediction->cls_id;
                row.confidence = selected_prediction->prop;
                row.detection_iou = static_cast<float>(largest_outer.iou);
                if (args.save_visualizations) {
                    const cv::Mat mismatch_visual = draw_detection_visual(
                        image, selected_gt, selected_prediction, row, args);
                    save_image(output_image_path(
                        class_mismatch_dir, image_path, args),
                        mismatch_visual);
                }
                rows.push_back(std::move(row));
                continue;
            }
            if (largest_outer.pred < 0) {
                row.status = "no_detection";
            } else {
                row.status = "ok";
                row.pred_class = selected_prediction->cls_id;
                row.confidence = selected_prediction->prop;
                row.detection_iou = static_cast<float>(largest_outer.iou);
                row.errors = calculate_errors(
                    selected_gt->box, detection_box(*selected_prediction));
                row.all_within_threshold = std::all_of(
                    row.errors.begin(), row.errors.end(),
                    [&](double value) { return value <= args.threshold_px; });
            }

            const GroundTruth* outer_gt = selected_gt;
            if (largest_outer.pred < 0) {
                row.outer_status = "no_outer_detection";
            } else {
                const int prediction_index = largest_outer.pred;
                const auto& prediction =
                    detections.results_box[prediction_index];
                const auto& masks = detections.results_mask[0].each_of_mask;
                const auto& probabilities =
                    detections.results_mask[0].each_of_mask_probability;
                const uint8_t* mask =
                    prediction_index < static_cast<int>(masks.size()) &&
                            masks[prediction_index]
                        ? masks[prediction_index].get()
                        : nullptr;
                const uint8_t* probability =
                    prediction_index < static_cast<int>(probabilities.size()) &&
                            probabilities[prediction_index]
                        ? probabilities[prediction_index].get()
                        : nullptr;
                const EllipseFitMode mode = args.force_outer_box
                    ? EllipseFitMode::ForceBox
                    : EllipseFitMode::PreferMaskNoEdge;
                const EllipseFitResult fit = fitter.Fit(
                    image, cv::Rect(prediction.x, prediction.y,
                                    prediction.w, prediction.h),
                    mask, mode, probability);
                if (!fit.valid) {
                    row.outer_status = "invalid_outer_ellipse";
                } else {
                    row.outer_status = "ok";
                    row.outer_source = EllipseSourceName(fit.source);
                    row.outer_ellipse = fit.ellipse;
                    row.outer_major_px =
                        std::max(fit.ellipse.size.width, fit.ellipse.size.height);
                    row.size_group = size_group(row.outer_major_px, args);
                    row.ellipse_iou = ellipse_polygon_iou(
                        image.size(), outer_gt->polygon, fit.ellipse);
                }
            }

            if (args.save_visualizations) {
                const cv::Mat detection_visual = draw_detection_visual(
                    image, selected_gt, selected_prediction, row, args);
                save_image(output_image_path(
                    detection_visual_dir, image_path, args), detection_visual);
                if (row.status == "ok" && !row.all_within_threshold)
                    save_image(output_image_path(
                        over_threshold_dir, image_path, args), detection_visual);
                const cv::Mat iou_visual = draw_iou_visual(
                    image, outer_gt ? &outer_gt->polygon : nullptr, row);
                save_image(output_image_path(
                    iou_visual_dir, image_path, args), iou_visual);
            }
            rows.push_back(std::move(row));
        }

        const auto detection_stats = detection_statistics(rows, args);
        const auto iou_stats = iou_statistics(rows, args);
        draw_detection_chart(detection_dir / "grouped_errors.png",
                             detection_stats);
        draw_iou_chart(iou_dir / "overall_and_grouped.png", iou_stats);

        Table top_table{{"rank", "image", "max_error_px",
                         "max_error_metric", "visualization"}};
        Table lowest_table{{"rank", "image", "iou_percent", "major_axis_px",
                            "group", "source", "visualization"}};
        save_ranked_visualizations(
            rows, args, detection_visual_dir, iou_visual_dir,
            detection_top_dir, lowest_iou_dir, top_table, lowest_table);

        const std::vector<std::pair<std::string, Table>> detection_sheets{
            {"OverallErrors", detection_stat_table(detection_stats, "overall")},
            {"GroupedErrors", detection_stat_table(detection_stats, "grouped")},
            {"PerImage", per_image_detection_table(rows)},
            {"Top10", top_table}};
        const std::vector<std::pair<std::string, Table>> iou_sheets{
            {"OverallAndGroups", iou_stat_table(iou_stats)},
            {"EvaluationStatus", outer_status_table(rows)},
            {"PerImageIoU", per_image_iou_table(rows)},
            {"Lowest50", lowest_table}};

        const fs::path detection_book =
            detection_dir / "detection_statistics.xml";
        const fs::path iou_book = iou_dir / "ellipse_iou_statistics.xml";
        const bool excel_ok =
            write_excel_xml(detection_book, detection_sheets) &&
            write_excel_xml(iou_book, iou_sheets);
        if (!excel_ok) {
            std::cerr << "[WARNING] Excel XML write failed; fallback to TXT\n";
            write_txt_table(detection_dir / "overall_errors.txt",
                            detection_sheets[0].second);
            write_txt_table(detection_dir / "grouped_errors.txt",
                            detection_sheets[1].second);
            write_txt_table(detection_dir / "per_image_errors.txt",
                            detection_sheets[2].second);
            write_txt_table(detection_top_dir / "top10.txt", top_table);
            write_txt_table(iou_dir / "overall_and_grouped.txt",
                            iou_sheets[0].second);
            write_txt_table(iou_dir / "evaluation_status.txt",
                            iou_sheets[1].second);
            write_txt_table(iou_dir / "per_image_iou.txt",
                            iou_sheets[2].second);
            write_txt_table(lowest_iou_dir / "lowest_iou.txt", lowest_table);
        }

        const int detection_ok = static_cast<int>(std::count_if(
            rows.begin(), rows.end(),
            [](const auto& row) { return row.status == "ok"; }));
        const int ellipse_ok = static_cast<int>(std::count_if(
            rows.begin(), rows.end(),
            [](const auto& row) { return row.outer_status == "ok"; }));
        const int class_mismatch = static_cast<int>(std::count_if(
            rows.begin(), rows.end(),
            [](const auto& row) { return row.status == "class_mismatch"; }));
        std::cout << "\nFinished: " << rows.size() << " images\n"
                  << "Detection valid: " << detection_ok << '\n'
                  << "Outer ellipse IoU valid: " << ellipse_ok << '\n'
                  << "Class mismatch excluded: " << class_mismatch << '\n'
                  << "Detection statistics: " << detection_book << '\n'
                  << "Ellipse IoU statistics: " << iou_book << '\n'
                  << "Output: " << args.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 2;
    }
}
