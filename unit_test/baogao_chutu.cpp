// 功能：
// 1) 给一个文件夹，递归读取所有图片
// 2) YOLOv8Seg 推理
// 3) 对每个检测：画框 + 椭圆拟合（优先 mask），失败则画“内切圆”
// 4) 如果整张图没有任何检测：Canny -> 找最大轮廓 -> minEnclosingCircle 画圆
//
// 编译：
// g++ -std=c++17 main.cpp `pkg-config --cflags --libs opencv4` -O2 -o app
//
// 运行：
// ./app <image_folder> <model.rknn> [out_dir] [--show]

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <optional>

#include "yolov8seg.hpp"
#include "image_process.hpp"
#include "common.hpp"

namespace fs = std::filesystem;

// -------------------- 是否图片 --------------------
static bool isImageFile(const fs::path &p)
{
    if (!p.has_extension()) return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    static const std::vector<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

// -------------------- 安全保存 --------------------
static bool saveImage(const cv::Mat &img, const fs::path &outPath)
{
    if (outPath.has_parent_path()) fs::create_directories(outPath.parent_path());
    if (!cv::imwrite(outPath.string(), img)) {
        std::cerr << "Failed to save: " << outPath << "\n";
        return false;
    }
    return true;
}


static inline void overlay_mask_roi(
    cv::Mat& bgr,
    const cv::Rect& roi,
    const cv::Mat& full_mask_u8,   // 与原图同尺寸的 0/255 mask
    const cv::Scalar& color_bgr,   // 叠加颜色
    float alpha = 0.45f            // 透明度
){
    if (full_mask_u8.empty() || full_mask_u8.type() != CV_8U) return;
    cv::Rect r = roi & cv::Rect(0,0,bgr.cols,bgr.rows);
    if (r.width <= 0 || r.height <= 0) return;

    for (int y = r.y; y < r.y + r.height; ++y) {
        cv::Vec3b* p = bgr.ptr<cv::Vec3b>(y);
        const uchar* m = full_mask_u8.ptr<uchar>(y);
        for (int x = r.x; x < r.x + r.width; ++x) {
            if (m[x] == 0) continue;
            p[x][0] = cv::saturate_cast<uchar>(p[x][0] * (1 - alpha) + color_bgr[0] * alpha);
            p[x][1] = cv::saturate_cast<uchar>(p[x][1] * (1 - alpha) + color_bgr[1] * alpha);
            p[x][2] = cv::saturate_cast<uchar>(p[x][2] * (1 - alpha) + color_bgr[2] * alpha);
        }
    }
}


// ============================================================
// 兜底 1：从检测框得到“内切圆”
// ============================================================
static void draw_inscribed_circle(cv::Mat& vis, const object_detect_result& det, const cv::Scalar& color)
{
    float cx = det.x + det.w * 0.5f;
    float cy = det.y + det.h * 0.5f;
    float r  = 0.5f * std::min(det.w, det.h);
    cv::circle(vis, cv::Point2f(cx, cy), (int)std::round(r), color, 2);
    cv::circle(vis, cv::Point2f(cx, cy), 2, color, -1);
}

// ============================================================
// 兜底 2：整张图没检测出来 -> Canny 提取圆（用轮廓最大内接圆）
// ============================================================
static bool fallback_circle_by_canny(const cv::Mat& bgr, cv::Point2f& bestCenter, float& bestR)
{
    cv::Mat gray, blurImg, edges;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurImg, cv::Size(5,5), 1.2);
    cv::Canny(blurImg, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;

    bestR = 0.0f;
    for (const auto& c : contours) {
        if (c.size() < 20) continue; // 太小的边缘噪声忽略
        cv::Point2f cc;
        float rr = 0.0f;
        cv::minEnclosingCircle(c, cc, rr);
        if (rr > bestR) {
            bestR = rr;
            bestCenter = cc;
        }
    }
    return bestR > 5.0f; // 半径太小就不算
}

// ============================================================
// 椭圆拟合：优先 mask 拟合，失败则返回 “内切圆”
// 注：这里假设 raw_mask_ptr 是与原图同尺寸的 8UC1 mask（你项目里就是这么用的）
// ============================================================
struct EllipseFitResult {
    bool ok = false;
    bool from_mask = false;
    cv::RotatedRect ellipse;
};

static EllipseFitResult fit_ellipse_from_mask_or_fallback(
    const cv::Mat& frame,
    const object_detect_result& det,
    uint8_t* raw_mask_ptr,
    float deviation_threshold = 0.3f
){
    EllipseFitResult out;

    // box roi
    int x = std::max(0, det.x);
    int y = std::max(0, det.y);
    int w = std::min((int)det.w, frame.cols - x);
    int h = std::min((int)det.h, frame.rows - y);
    if (w <= 0 || h <= 0) return out;

    cv::Point2f box_center(x + w * 0.5f, y + h * 0.5f);

    bool fit_success = false;
    cv::RotatedRect fitted;

    // --- 尝试 mask 拟合 ---
    if (raw_mask_ptr) {
        // ⚠️ 假设 mask 与 frame 同尺寸（你原代码/池子里也是这么构造的）
        cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, raw_mask_ptr);
        cv::Rect roi_rect(x, y, w, h);
        cv::Mat roi_mask = full_mask(roi_rect);

        // findContours 会改数据，所以 clone 一份
        cv::Mat contour_input = roi_mask.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(contour_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            // 选点最多的轮廓
            auto max_itr = std::max_element(
                contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b){
                    return a.size() < b.size();
                }
            );

            if (max_itr->size() >= 5) {
                cv::RotatedRect local = cv::fitEllipse(*max_itr);
                local.center.x += x;
                local.center.y += y;

                // 偏差校验：mask 椭圆中心不能离 box 中心太远
                float dist = cv::norm(local.center - box_center);
                float limit = std::min((float)w, (float)h) * deviation_threshold;
                if (dist <= limit) {
                    fitted = local;
                    fit_success = true;
                }
            }
        }
    }

    if (fit_success) {
        out.ok = true;
        out.from_mask = true;
        out.ellipse = fitted;
        return out;
    }

    // --- 失败：返回“box 内切圆”（用 RotatedRect 表示一个正圆） ---
    float r = 0.5f * std::min((float)w, (float)h);
    out.ok = true;
    out.from_mask = false;
    out.ellipse = cv::RotatedRect(box_center, cv::Size2f(2*r, 2*r), 0.0f);
    return out;
}

// ============================================================
// 单张图片处理：推理 + 绘制
// ============================================================
static bool processOneImage(
    const cv::Mat& img_bgr,
    const fs::path& imgPath,
    yolov8seg& yolo,
    const fs::path& outDir,
    bool show
){
    if (img_bgr.empty()) return false;

    cv::Mat vis = img_bgr.clone();

    // ---------- 推理（保持你原来的调用方式） ----------
    cv::Mat tmp = img_bgr.clone();
    image_process ip(tmp);
    ip.image_preprocessing(640, 640);

    int img_len = 0;
    uint8_t* buffer = ip.get_image_buffer(&img_len);
    if (!buffer) {
        std::cerr << "get_image_buffer error: " << imgPath << "\n";
        return false;
    }

    int err = yolo.set_input_data(buffer, img_len);
    if (err != 0) { std::cerr << "set_input_data error\n"; return false; }

    err = yolo.rknn_model_inference();
    if (err != 0) { std::cerr << "rknn_model_inference error\n"; return false; }

    err = yolo.get_output_data();
    if (err != 0) { std::cerr << "get_output_data error\n"; return false; }

    object_detect_result_list result;
    letterbox lb = ip.get_letterbox();

    err = yolo.post_process(result, lb);
    if (err < 0) {
        std::cerr << "post_process error\n";
        yolo.release_output_data();
        return false;
    }

    // ---------- 画检测框 + 椭圆/内切圆 ----------
    if (result.count > 0) {
        const auto& seg_result = result.results_mask[0];

        for (int i = 0; i < result.count; ++i) {
            const auto& det = result.results_box[i];

            // box
            int x = std::max(0, det.x);
            int y = std::max(0, det.y);
            int w = std::min((int)det.w, vis.cols - x);
            int h = std::min((int)det.h, vis.rows - y);
            if (w <= 0 || h <= 0) continue;

            cv::Rect box(x, y, w, h);
            cv::rectangle(vis, box, cv::Scalar(0, 0, 255), 2);

            // label
            char text[128];
            std::snprintf(text, sizeof(text), "cls=%d conf=%.2f", det.cls_id, det.prop);
            cv::putText(vis, text, cv::Point(x, std::max(0, y-5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);

            // mask ptr（按你的 each_of_mask 习惯：用 cls_id 索引）
            uint8_t* raw_mask_ptr = nullptr;
            int cid = det.cls_id;
            if (cid >= 0 && cid < (int)seg_result.each_of_mask.size()) {
                if (seg_result.each_of_mask[cid]) {
                    raw_mask_ptr = seg_result.each_of_mask[cid].get();
                    // 画 mask 叠加（如果有 mask）
if (raw_mask_ptr) {
    cv::Mat full_mask(vis.rows, vis.cols, CV_8UC1, raw_mask_ptr);

    // 给不同类一个颜色（你可以自己改）
    cv::Scalar color;
    if (det.cls_id % 3 == 0) color = cv::Scalar(0, 0, 255);      // 红
    else if (det.cls_id % 3 == 1) color = cv::Scalar(0, 255, 0); // 绿
    else color = cv::Scalar(255, 255, 0);                        // 青

    overlay_mask_roi(vis, box, full_mask, color, 0.45f);
}

                }
            }

            // fit ellipse (mask优先)，失败则内切圆
            EllipseFitResult e = fit_ellipse_from_mask_or_fallback(vis, det, raw_mask_ptr, 0.2f);

            if (e.ok) {
                cv::Scalar ec = e.from_mask ? cv::Scalar(0,255,0) : cv::Scalar(0,255,255); // mask绿 / 兜底黄
                // 画椭圆（如果是兜底，这里画的是圆）
                cv::ellipse(vis, e.ellipse, ec, 2);
                cv::circle(vis, e.ellipse.center, 2, ec, -1);
            } else {
                // 极端兜底（一般不会到这里）：直接画内切圆
                draw_inscribed_circle(vis, det, cv::Scalar(0,255,255));
            }
        }
    } else {
        // ---------- 没任何检测：Canny 提取圆 ----------
        cv::Point2f c;
        float r = 0.f;
        bool ok = fallback_circle_by_canny(vis, c, r);
        if (ok) {
            cv::circle(vis, c, (int)std::round(r), cv::Scalar(255, 0, 255), 3);
            cv::circle(vis, c, 3, cv::Scalar(255, 0, 255), -1);
            cv::putText(vis, "Canny fallback circle", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255,0,255), 2);
        } else {
            cv::putText(vis, "No det & no circle found", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0,0,255), 2);
        }
    }

    yolo.release_output_data();

    // ---------- 保存 ----------
    // 保持相对路径结构：out_dir / relative_path / xxx_vis.jpg
    // 这里只用文件名，不包含子目录的话也行
    fs::path outPath = outDir / (imgPath.stem().string() + "_vis.jpg");
    saveImage(vis, outPath);

    // ---------- 可选显示 ----------
    if (show) {
        cv::imshow("vis", vis);
        int k = cv::waitKey(1);
        if (k == 27 || k == 'q') return false; // 提前退出
    }

    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_folder> <model.rknn> [out_dir] [--show]\n";
        std::cerr << "Example: " << argv[0] << " ./images best.rknn ./out --show\n";
        return 1;
    }

    fs::path inDir = argv[1];
    std::string modelPath = argv[2];
    fs::path outDir = (argc >= 4 && std::string(argv[3]).rfind("--", 0) != 0) ? fs::path(argv[3]) : fs::path("out_vis");

    bool show = false;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--show") show = true;
    }

    if (!fs::exists(inDir) || !fs::is_directory(inDir)) {
        std::cerr << "Input folder invalid: " << inDir << "\n";
        return 1;
    }

    // 初始化 yolo
    yolov8seg yolo(modelPath.c_str());
    if (yolo.init() != 0) {
        std::cerr << "yolo init failed: " << modelPath << "\n";
        return 1;
    }

    if (show) {
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
        cv::resizeWindow("vis", 1280, 720);
    }

    // 收集所有图片（递归）
    std::vector<fs::path> images;
    for (auto& p : fs::recursive_directory_iterator(inDir)) {
        if (!p.is_regular_file()) continue;
        if (isImageFile(p.path())) images.push_back(p.path());
    }
    std::sort(images.begin(), images.end());

    std::cout << "Found images: " << images.size() << "\n";
    fs::create_directories(outDir);

    for (const auto& imgPath : images) {
        cv::Mat img = cv::imread(imgPath.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "imread failed: " << imgPath << "\n";
            continue;
        }

        bool ok = processOneImage(img, imgPath, yolo, outDir, show);
        if (!ok) break;
    }

    if (show) {
        std::cout << "Press any key to exit...\n";
        cv::waitKey(0);
    }

    std::cout << "Done. Output: " << outDir << "\n";
    return 0;
}
