// 功能：
// 1) 给定 images_root / labels_root / prefixes (逗号分隔)
// 2) 在 images_root 下查找所有 “目录名匹配任意 prefix” 的子目录
// 3) 对每个子目录 D：读取 D 里的所有图片
// 4) 去 labels_root/D 里读取同名标签（默认 .txt）
// 5) 处理图片并保存数据
//
// 编译（Linux/macOS）：
//   g++ -std=c++17 main.cpp `pkg-config --cflags --libs opencv4` -o app
//
// 运行示例（支持多个前缀，用逗号隔开）：
//   ./app /path/to/images /path/to/labels "scene_01,scene_03"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <optional>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "yolov8seg.hpp"
#include "image_process.hpp"
#include "common.hpp"

using namespace std;
namespace fs = std::filesystem;

// ---------- 判断是否图片 ----------
static bool isImageFile(const fs::path &p)
{
    if (!p.has_extension())
        return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });

    static const std::vector<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

// ---------- 判断字符串前缀 ----------
static bool startsWith(const std::string &s, const std::string &prefix)
{
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// ---------- [新功能] 按逗号分割字符串 ----------
static std::vector<std::string> splitByComma(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// ---------- [新功能] 匹配任意前缀 ----------
static bool matchAnyPrefix(const std::string& name, const std::vector<std::string>& prefixes) {
    for (const auto& p : prefixes) {
        if (!p.empty() && startsWith(name, p)) return true;
    }
    return false;
}

// ---------- YOLO 单行标签：class_id + 4个数 ----------
struct Label4
{
    int id = -1;                       // class_id
    double a = 0, b = 0, c = 0, d = 0; // xc, yc, w, h（归一化）
};

// ---------- 读同名标签文件 ----------
static std::optional<Label4> readLabel4ForImage(const fs::path &imagePath,
                                                const fs::path &labelDir,
                                                const std::string &labelExt = ".txt")
{
    std::string stem = imagePath.stem().string();      // 001.jpg -> "001"
    fs::path labelPath = labelDir / (stem + labelExt); // labels/001.txt

    if (!fs::exists(labelPath) || !fs::is_regular_file(labelPath))
        return std::nullopt;

    std::ifstream ifs(labelPath);
    if (!ifs.is_open())
        return std::nullopt;

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string content = buffer.str();
    for (char &ch : content)
        if (ch == ',')
            ch = ' ';

    std::istringstream iss(content);
    Label4 lab;
    if (!(iss >> lab.id >> lab.a >> lab.b >> lab.c >> lab.d))
        return std::nullopt;

    return lab;
}

// 图像最终保存
bool saveImage(const cv::Mat &img, const std::string &outPath)
{
    fs::path p(outPath);
    if (p.has_parent_path())
    {
        fs::create_directories(p.parent_path());
    }
    if (!cv::imwrite(outPath, img))
    {
        std::cerr << "Failed to save image: " << outPath << std::endl;
        return false;
    }
    return true;
}

// ---------- 处理逻辑 ----------
static void processImage(const cv::Mat &img,
                         const std::string &imgPath,
                         const std::optional<Label4> &labelOpt,
                         yolov8seg &yolo,
                         std::vector<double>& src_x, std::vector<double>& src_y,
                         std::vector<double>& detect_x, std::vector<double>& detect_y,
                         std::vector<double>& Deviation_x, std::vector<double>& Deviation_y)
{
    if (!labelOpt.has_value())
    {
        std::cerr << "[WARN] No/Bad label for: " << imgPath << "\n";
        return;
    }

    const Label4 &lab = labelOpt.value();
    int W = img.cols, H = img.rows;

    double cx = lab.a * W;
    double cy = lab.b * H;
    src_x.push_back(cx);
    src_y.push_back(cy);
    double bw = lab.c * W;
    double bh = lab.d * H;

    int x1 = (int)std::round(cx - bw / 2.0);
    int y1 = (int)std::round(cy - bh / 2.0);
    int x2 = (int)std::round(cx + bw / 2.0);
    int y2 = (int)std::round(cy + bh / 2.0);

    x1 = std::max(0, std::min(x1, W - 1));
    y1 = std::max(0, std::min(y1, H - 1));
    x2 = std::max(0, std::min(x2, W - 1));
    y2 = std::max(0, std::min(y2, H - 1));

    cv::Mat vis = img.clone();
    cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    int xx = (x1 + x2) >> 1;
    int yy = (y1 + y2) >> 1;
    cv::circle(vis, cv::Point(xx, yy), 3, cv::Scalar(0, 255, 0), -1);

    cv::putText(vis, "Drogue",
                cv::Point(x1, std::max(0, y1 - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::Mat test_img = img.clone();
    image_process ip(test_img);

    ip.image_preprocessing(640, 640);
    int img_len;
    uint8_t *buffer = ip.get_image_buffer(&img_len);
    if (!buffer) { cout << "get_image_buffer error" << endl; return; }

    int err = yolo.set_input_data(buffer, img_len);
    if (err != 0) { std::cout << "yolo set_input_data error" << std::endl; return; }

    err = yolo.rknn_model_inference();
    if (err != 0) { std::cout << "yolo rknn_model_inference error" << std::endl; return; }

    err = yolo.get_output_data();
    if (err != 0) { std::cout << "yolo get_output_data error" << std::endl; return; }

    object_detect_result_list result;
    letterbox letter_box = ip.get_letterbox();

    err = yolo.post_process(result, letter_box);
    if (err < 0) { std::cout << "post_process error" << std::endl; return; }

    for (int i = 0; i < result.count; i++)
    {
        double temx1 = result.results_box[i].x;
        double temy1 = result.results_box[i].y;
        double temx2 = result.results_box[i].w + temx1;
        double temy2 = result.results_box[i].h + temy1;
        cv::Point pt1(temx1, temy1);
        cv::Point pt2(temx2, temy2);
        cv::Scalar color(0, 0, 255);
        cv::rectangle(vis, pt1, pt2, color, 2);

        if (i == 0)
        {
            double zz1 = (temx1 + temx2) / 2;
            double zz2 = (temy1 + temy2) / 2;
            cv::circle(vis, cv::Point(zz1, zz2), 3, cv::Scalar(0, 0, 255), -1);
            
            detect_x.push_back(zz1);
            detect_y.push_back(zz2);
            Deviation_x.push_back(zz1 - cx);
            Deviation_y.push_back(zz2 - cy);

            std::string text = "Ref: (" + std::to_string(cx) + ", " + std::to_string(cy) + ")";
            cv::putText(vis, text, cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 3);

            std::string text1 = "Det: (" + std::to_string(zz1) + ", " + std::to_string(zz2) + ")";
            cv::putText(vis, text1, cv::Point(20, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 3);

            std::string text2 = "Err: (" + std::to_string(zz1 - cx) + ", " + std::to_string(zz2 - cy) + ")";
            cv::putText(vis, text2, cv::Point(20, 300), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 3);
        }
    }

    // 1. 使用 filesystem 解析原始路径
    fs::path originalPath(imgPath);

    // 2. 提取不带扩展名的文件名 (例如 "001")
    std::string stem = originalPath.stem().string();

    // 3. 提取父目录名 (例如 "scene_01")，用于保持输出目录结构
    std::string parentDirName = originalPath.parent_path().filename().string();

    // 4. 定义一个统一的输出根目录，例如 "results"
    fs::path outputRoot("results_iamge");

    // 5. 构造完整的输出路径：results / scene_01 / 001_result.jpg
    // 这样既不会覆盖，又能保持原始的目录结构
    fs::path outputDir = outputRoot / parentDirName;
    fs::path outputPath = outputDir / (stem + "_result.jpg");

    // 6. 调用 saveImage 保存。saveImage 会自动创建不存在的目录。
    std::cout << "Saving result to: " << outputPath.string() << std::endl;
    saveImage(vis, outputPath.string());



    cv::imshow("vis", vis);
    cv::waitKey(1); // 1ms delay for display

    yolo.release_output_data();
}

int main(int argc, char **argv)
{
    // 修改用法说明
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <images_root> <labels_root> <prefixes_comma_separated>\n";
        std::cerr << "Example: " << argv[0] << " ./data/images ./data/labels \"scene_01,scene_03\"\n";
        return 1;
    }

    fs::path imagesRoot = argv[1];
    fs::path labelsRoot = argv[2];
    std::string prefixInput = argv[3]; // 获取逗号分隔的字符串

    if (!fs::exists(imagesRoot) || !fs::is_directory(imagesRoot))
    {
        std::cerr << "Error: images_root invalid: " << imagesRoot << "\n";
        return 1;
    }
    if (!fs::exists(labelsRoot) || !fs::is_directory(labelsRoot))
    {
        std::cerr << "Error: labels_root invalid: " << labelsRoot << "\n";
        return 1;
    }

    // 1) 解析前缀列表
    std::vector<std::string> prefixes = splitByComma(prefixInput);
    std::cout << "Target Prefixes: ";
    for(auto& p : prefixes) std::cout << "[" << p << "] ";
    std::cout << "\n";

    // 2) 查找 images_root 下匹配任意前缀的子目录
    std::vector<fs::path> matchedImageDirs;
    for (const auto &entry : fs::directory_iterator(imagesRoot))
    {
        if (!entry.is_directory())
            continue;
        std::string dirName = entry.path().filename().string();
        
        // 使用多前缀匹配函数
        if (matchAnyPrefix(dirName, prefixes))
        {
            matchedImageDirs.push_back(entry.path());
        }
    }

    if (matchedImageDirs.empty())
    {
        std::cerr << "No sub-directories matched prefixes under: " << imagesRoot << "\n";
        return 0;
    }

    std::cout << "Matched dirs count: " << matchedImageDirs.size() << "\n";

    // 初始化模型
    yolov8seg yolo("best.rknn");
    yolo.init();

    std::vector<double> Deviation_x;
    std::vector<double> Deviation_y;
    std::vector<double> src_x;
    std::vector<double> src_y;
    std::vector<double> detect_x;
    std::vector<double> detect_y;

    // 3) 逐目录处理
    for (const auto &imgDir : matchedImageDirs)
    {
        std::string dirName = imgDir.filename().string();
        fs::path labelDir = labelsRoot / dirName;

        if (!fs::exists(labelDir) || !fs::is_directory(labelDir))
        {
            std::cerr << "[WARN] Label dir missing: " << labelDir << "\n";
        }

        std::vector<fs::path> images;
        for (const auto &e : fs::directory_iterator(imgDir))
        {
            if (!e.is_regular_file()) continue;
            if (isImageFile(e.path()))
                images.push_back(e.path());
        }
        std::sort(images.begin(), images.end());

        std::cout << "Processing Dir [" << dirName << "]: " << images.size() << " images\n";

        for (const auto &imgPath : images)
        {
            cv::Mat img = cv::imread(imgPath.string(), cv::IMREAD_COLOR);
            if (img.empty()) continue;

            std::optional<Label4> labelOpt = readLabel4ForImage(imgPath, labelDir, ".txt");
            processImage(img, imgPath.string(), labelOpt, yolo, src_x, src_y, detect_x, detect_y, Deviation_x, Deviation_y);
        }
    }

    // ---------- 文件写入 ----------
    // 确保 val 目录存在
    if(!fs::exists("val")) fs::create_directory("val");

    int fd1 = open("val/误差x.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
    int fd2 = open("val/误差y.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
    int fd3 = open("val/参考值x.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
    int fd4 = open("val/参考值y.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
    int fd5 = open("val/检测值x.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
    int fd6 = open("val/检测值y.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);

    // [Fix] 写入 Deviation_x (误差) 到 fd1
    string arr;
    for (auto &x : Deviation_x) { arr += to_string(x) + " "; }
    write(fd1, arr.c_str(), arr.size());
    close(fd1);

    // [Fix] 写入 Deviation_y (误差) 到 fd2
    string brr;
    for (auto &x : Deviation_y) { brr += to_string(x) + " "; }
    write(fd2, brr.c_str(), brr.size());
    close(fd2);

    // 写入 src_x (参考值) 到 fd3
    string crr;
    for (auto &x : src_x) { crr += to_string(x) + " "; }
    write(fd3, crr.c_str(), crr.size());
    close(fd3);

    // 写入 src_y (参考值) 到 fd4
    string drr;
    for (auto &x : src_y) { drr += to_string(x) + " "; }
    write(fd4, drr.c_str(), drr.size());
    close(fd4);

    // 写入 detect_x (检测值) 到 fd5
    string frr;
    for (auto &x : detect_x) { frr += to_string(x) + " "; }
    write(fd5, frr.c_str(), frr.size());
    close(fd5);

    // 写入 detect_y (检测值) 到 fd6
    string err_str;
    for (auto &x : detect_y) { err_str += to_string(x) + " "; }
    write(fd6, err_str.c_str(), err_str.size());
    close(fd6);

    std::cout << "Done. Results saved to ./val/\n";
    cv::waitKey(0); // 如果不需要最后卡住可以注释掉
    return 0;
}