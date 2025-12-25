// #include <opencv2/opencv.hpp>
// #include <atomic>
// #include <memory>
// #include <type_traits>
// #include <iostream>

// #include "npu_infer_pool.hpp"
// #include <image_process.hpp>

// // 自动适配 image_process 构造函数（如果不匹配，编译会提示你改这里）
// static std::unique_ptr<image_process> make_proc(cv::Mat& bgr)
// {
//     if constexpr (std::is_constructible_v<image_process, cv::Mat&>) {
//         return std::make_unique<image_process>(bgr);
//     } else if constexpr (std::is_constructible_v<image_process, const cv::Mat&>) {
//         return std::make_unique<image_process>((const cv::Mat&)bgr);
//     } else if constexpr (std::is_constructible_v<image_process, cv::Mat*>) {
//         return std::make_unique<image_process>(&bgr);
//     } else {
//         static_assert(
//             std::is_constructible_v<image_process, cv::Mat&> ||
//             std::is_constructible_v<image_process, const cv::Mat&> ||
//             std::is_constructible_v<image_process, cv::Mat*>,
//             "image_process has no supported constructor. Please edit make_proc() to match your image_process ctor."
//         );
//         return nullptr;
//     }
// }

// int main(int argc, char** argv)
// {
//     if (argc < 3) {
//         std::cerr << "Usage: " << argv[0] << " <model.rknn> <video_path>\n";
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::string video_path = argv[2];

//     cv::VideoCapture cap(video_path);
//     if (!cap.isOpened()) {
//         std::cerr << "Open video failed: " << video_path << "\n";
//         return 1;
//     }

//     npu_infer_pool pool(model_path, 6, 4);

//     // 可选：固定距离显示（如果你的 pool 已加入这两个接口）
//     // pool.set_pose_display_fixed(false);
//     // pool.set_pose_fixed_distance_mm(3000.0);

//     // 可选：cls2 mask 拟合开关 & 偏差阈值（如果你的 pool 有这两个接口）
//     // pool.set_class2_mask_fit_mode(true);
//     // pool.set_deviation_threshold(0.3f);

//     // 丢弃过期帧（视频一般不需要，但保留不影响）
//     auto expect_id = std::make_shared<std::atomic<uint64_t>>(0);
//     pool.set_expect_id_ptr(expect_id);

//     cv::namedWindow("Result", cv::WINDOW_NORMAL);

//     uint64_t fid = 0;

// double fps = 0.0;
// int64 lastTick = cv::getTickCount();
// const double tickFreq = cv::getTickFrequency();


//     while (true) {
//         cv::Mat frame;
//         if (!cap.read(frame) || frame.empty()) break;

//         expect_id->store(fid, std::memory_order_relaxed);
//         pool.AddInferenceTask(make_proc(frame));
//         fid++;

//         // 从输出队列取结果（你 BlockingQueue 的 pop() 若不是 optional，按你的实现改）
//         auto outOpt = pool.get_npu_infer_out().pop();
//         if (!outOpt) continue;

//         auto& out = *outOpt;

//         // 你在业务线程里已经画到 p->_src_image_frame 上了
//         if (out.proc && out.proc->_src_image_frame) {
//            cv::Mat show = *(out.proc->_src_image_frame);   // 注意：这里只是引用/浅拷贝
// // 如果你要缩放显示，可取消注释：
// // cv::resize(show, show, cv::Size(1280, 720));

// int64 nowTick = cv::getTickCount();
// double dt = (nowTick - lastTick) / tickFreq;
// lastTick = nowTick;

// double inst_fps = (dt > 1e-9) ? (1.0 / dt) : 0.0;
// // 平滑一下，避免跳动
// fps = (fps <= 0.0) ? inst_fps : (0.9 * fps + 0.1 * inst_fps);

// char buf[64];
// std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);

// // 画黑底+白字
// cv::putText(show, buf, cv::Point(20, 40),
//             cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,0), 4);
// cv::putText(show, buf, cv::Point(20, 40),
//             cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);

// cv::imshow("Result", show);

//             cv::resizeWindow("Result", 1280, 720);  // 你想要的窗口尺寸
//         } else {
//             // fallback（一般不会走到）
//             cv::imshow("Result", frame);
//         }

//         int k = cv::waitKey(1);
//         if (k == 27 || k == 'q') break;
//     }

//     pool.Stop();
//     cap.release();
//     cv::destroyAllWindows();
//     return 0;
// }



#include <opencv2/opencv.hpp>
#include <atomic>
#include <memory>
#include <type_traits>
#include <iostream>
#include <mutex>
#include <thread>

#include "npu_infer_pool.hpp"
#include <image_process.hpp>

// 自动适配 image_process 构造函数（如果不匹配，编译会提示你改这里）
static std::unique_ptr<image_process> make_proc(cv::Mat& bgr)
{
    if constexpr (std::is_constructible_v<image_process, cv::Mat&>) {
        return std::make_unique<image_process>(bgr);
    } else if constexpr (std::is_constructible_v<image_process, const cv::Mat&>) {
        return std::make_unique<image_process>((const cv::Mat&)bgr);
    } else if constexpr (std::is_constructible_v<image_process, cv::Mat*>) {
        return std::make_unique<image_process>(&bgr);
    } else {
        static_assert(
            std::is_constructible_v<image_process, cv::Mat&> ||
            std::is_constructible_v<image_process, const cv::Mat&> ||
            std::is_constructible_v<image_process, cv::Mat*>,
            "image_process has no supported constructor. Edit make_proc() to match your image_process ctor."
        );
        return nullptr;
    }
}

static inline void draw_fps(cv::Mat& img, double fps)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
    cv::putText(img, buf, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,0), 4);
    cv::putText(img, buf, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.rknn> <video_path>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Open video failed: " << video_path << "\n";
        return 1;
    }

    // 可选：减少 OpenCV 自身多线程抢资源（有时更快更稳）
    cv::setNumThreads(1);

    // 建 pool
    npu_infer_pool pool(model_path, 2, 5);

    // ====== 视频处理：不启用 expect_id 丢帧（避免“黑框没输出”）======
    // 所以不调用 set_expect_id_ptr()

    // 如果你 pool 支持这些开关，可以按需打开：
    // pool.set_class2_mask_fit_mode(true);
    // pool.set_deviation_threshold(0.3f);
    // pool.set_pose_display_fixed(false);
    // pool.set_pose_fixed_distance_mm(3000.0);

    // 显示窗口
    cv::namedWindow("Result", cv::WINDOW_NORMAL);
    cv::resizeWindow("Result", 1280, 720);

    // latest 结果缓存（consumer 写，主线程读）
    std::mutex mtx;
    cv::Mat latest_show;

    std::atomic<bool> running(true);

    // 消费线程：不停从 out_queue 取结果，保存最新图
    std::thread consumer([&]{
        while (running.load()) {
            auto outOpt = pool.get_npu_infer_out().pop(); // 阻塞OK
            if (!outOpt) continue;

            auto& out = *outOpt;
            if (out.proc && out.proc->_src_image_frame) {
                std::lock_guard<std::mutex> lk(mtx);
                latest_show = out.proc->_src_image_frame->clone(); // 深拷贝给显示用
            }
        }
    });

    // FPS 统计（显示端到端/显示吞吐）
    double fps = 0.0;
    int64 lastTick = cv::getTickCount();
    const double tickFreq = cv::getTickFrequency();

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        // 喂给 NPU（不等待结果）
        pool.AddInferenceTask(make_proc(frame));

        // 拿最新结果（没有就用原始 frame，避免黑屏）
        cv::Mat show;
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (!latest_show.empty()) show = latest_show;
        }
        if (show.empty()) show = frame;

        // FPS
        int64 nowTick = cv::getTickCount();
        double dt = (nowTick - lastTick) / tickFreq;
        lastTick = nowTick;
        double inst_fps = (dt > 1e-9) ? (1.0 / dt) : 0.0;
        fps = (fps <= 0.0) ? inst_fps : (0.9 * fps + 0.1 * inst_fps);
        draw_fps(show, fps);

        cv::imshow("Result", show);

        int k = cv::waitKey(1);
        if (k == 27 || k == 'q') break;
    }

    running.store(false);
    pool.Stop();
    if (consumer.joinable()) consumer.join();

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
