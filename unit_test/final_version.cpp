#include <opencv2/opencv.hpp>
#include <atomic>
#include <memory>
#include <iostream>
#include <mutex>
#include <thread>
#include <deque>
#include <condition_variable>
#include <chrono>
#include <string>

#include "npu_infer_pool.hpp"
#include <image_process.hpp>

static inline void draw_fps(cv::Mat& img, double fps)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
    cv::putText(img, buf, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,0), 4);
    cv::putText(img, buf, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);
}

struct Args {
    std::string model;
    bool use_cam = false;
    int cam_id = 0;
    std::string video_path;
    int target_fps = 0; // 0=不限速
};

static Args parse_args(int argc, char** argv)
{
    Args a;
    if (argc >= 2) a.model = argv[1];

    // 兼容旧用法：./app model.rknn video.mp4
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
        a.use_cam = false;
        a.video_path = argv[2];
    }

    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--cam" && i + 1 < argc) {
            a.use_cam = true;
            a.cam_id = std::stoi(argv[++i]);
        } else if (s == "--video" && i + 1 < argc) {
            a.use_cam = false;
            a.video_path = argv[++i];
        } else if (s == "--fps" && i + 1 < argc) {
            a.target_fps = std::stoi(argv[++i]);
        }
    }
    return a;
}



int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  Video: " << argv[0] << " <model.rknn> --video <path> [--fps 40]\n"
                  << "  Cam  : " << argv[0] << " <model.rknn> --cam <id>\n"
                  << "  (Compat) " << argv[0] << " <model.rknn> <video_path>\n";
        return 1;
    }

    Args args = parse_args(argc, argv);
    if (args.model.empty()) {
        std::cerr << "Model path empty.\n";
        return 1;
    }
    if (!args.use_cam && args.video_path.empty()) {
        std::cerr << "Video mode needs video path.\n";
        return 1;
    }

    // 可选：减少 OpenCV 自身多线程抢资源（有时更快更稳）
    cv::setNumThreads(1);

    // 建 npu_pool
    npu_infer_pool pool(args.model, 6, 6);

    // =======================
    // 多线程控制变量
    // =======================
    std::atomic<bool> running(true);

    // inflight 背压：限制“已提交但未出结果”的数量，避免积压导致延迟越来越大
    std::atomic<int> inflight(0);
    const int MAX_INFLIGHT = args.use_cam ? 4 : 12; // 摄像头更低延迟；视频可更大吞吐

    // 显示队列：consumer push，主线程 pop
    std::mutex show_mtx;
    std::condition_variable show_cv;  //条件变量
    std::deque<cv::Mat> show_q;

    // 摄像头模式：为了低延迟，只保留最新 1~2 帧（避免显示跟不上导致延迟堆积）
    // 视频模式：一般不丢，但也给个保护上限（理论上 inflight 限制后不会长）
    const size_t SHOW_Q_MAX = args.use_cam ? 2 : 60;

    // =======================
    // consumer 线程：从 out_queue 拿结果 -> 放到 show_q
    // =======================
    std::thread consumer([&]{
        while (running.load()) {
            auto outOpt = pool.get_npu_infer_out().pop(); // 阻塞等结果
            if (!outOpt) {
                // 通常是 Stop 后队列被关闭：直接退出更合理
                break;
            }

            auto& out = *outOpt;

            // 有结果就认为一个 inflight 结束
            inflight.fetch_sub(1, std::memory_order_relaxed);   //原子地把 inflight 减 1

            if (out.proc && out.proc->_src_image_frame) {
                cv::Mat img = out.proc->_src_image_frame->clone(); // 深拷贝，显示安全

                {
                    std::lock_guard<std::mutex> lk(show_mtx);//实现了“进入代码块自动上锁，离开自动解锁”    就是lk离开作用域会立即解锁
//                                                            //用 RAII 自动加锁/解锁，保护临界区，防止多线程同时读写 latest_show 造成数据竞争和崩溃

                    // 摄像头实时：队列满了就丢旧的，保证低延迟（只看最新）
                    while (show_q.size() >= SHOW_Q_MAX) show_q.pop_front();

                    show_q.push_back(std::move(img));
                }
                show_cv.notify_one();
            }
        }
    });

    // =======================
    // feed 线程：读视频/摄像头 -> 投喂 pool（不等待结果）
    // =======================
    std::thread feeder([&]{
        cv::VideoCapture cap;
        if (args.use_cam) {
            cap.open(args.cam_id);
        // === 关键修改 ===
        // 降低分辨率，减少 CPU 负担
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640); 
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
        // 使用压缩格式，降低 USB 带宽压力
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        // ================

        } else {
            cap.open(args.video_path);
        }

        if (!cap.isOpened()) {
            std::cerr << "Open source failed: "
                      << (args.use_cam ? ("cam " + std::to_string(args.cam_id)) : args.video_path)
                      << "\n";
            running.store(false);
            show_cv.notify_all();
            return;
        }

        // 视频可选限速到 target_fps（例如40）   
        //如果是视频文件并且设置了 target_fps，就计算每帧应该等待的时间间隔，用来把处理速度限制在目标帧率
        using clock = std::chrono::steady_clock;   //单调提增的时钟，用作“帧间隔控制”
        auto next_tp = clock::now();       //记录下一帧开始的时间点
        std::chrono::microseconds period(0);  //定义一个帧周期，默认0不限速
        if (!args.use_cam && args.target_fps > 0) {
            period = std::chrono::microseconds(1000000 / args.target_fps);
        }

     
                while (running.load()) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) break;

            // ==========================
            // 背压策略（最优方案）
            // 1）摄像头：inflight 满了就丢输入，不等待 -> 低延迟实时
            // 2）视频：背压等待 -> 防止积压导致延迟越来越大
            // ==========================
            if (args.use_cam) {
                // 摄像头画面一直变化：不等待，否则你会“卡在过去的帧”
                if (inflight.load(std::memory_order_relaxed) >= MAX_INFLIGHT) {
                    continue; // 直接丢这帧输入（你的工程内部也有丢帧策略，这里是输入端再保证实时）
                }
            } else {
                // 视频文件：可以等，保证不积压（不会越跑越落后）  也就是不让读的那么快
                while (running.load() &&
                       inflight.load(std::memory_order_relaxed) >= MAX_INFLIGHT) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));   //避免空转消耗cpu，每次睡眠1ms，让出时间片给推理线程/consumer线程去把 inflight 降下来。
                }
                if (!running.load()) break;
            }

            // 投喂（不等待）
            auto tem_image = std::make_unique<image_process>(frame);
            pool.AddInferenceTask(std::move(tem_image));
            inflight.fetch_add(1, std::memory_order_relaxed);

            // 视频模式：限速到目标FPS（摄像头不需要）
            if (!args.use_cam && period.count() > 0) {
                next_tp += period;
                std::this_thread::sleep_until(next_tp); //睡眠到下一个读取帧的时间
            }
        }


        cap.release();
        running.store(false);
        show_cv.notify_all();
    });

    // =======================
    // 主线程：只负责显示（尽量显示每个输出结果）
    // =======================
    cv::namedWindow("Result", cv::WINDOW_NORMAL);
    cv::resizeWindow("Result", 1280, 720);


    double fps = 0.0;
    int64 lastTick = cv::getTickCount();  //返回一个 “高精度计时器”的当前计数值 ，  后面每帧都会再取一次 nowTick，两者相减得到经过了多少 tick。
    const double tickFreq = cv::getTickFrequency();   //返回计时器的频率，没秒有多少tick

    while (running.load() || !show_q.empty()) {
        cv::Mat show;

        {
            std::unique_lock<std::mutex> lk(show_mtx);
            if (show_q.empty()) {
                // 等一小会儿，避免空转
                show_cv.wait_for(lk, std::chrono::milliseconds(10));  //这里会释放锁   被唤醒或超时后 自动重新拿锁
            }
            if (!show_q.empty()) {
                show = std::move(show_q.front());
                show_q.pop_front();
            }
        }

        if (!show.empty()) {
            // FPS（按显示帧统计）
            int64 nowTick = cv::getTickCount();
            double dt = (nowTick - lastTick) / tickFreq;
            lastTick = nowTick;
            double inst_fps = (dt > 1e-9) ? (1.0 / dt) : 0.0;
            fps = (fps <= 0.0) ? inst_fps : (0.9 * fps + 0.1 * inst_fps);
            draw_fps(show, fps);

            cv::imshow("Result", show);
        }

        int k = cv::waitKey(1);
        if (k == 27 || k == 'q') {
            running.store(false);
            show_cv.notify_all();
            break;
        }
    }

    // =======================
    // 退出清理：先停 feeder，再 Stop pool，再停 consumer
    // =======================
    running.store(false);
    show_cv.notify_all();

    if (feeder.joinable()) feeder.join();

    pool.Stop(); // 让 out_queue pop 解除阻塞

    if (consumer.joinable()) consumer.join();

    cv::destroyAllWindows();
    return 0;
}
