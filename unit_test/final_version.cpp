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



// #include <opencv2/opencv.hpp>
// #include <atomic>
// #include <memory>
// #include <type_traits>
// #include <iostream>
// #include <mutex>
// #include <thread>

// #include "npu_infer_pool.hpp"
// #include <image_process.hpp>



// static inline void draw_fps(cv::Mat& img, double fps)
// {
//     char buf[64];
//     std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
//     cv::putText(img, buf, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,0), 4);
//     cv::putText(img, buf, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);
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

//     // 可选：减少 OpenCV 自身多线程抢资源（有时更快更稳）
//     cv::setNumThreads(1);

//     // 建 pool
//     npu_infer_pool pool(model_path, 6, 3);

//     // ====== 视频处理：不启用 expect_id 丢帧（避免“黑框没输出”）======
//     // 所以不调用 set_expect_id_ptr()

//     // 如果你 pool 支持这些开关，可以按需打开：
//     // pool.set_class2_mask_fit_mode(true);
//     // pool.set_deviation_threshold(0.3f);
//     // pool.set_pose_display_fixed(false);
//     // pool.set_pose_fixed_distance_mm(3000.0);

//     // 显示窗口
//     cv::namedWindow("Result", cv::WINDOW_NORMAL);
//     cv::resizeWindow("Result", 1280, 720);

//     // latest 结果缓存（consumer 写，主线程读）
//     std::mutex mtx;
//     cv::Mat latest_show;

//     std::atomic<bool> running(true);     //控制consumer线程是否继续运行的开关

//     // 消费线程：不停从 out_queue 取结果，保存最新图
//     std::thread consumer([&]{
//         while (running.load()) {
//             auto outOpt = pool.get_npu_infer_out().pop(); // 阻塞OK
//             if (!outOpt) continue;

//             auto& out = *outOpt;
//             if (out.proc && out.proc->_src_image_frame) {
//                  auto tem_frame = out.proc->_src_image_frame->clone(); // 深拷贝给显示用

//                  {
//                 std::lock_guard<std::mutex> lk(mtx);    //实现了“进入代码块自动上锁，离开自动解锁”    就是lk离开作用域会立即解锁
//                                                         //用 RAII 自动加锁/解锁，保护临界区，防止多线程同时读写 latest_show 造成数据竞争和崩溃
//                 latest_show = std::move(tem_frame);
//                                                     }
            
//             }
//         }
//     });

//     // FPS 统计（显示端到端/显示吞吐）
//     double fps = 0.0;
//     int64 lastTick = cv::getTickCount();  //返回一个 “高精度计时器”的当前计数值 ，  后面每帧都会再取一次 nowTick，两者相减得到经过了多少 tick。
//     const double tickFreq = cv::getTickFrequency();   //返回计时器的频率，没秒有多少tick

//     while (true) {
//         cv::Mat frame;
//         if (!cap.read(frame) || frame.empty()) break;  //读帧+检查有效性

//         // 喂给 NPU（不等待结果）
//         auto tem_image=std::make_unique<image_process>(frame);
//         tem_image->image_preprocessing(640,640);
//         pool.AddInferenceTask(std::move(tem_image));

//         // 拿最新结果（没有就用原始 frame，避免黑屏）
//         cv::Mat show;
//         {
//             std::lock_guard<std::mutex> lk(mtx);
//             if (!latest_show.empty()) show = latest_show;
//         }
//         if (show.empty()) show = frame;

//         // FPS
//         int64 nowTick = cv::getTickCount();
//         double dt = (nowTick - lastTick) / tickFreq;
//         lastTick = nowTick;
//         double inst_fps = (dt > 1e-9) ? (1.0 / dt) : 0.0;
//         fps = (fps <= 0.0) ? inst_fps : (0.9 * fps + 0.1 * inst_fps);
//         draw_fps(show, fps);

//         cv::imshow("Result", show);

//         int k = cv::waitKey(1);
//         if (k == 27 || k == 'q') break;
//     }

//     running.store(false);
//     pool.Stop();  //
//     if (consumer.joinable())  //防御性判断
//      consumer.join();

//     cap.release();
//     cv::destroyAllWindows();
//     return 0;
// }




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
    npu_infer_pool pool(args.model, 6, 3);

    // =======================
    // 多线程控制变量
    // =======================
    std::atomic<bool> running(true);

    // inflight 背压：限制“已提交但未出结果”的数量，避免积压导致延迟越来越大
    std::atomic<int> inflight(0);
    const int MAX_INFLIGHT = args.use_cam ? 2 : 10; // 摄像头更低延迟；视频可更大吞吐

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
