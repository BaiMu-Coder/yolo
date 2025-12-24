// #include <iostream>
// #include <memory>
// #include <opencv2/opencv.hpp>
// #include <unistd.h>
// #include <chrono>
// #include <string>
// #include <thread> // 必须引入线程库

// #include "npu_infer_pool.hpp"
// #include "business.hpp"
// #include "videofile.hpp"

// using namespace std;

// // =========================================================
// // 全局显示队列：连接 OrderedProcessor 和 主线程显示循环
// // =========================================================
// BlockingQueue<InferOut> g_display_queue;

// // =========================================================
// // 【修改后】真正的显示逻辑 (将在主线程运行)
// // 现在它只负责 FPS 计算、旋转和最终显示，所有脏活累活都在后台完成了
// // =========================================================
// void display_logic(InferOut&& out)
// {
//     // --- FPS 统计 ---
//     static auto last_time = std::chrono::steady_clock::now();
//     static int frame_count = 0;
//     static double current_fps = 0.0;

//     frame_count++;
//     auto now = std::chrono::steady_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();

//     if (duration >= 1000) {
//         current_fps = frame_count * 1000.0 / duration;
//         frame_count = 0;
//         last_time = now;
//         std::cout << "FPS: " << current_fps << std::endl; 
//     }

//     // 获取图像帧（此时图像上已经画好了框和掩码）
//     cv::Mat& frame = *(out.proc->_src_image_frame);

//     // 画 FPS (这个比较轻量，留在主线程画也没问题)
//     // 如果想极致优化，也可以移到后台画
//     std::string fps_text = "FPS: " + std::to_string((int)current_fps);
//     // 为了演示，我这里去掉了你原来的 +20
//     cv::rectangle(frame, cv::Point(0, 0), cv::Point(140, 40), cv::Scalar(0, 0, 0), -1);
//     cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

//     // 逆时针旋转 90 度 (根据需要保留或删除)
//     cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

//     // 【最终上屏】
//     cv::imshow("1", frame);
//     // 这里的 waitKey 在主线程运行，非常安全，用于刷新UI
//     cv::waitKey(1); 
// }

// // =========================================================
// // 回调函数：OrderedProcessor 调用的，它只负责把数据推送到队列
// // =========================================================
// void push_to_display_queue(uint64_t frame_id, InferOut&& out) {
//     // 简单地把结果推送到全局队列，不涉及任何 UI 操作
//     g_display_queue.push(std::move(out));
// }

// // =========================================================
// // 主函数
// // =========================================================
// int main(int argc, char* argv[])
// {
//     if (argc < 2) { return -1; }

//     // 1. 初始化 NPU 池
//     // 6个推理线程，5个业务线程
//     npu_infer_pool infer_pool("best.rknn", 6, 5);
    
//     // 【可选】如果你需要设置额外的业务回调，可以在这里设置
//     // infer_pool.set_business_callback([](uint64_t fid, std::shared_ptr<image_process> p, object_detect_result_list& r){
//     //     cout << "Business callback for frame " << fid << endl;
//     // });

//     // 2. 初始化排序模块
//     BlockingQueue<InferOut>& link_queue = infer_pool.get_npu_infer_out();
//     OrderedProcessor<InferOut> bus(link_queue, 20, 4, push_to_display_queue);
//     bus.Start();

//     string video_path = argv[1];

//     // 3. 启动视频读取线程
//     std::thread reader_thread([&]() {
//         cv::VideoCapture cap(video_path);
//         if (!cap.isOpened()) {
//             cerr << "Open video failed!" << endl;
//             return;
//         }

//         cv::Mat frame;
//         while (cap.read(frame)) {
//             // 深拷贝图像，因为后续在后台线程会修改它（画框）
//             cv::Mat frame_copy = frame.clone();
//             auto proc = std::make_unique<image_process>(frame_copy);
//             infer_pool.AddInferenceTask(std::move(proc));
            
//             // 可选：限速防止爆内存
//             // std::this_thread::sleep_for(std::chrono::milliseconds(10));
//         }
//         cout << "Video read complete. Flushing pipeline..." << endl;

//         // 优雅退出流程
//         infer_pool.Stop(); 
//         bus.Join();
//         g_display_queue.stop(); 
//     });

//     // 4. 【主线程】只做最后的显示
//     while (auto out = g_display_queue.pop()) {
//         display_logic(std::move(*out));
//     }

//     if (reader_thread.joinable()) {
//         reader_thread.join();
//     }

//     cout << "All done. Exiting safely." << endl;
//     return 0;
// }










#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <chrono>
#include <string>
#include <thread>
#include <atomic> // 必须引入
#include "block_queue.hpp" 
#include "npu_infer_pool.hpp"

#include "business.hpp"
#include "videofile.hpp"

using namespace std;

// =========================================================
// 全局显示队列：连接 OrderedProcessor 和 主线程显示循环
// =========================================================
BlockingQueue<InferOut> g_display_queue;

// =========================================================
// 显示逻辑 (主线程运行)
// 此时拿到的 out.proc->_src_image_frame 已经是画好框和拟合圆的成品了
// =========================================================
void display_logic(InferOut&& out)
{
    // --- FPS 统计 ---
    static auto last_time = std::chrono::steady_clock::now();
    static int frame_count = 0;
    static double current_fps = 0.0;

    frame_count++;
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();

    if (duration >= 1000) {
        current_fps = frame_count * 1000.0 / duration;
        frame_count = 0;
        last_time = now;
        std::cout << "FPS: " << current_fps << std::endl; 
    }

    // 获取成品图像
    cv::Mat& frame = *(out.proc->_src_image_frame);

    // 绘制 FPS (轻量级)
    std::string fps_text = "FPS: " + std::to_string((int)current_fps);
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(140, 40), cv::Scalar(0, 0, 0), -1);
    cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    // 旋转 (根据实际安装角度调整)
    cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

    // 上屏
    cv::imshow("1", frame);
    cv::waitKey(1); 
}

// =========================================================
// 回调桥梁
// =========================================================
void push_to_display_queue(uint64_t frame_id, InferOut&& out) {
    g_display_queue.push(std::move(out));
}

// =========================================================
// 主函数
// =========================================================
int main(int argc, char* argv[])
{
    if (argc < 2) { 
        cerr << "Usage: " << argv[0] << " <video_path>" << endl;
        return -1; 
    }

    // -------------------------------------------------------------
    // 1. 创建共享的“反压信号灯” (Atomic Variable)
    // -------------------------------------------------------------
    // 这个变量存储“当前界面显示的帧号”，告诉 NPU 哪些帧已经过期了可以扔掉
    auto shared_expect_id = std::make_shared<std::atomic<uint64_t>>(0);

    // -------------------------------------------------------------
    // 2. 初始化 NPU 推理池
    // -------------------------------------------------------------
    // 6个推理线程，5个业务线程
    npu_infer_pool infer_pool("best.rknn", 6, 5);
    
    // 【关键配置 A】连接反压信号
    infer_pool.set_expect_id_ptr(shared_expect_id);

    // 【关键配置 B】设置类别 2 (如加油锥套) 的拟合模式
    // true  = 优先尝试掩码拟合 (Mask Fit)
    // false = 强制使用检测框内切圆 (Box Inner Circle)
    // 根据你之前的需求，这里可以设为 false 强制用内切圆，或者设为 true 自动降级
    infer_pool.set_class2_mask_fit_mode(false); 

    // 【关键配置 C】设置鲁棒性阈值
    // 如果 Mask 拟合出的圆心，偏离检测框中心超过 检测框短边的 30%，则强制回退到内切圆
    infer_pool.set_deviation_threshold(0.3f);

    // -------------------------------------------------------------
    // 3. 初始化排序模块 (OrderedProcessor)
    // -------------------------------------------------------------
    BlockingQueue<InferOut>& link_queue = infer_pool.get_npu_infer_out();
    
    // 初始化排序器：缓存20帧，超时容忍4帧
    OrderedProcessor<InferOut> bus(link_queue, 20, 4, push_to_display_queue);
    
    // 【关键配置 D】让排序器能够更新“反压信号灯”
    // 只有把这个传进去，排序器每显示一帧，shared_expect_id 才会增加，NPU 才知道该扔哪些帧
    bus.set_expect_id_notifier(shared_expect_id); 
    
    bus.Start();

    // -------------------------------------------------------------
    // 4. 启动读取线程 (生产者)
    // -------------------------------------------------------------
    string video_path = argv[1];
    std::thread reader_thread([&]() {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            cerr << "Open video failed!" << endl;
            return;
        }

        cout << "Video started. Reading frames..." << endl;
        cv::Mat frame;
        while (cap.read(frame)) {
            // 深拷贝，因为是异步流水线
            cv::Mat frame_copy = frame.clone();
            auto proc = std::make_unique<image_process>(frame_copy);
            infer_pool.AddInferenceTask(std::move(proc));
            
            // 可选：如果读文件太快导致内存暴涨，可以稍微 sleep 一下
            // std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        // cout << "Video read complete. Flushing pipeline..." << endl;

        // --- 优雅退出序列 ---
        // 1. 停止 NPU 接收新任务
        infer_pool.Stop(); 
        // 2. 等待排序器处理完剩余缓存
        bus.Join();
        // 3. 通知主线程退出循环
        g_display_queue.stop(); 
    });

    // -------------------------------------------------------------
    // 5. 主线程 UI 循环 (消费者)
    // -------------------------------------------------------------
    // 只要队列里有数据，或者还没收到 stop 信号，就一直显示
    while (auto out = g_display_queue.pop()) {
        display_logic(std::move(*out));
    }

    // 等待读取线程收尾
    if (reader_thread.joinable()) {
        reader_thread.join();
    }

    cout << "All done. Exiting safely." << endl;
    return 0;
}
