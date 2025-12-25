// #pragma once
// #include "yolov8seg.hpp"
// #include "thread_pool.hpp"
// #include "block_queue.hpp"
// #include <image_process.hpp>
// #include <string>
// #include <memory>
// #include <vector>
// #include <atomic>
// #include <functional>
// #include <opencv2/opencv.hpp> // 必须包含

// // 输出结构体 (使用 shared_ptr)
// struct InferOut
// {
//     uint64_t frame_id;
//     std::shared_ptr<image_process> proc;
//     object_detect_result_list result;
// };

// // 输入任务结构体
// struct InferJob
// {
//     uint64_t frame_id;
//     std::unique_ptr<image_process> proc;
// };

// // 椭圆拟合结果结构体
// struct EllipseResult {
//     cv::RotatedRect rect;
//     bool is_from_mask; // true=掩码拟合, false=Box保底
// };

// class npu_infer_pool
// {
// public:
//     // 业务回调类型
//     using BusinessCallback = std::function<void(uint64_t, std::shared_ptr<image_process>, object_detect_result_list&)>;

//     npu_infer_pool(std::string model_path, int woeker_size = 1, int bussiness_size = 5)
//         : _woeker_size(woeker_size), _model_path(model_path)
//     {
//         try {
//             for (int i = 0; i < _woeker_size; ++i) {
//                 _models.push_back(std::make_unique<yolov8seg>(_model_path));
//             }
//         } catch (const std::bad_alloc &e) {
//             LOG_ERROR("Out of memory: {}", e.what());
//             exit(EXIT_FAILURE);
//         }
        
//         // yolo模型的初始化
//         for (int i = 0; i < _woeker_size; ++i)
//         {
//             int err = 0;
//             if (i == 0) err = _models[i]->init();
//             else err = _models[i]->init(_models[0]->get_rknn_context());
//             if (err != 0) { LOG_ERROR("Init rknn model failed!"); exit(err); }
//             switch (i % 3) {
//             case 0: _models[i]->set_npu_core(RKNN_NPU_CORE_0); break;
//             case 1: _models[i]->set_npu_core(RKNN_NPU_CORE_1); break;
//             case 2: _models[i]->set_npu_core(RKNN_NPU_CORE_2); break;
//             }
//         }

//         bussiness_pool = std::make_unique<ThreadPool>(bussiness_size);
//         _workers.reserve(_woeker_size);
//         for (int i = 0; i < _woeker_size; ++i)
//         {
//             _workers.emplace_back([this, i] { worker_loop(i); });
//         }
//     }

//     ~npu_infer_pool() { Stop(); }

//     void set_business_callback(BusinessCallback cb) { _biz_callback = cb; }
//     void set_expect_id_ptr(std::shared_ptr<std::atomic<uint64_t>> ptr) { _min_expect_id_ptr = ptr; }

//     void AddInferenceTask(std::unique_ptr<image_process> image_processor)
//     {
//         uint64_t id = _frame_seq.fetch_add(1, std::memory_order_relaxed);
//         _in_queue.push(InferJob{id, std::move(image_processor)});
//     }

//     BlockingQueue<InferOut> &get_npu_infer_out() { return _out_queue; }

//     void Stop() {
//         _in_queue.stop();
//         for (auto &t : _workers) { if (t.joinable()) t.join(); }
//         _workers.clear();
//         _out_queue.stop();
//     }

// private:
// // =====================================================================
//     // 静态辅助函数区 (已适配你的结构体定义)
//     // =====================================================================

//     // 1. 核心算法：掩码椭圆拟合 + Box保底策略
//     // 注意：第三个参数改为 raw pointer，因为我们只是读取数据，不拥有所有权
//     static EllipseResult calculate_best_ellipse(const cv::Mat& frame, const object_detect_result& det_result, uint8_t* mask_data) {
//         EllipseResult result;
//         bool fit_success = false;

//         // --- 策略 1: 尝试利用 Mask 进行高精度拟合 ---
//         if (mask_data) {
//             // 边界保护
//             int x = std::max(0, det_result.x);
//             int y = std::max(0, det_result.y);
//             int w = std::min((int)det_result.w, frame.cols - x);
//             int h = std::min((int)det_result.h, frame.rows - y);
            
//             if (w > 0 && h > 0) {
//                 cv::Rect roi_rect(x, y, w, h);
//                 // 构造全图掩码的 Mat (不拷贝数据)
//                 cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, mask_data);
//                 // 截取 ROI (软引用)
//                 cv::Mat roi_mask = full_mask(roi_rect);

//                 // 查找轮廓 (必须 clone，因为 findContours 会修改图像)
//                 cv::Mat contour_input = roi_mask.clone(); 
//                 std::vector<std::vector<cv::Point>> contours;
//                 cv::findContours(contour_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

//                 if (!contours.empty()) {
//                     auto max_itr = std::max_element(contours.begin(), contours.end(),
//                         [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
//                             return a.size() < b.size();
//                         });

//                     if (max_itr->size() >= 5) {
//                         cv::RotatedRect local_ellipse = cv::fitEllipse(*max_itr);
//                         // 还原坐标到全图
//                         result.rect = local_ellipse;
//                         result.rect.center.x += x;
//                         result.rect.center.y += y;
//                         result.is_from_mask = true;
//                         fit_success = true;
//                     }
//                 }
//             }
//         }

//         // --- 策略 2: Box 保底 ---
//         if (!fit_success) {
//             cv::Point2f center(det_result.x + det_result.w / 2.0f, det_result.y + det_result.h / 2.0f);
//             cv::Size2f size(det_result.w, det_result.h);
//             result.rect = cv::RotatedRect(center, size, 0.0f);
//             result.is_from_mask = false;
//         }

//         return result;
//     }

//     // 2. 像素混合 (内联加速)
//     static inline void blend_pixel(uchar* b, uchar* g, uchar* r, const int* color_weight, float alpha_beta) {
//         *b = cv::saturate_cast<uchar>(*b * alpha_beta + color_weight[0]);
//         *g = cv::saturate_cast<uchar>(*g * alpha_beta + color_weight[1]);
//         *r = cv::saturate_cast<uchar>(*r * alpha_beta + color_weight[2]);
//     }

//     // 3. 绘制结果 (包含 ROI 优化和多核并行)
//     static void draw_results_on_frame(cv::Mat& frame, const object_detect_result_list& result) {
//         float alpha = 0.5f;
//         float beta = 1.0f - alpha;
        
//         // 预计算颜色权重
//         int color_weights[3][3]; 
//         color_weights[0][0] = 0; color_weights[0][1] = 0; color_weights[0][2] = (int)(255 * alpha); // 红
//         color_weights[1][0] = 0; color_weights[1][1] = (int)(255 * alpha); color_weights[1][2] = 0; // 绿
//         color_weights[2][0] = (int)(255 * alpha); color_weights[2][1] = (int)(255 * alpha); color_weights[2][2] = 0; // 青

//         // 获取唯一的 mask 结果对象 (根据你的定义 results_mask 是数组[1])
//         const auto& seg_result = result.results_mask[0];

//         // 遍历所有检测框
//         for (int i = 0; i < result.count; i++) {
//             const auto& det_box = result.results_box[i];
            
//             // A. 画检测框
//             int x = std::max(0, det_box.x);
//             int y = std::max(0, det_box.y);
//             int w = std::min((int)det_box.w, frame.cols - x);
//             int h = std::min((int)det_box.h, frame.rows - y);
//             cv::Rect box(x, y, w, h);
//             cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);

//             // B. 获取有效 Mask 指针
//             // 假设逻辑：如果检测到的物体类别 ID (cls_id) 在 each_of_mask 中有对应的掩码，则使用它
//             // 注意：需要确保 det_box.cls_id 不会越界访问 each_of_mask
//             uint8_t* raw_mask_ptr = nullptr;
//             int mask_idx = -1; // 用于选择颜色

//             // 安全检查：确保 vector 足够大且指针有效
//             int class_id = det_box.cls_id; // 假设用类别ID作为索引
            
//             // 如果你的逻辑是固定的 0,1,2 对应红绿蓝，这里做个简单的映射保护
//             if (class_id >= 0 && class_id < seg_result.each_of_mask.size()) {
//                 if (seg_result.each_of_mask[class_id]) {
//                     raw_mask_ptr = seg_result.each_of_mask[class_id].get();
//                     mask_idx = class_id % 3; // 循环使用3种颜色
//                 }
//             }
            
//             // C. 椭圆拟合与绘制
//             EllipseResult ellipse_res = calculate_best_ellipse(frame, det_box, raw_mask_ptr);
//             cv::Scalar e_color = ellipse_res.is_from_mask ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
//             cv::ellipse(frame, ellipse_res.rect, e_color, 2);
//             // 画中心点
//             cv::circle(frame, ellipse_res.rect.center, 2, cv::Scalar(0,0,255), -1);

//             // D. 极速 Mask 绘制 (只处理 ROI)
//             if (raw_mask_ptr && w > 0 && h > 0 && mask_idx >= 0) {
//                 // 构造全图 mask view
//                 cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, raw_mask_ptr);
                
//                 // OpenMP 并行处理 ROI 区域
//                 #pragma omp parallel for
//                 for (int r = 0; r < h; r++) {
//                     // 获取 ROI 区域在原图中的绝对行号
//                     int abs_row = y + r;
//                     uchar* ptr_img = frame.ptr<uchar>(abs_row);
//                     uchar* ptr_mask = full_mask.ptr<uchar>(abs_row); // mask 行指针

//                     for (int c = 0; c < w; c++) {
//                         int abs_col = x + c;
//                         if (ptr_mask[abs_col] > 0) {
//                              blend_pixel(&ptr_img[3*abs_col], &ptr_img[3*abs_col+1], &ptr_img[3*abs_col+2], color_weights[mask_idx], beta);
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     // =====================================================================
//     // NPU 工作线程
//     // =====================================================================
//     void worker_loop(int model_id)
//     {
//         int err = 0;
//         while (true)
//         {
//             auto jobOpt = _in_queue.pop();
//             if (!jobOpt) break;

//             auto job = std::move(*jobOpt);
//             auto proc = std::move(job.proc); 

//             // NPU 前处理 & 推理
//             proc->image_preprocessing(640, 640);
//             int image_len = 0;
//             uint8_t *buffer = proc->get_image_buffer(&image_len);

//             err = _models[model_id]->set_input_data(buffer, image_len);
//             if (err != RKNN_SUCC) break;
//             err = _models[model_id]->rknn_model_inference();
//             if (err != RKNN_SUCC) break;
//             err = _models[model_id]->get_output_data();
//             if (err != RKNN_SUCC) break;

//             // NPU 后处理
//             object_detect_result_list result;
//             letterbox letter_box = proc->get_letterbox();
//             err = _models[model_id]->post_process(result, letter_box);
//             if (err != RKNN_SUCC) break;

//             // 反压丢帧检查
//             if (_min_expect_id_ptr) {
//                 uint64_t min_needed = _min_expect_id_ptr->load(std::memory_order_relaxed);
//                 if (job.frame_id < min_needed) continue; 
//             }

//             // 转为 shared_ptr
//             std::shared_ptr<image_process> shared_proc = std::move(proc);

//             // =================================================================
//             // 【核心优化】把所有脏活累活扔进业务线程池！
//             // =================================================================
//             bussiness_pool->enqueue(
//                 [this, fid = job.frame_id, p = shared_proc, r = std::move(result)]() mutable
//                 {
//                     // 1. 获取图片引用
//                     cv::Mat& frame_to_draw = *(p->_src_image_frame);

//                     // 2. 在业务线程里画图 (并行执行，不卡 NPU，不卡 UI)
//                     draw_results_on_frame(frame_to_draw, r);

//                     // 3. 执行额外的业务回调 (如数据上报)
//                     if (this->_biz_callback) {
//                          this->_biz_callback(fid, p, r);
//                     }

//                     // 4. 推送到输出队列 -> OrderedProcessor
//                     this->_out_queue.push(InferOut{fid, p, std::move(r)});
//                 });
            
//             // NPU 线程立刻释放，处理下一帧！
//         }
//     }

// private:
//     int _woeker_size;
//     std::string _model_path;
//     std::vector<std::unique_ptr<yolov8seg>> _models;
//     BlockingQueue<InferJob> _in_queue;
//     std::vector<std::thread> _workers;
//     std::atomic<uint64_t> _frame_seq{0};
//     BlockingQueue<InferOut> _out_queue;
//     std::unique_ptr<ThreadPool> bussiness_pool = nullptr;
//     std::shared_ptr<std::atomic<uint64_t>> _min_expect_id_ptr = nullptr;
//     BusinessCallback _biz_callback = nullptr;
// };























#pragma once
#include "yolov8seg.hpp"
#include "thread_pool.hpp"
#include "block_queue.hpp"
#include <image_process.hpp>
#include <string>
#include <memory>
#include <vector>
#include <atomic>
#include <functional>
#include <opencv2/opencv.hpp>
#include <cmath> // 用于计算距离

// 结构体保持不变
struct InferOut {
    uint64_t frame_id;
    std::shared_ptr<image_process> proc;
    object_detect_result_list result;
};

struct InferJob {
    uint64_t frame_id;
    std::unique_ptr<image_process> proc;
};

struct EllipseResult {
    cv::RotatedRect rect;
    bool is_from_mask; 
};

class npu_infer_pool
{
public:
    using BusinessCallback = std::function<void(uint64_t, std::shared_ptr<image_process>, object_detect_result_list&)>;

    npu_infer_pool(std::string model_path, int woeker_size = 1, int bussiness_size = 5)
        : _woeker_size(woeker_size), _model_path(model_path)
    {
        // 1. 加载模型
        try {
            for (int i = 0; i < _woeker_size; ++i) {
                _models.push_back(std::make_unique<yolov8seg>(_model_path));
            }
        } catch (const std::bad_alloc &e) { exit(EXIT_FAILURE); }
        
        // 2. 初始化模型
        for (int i = 0; i < _woeker_size; ++i) {
             int err = 0;
            if (i == 0) err = _models[i]->init();
            else err = _models[i]->init(_models[0]->get_rknn_context());
            if (err != 0) exit(err);
            switch (i % 3) {
            case 0: _models[i]->set_npu_core(RKNN_NPU_CORE_0); break;
            case 1: _models[i]->set_npu_core(RKNN_NPU_CORE_1); break;
            case 2: _models[i]->set_npu_core(RKNN_NPU_CORE_2); break;
            }
        }

        // 3. 启动线程池
        bussiness_pool = std::make_unique<ThreadPool>(bussiness_size);
        _workers.reserve(_woeker_size);
        for (int i = 0; i < _woeker_size; ++i) _workers.emplace_back([this, i] { worker_loop(i); });
    }

    ~npu_infer_pool() { Stop(); }

    void set_business_callback(BusinessCallback cb) { _biz_callback = cb; }
    void set_expect_id_ptr(std::shared_ptr<std::atomic<uint64_t>> ptr) { _min_expect_id_ptr = ptr; }
    
    // 【设置】类别2是否优先使用掩码拟合 (true=尝试拟合, false=强制内切圆)
    void set_class2_mask_fit_mode(bool enable) { _enable_mask_fit_for_class2 = enable; }

    // 【新增】设置偏差阈值
    // 默认 0.3 (即允许圆心偏离检测框短边长度的 30%)
    // 如果 mask 拟合的圆心跑太远，就会被强制拉回使用内切圆
    void set_deviation_threshold(float ratio) { _max_deviation_ratio = ratio; }

    void AddInferenceTask(std::unique_ptr<image_process> image_processor) {
        uint64_t id = _frame_seq.fetch_add(1, std::memory_order_relaxed);
        _in_queue.push(InferJob{id, std::move(image_processor)});
    }

    BlockingQueue<InferOut> &get_npu_infer_out() { return _out_queue; }

    void Stop() {
        _in_queue.stop();
        for (auto &t : _workers) { if (t.joinable()) t.join(); }
        _workers.clear();
        _out_queue.stop();
    }

private:
    // =====================================================================
    // 静态辅助函数区
    // =====================================================================

    // 1. 核心算法：椭圆拟合 (含偏差校验)
    static EllipseResult calculate_best_ellipse(const cv::Mat& frame, 
                                              const object_detect_result& det_result, 
                                              uint8_t* mask_data,
                                              bool force_box_mode,
                                              float deviation_threshold) 
    {
        EllipseResult result;
        bool fit_success = false;

        // 预先计算 Box 的中心点，用于后面的校验和保底
        cv::Point2f box_center(det_result.x + det_result.w / 2.0f, det_result.y + det_result.h / 2.0f);

        // --- 策略 1: 尝试 Mask 拟合 ---
        if (mask_data && !force_box_mode) {
            int x = std::max(0, det_result.x);
            int y = std::max(0, det_result.y);
            int w = std::min((int)det_result.w, frame.cols - x);
            int h = std::min((int)det_result.h, frame.rows - y);
            
            if (w > 0 && h > 0) {
                cv::Rect roi_rect(x, y, w, h);
                cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, mask_data);
                cv::Mat roi_mask = full_mask(roi_rect);
                cv::Mat contour_input = roi_mask.clone(); 
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(contour_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                if (!contours.empty()) {
                    auto max_itr = std::max_element(contours.begin(), contours.end(),
                        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                            return a.size() < b.size();
                        });

                    if (max_itr->size() >= 5) {
                        cv::RotatedRect local_ellipse = cv::fitEllipse(*max_itr);
                        
                        // 转换坐标
                        result.rect = local_ellipse;
                        result.rect.center.x += x;
                        result.rect.center.y += y;

                        // =================================================
                        // 【新增】偏差校验逻辑
                        // =================================================
                        // 计算 Mask拟合中心 与 Box中心 的距离
                        float dist_sq = std::pow(result.rect.center.x - box_center.x, 2) + 
                                        std::pow(result.rect.center.y - box_center.y, 2);
                        float dist = std::sqrt(dist_sq);
                        
                        // 计算允许的最大偏移量 (基于 Box 的短边)
                        float limit_dist = std::min(det_result.w, det_result.h) * deviation_threshold;

                        if (dist <= limit_dist) {
                            result.is_from_mask = true;
                            fit_success = true;
                        } else {56
                            // 偏离太远，fit_success 保持 false，自动掉入下面的保底逻辑
                            // std::cout << "Mask ellipse rejected! Deviation: " << dist << " > " << limit_dist << std::endl;
                        }
                    }
                }
            }
        }

        // --- 策略 2: Box 保底 (内切椭圆/圆) ---
        // 触发条件：强制模式 OR 掩码不存在 OR 掩码拟合失败 OR 偏差太大
        if (!fit_success) {
            cv::Size2f size(det_result.w, det_result.h);
            // 构造 RotatedRect, 角度为0，中心为 Box 中心
            result.rect = cv::RotatedRect(box_center, size, 0.0f);
            result.is_from_mask = false; 
        }

        return result;
    }

    // 2. 像素混合
    static inline void blend_pixel(uchar* b, uchar* g, uchar* r, const int* color_weight, float alpha_beta) {
        *b = cv::saturate_cast<uchar>(*b * alpha_beta + color_weight[0]);
        *g = cv::saturate_cast<uchar>(*g * alpha_beta + color_weight[1]);
        *r = cv::saturate_cast<uchar>(*r * alpha_beta + color_weight[2]);
    }

    // 3. 绘制结果
    static void draw_results_on_frame(cv::Mat& frame, 
                                    const object_detect_result_list& result,
                                    bool use_mask_fit_class2,
                                    float deviation_threshold) 
    {
        float alpha = 0.5f;
        float beta = 1.0f - alpha;
        int color_weights[3][3]; 
        color_weights[0][0] = 0; color_weights[0][1] = 0; color_weights[0][2] = (int)(255 * alpha); // 红
        color_weights[1][0] = 0; color_weights[1][1] = (int)(255 * alpha); color_weights[1][2] = 0; // 绿
        color_weights[2][0] = (int)(255 * alpha); color_weights[2][1] = (int)(255 * alpha); color_weights[2][2] = 0; // 青

        const auto& seg_result = result.results_mask[0];

        for (int i = 0; i < result.count; i++) {
            const auto& det_box = result.results_box[i];
            
            // A. 画检测框
            int x = std::max(0, det_box.x);
            int y = std::max(0, det_box.y);
            int w = std::min((int)det_box.w, frame.cols - x);
            int h = std::min((int)det_box.h, frame.rows - y);
            cv::Rect box(x, y, w, h);
            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);

            // B. 准备 Mask
            uint8_t* raw_mask_ptr = nullptr;
            int mask_idx = -1;
            int class_id = det_box.cls_id;
            
            if (class_id >= 0 && class_id < (int)seg_result.each_of_mask.size()) {
                if (seg_result.each_of_mask[class_id]) {
                    raw_mask_ptr = seg_result.each_of_mask[class_id].get();
                    mask_idx = class_id % 3;
                }
            }

            // C. 椭圆拟合与绘制 (逻辑判断)
            bool force_box = false;
            // 如果是类别2，并且开关被关闭，则强制使用 Box 拟合
            if (class_id == 2 && !use_mask_fit_class2) {
                force_box = true;
            }

            // 【传入 deviation_threshold 参数】
            EllipseResult ellipse_res = calculate_best_ellipse(frame, det_box, raw_mask_ptr, force_box, deviation_threshold);
            
            // Mask拟合用绿色，Box保底用黄色
            cv::Scalar e_color = ellipse_res.is_from_mask ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
            cv::ellipse(frame, ellipse_res.rect, e_color, 2);
            cv::circle(frame, ellipse_res.rect.center, 2, cv::Scalar(0,0,255), -1);

            // D. 极速 Mask 绘制
            if (raw_mask_ptr && w > 0 && h > 0 && mask_idx >= 0) {
                cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, raw_mask_ptr);
                #pragma omp parallel for
                for (int r = 0; r < h; r++) {
                    int abs_row = y + r;
                    uchar* ptr_img = frame.ptr<uchar>(abs_row);
                    uchar* ptr_mask = full_mask.ptr<uchar>(abs_row);
                    for (int c = 0; c < w; c++) {
                        int abs_col = x + c;
                        if (ptr_mask[abs_col] > 0) {
                             blend_pixel(&ptr_img[3*abs_col], &ptr_img[3*abs_col+1], &ptr_img[3*abs_col+2], color_weights[mask_idx], beta);
                        }
                    }
                }
            }
        }
    }

    // =====================================================================
    // NPU 工作线程
    // =====================================================================
    void worker_loop(int model_id)
    {
        int err = 0;
        while (true)
        {
            auto jobOpt = _in_queue.pop();
            if (!jobOpt) break;

            auto job = std::move(*jobOpt);
            auto proc = std::move(job.proc); 

            proc->image_preprocessing(640, 640);
            int image_len = 0;
            uint8_t *buffer = proc->get_image_buffer(&image_len);
            if (_models[model_id]->set_input_data(buffer, image_len) != RKNN_SUCC) break;
            if (_models[model_id]->rknn_model_inference() != RKNN_SUCC) break;
            if (_models[model_id]->get_output_data() != RKNN_SUCC) break;
            object_detect_result_list result;
            letterbox letter_box = proc->get_letterbox();
            if (_models[model_id]->post_process(result, letter_box) != RKNN_SUCC) break;

            if (_min_expect_id_ptr) {
                uint64_t min_needed = _min_expect_id_ptr->load(std::memory_order_relaxed);
                if (job.frame_id < min_needed) continue; 
            }

            std::shared_ptr<image_process> shared_proc = std::move(proc);

            bussiness_pool->enqueue(
                [this, fid = job.frame_id, p = shared_proc, r = std::move(result)]() mutable
                {
                    cv::Mat& frame_to_draw = *(p->_src_image_frame);
                    
                    // 【调用时传入 2 个控制参数】
                    // 1. 类别2是否使用掩码模式
                    // 2. 允许的圆心最大偏差阈值
                    draw_results_on_frame(frame_to_draw, r, 
                                          this->_enable_mask_fit_for_class2, 
                                          this->_max_deviation_ratio);

                    if (this->_biz_callback) {
                         this->_biz_callback(fid, p, r);
                    }
                    this->_out_queue.push(InferOut{fid, p, std::move(r)});
                });
        }
    }

private:
    int _woeker_size;
    std::string _model_path;
    std::vector<std::unique_ptr<yolov8seg>> _models;
    BlockingQueue<InferJob> _in_queue;
    std::vector<std::thread> _workers;
    std::atomic<uint64_t> _frame_seq{0};
    BlockingQueue<InferOut> _out_queue;
    std::unique_ptr<ThreadPool> bussiness_pool = nullptr;
    std::shared_ptr<std::atomic<uint64_t>> _min_expect_id_ptr = nullptr;
    BusinessCallback _biz_callback = nullptr;

    // 参数控制
    bool _enable_mask_fit_for_class2 = true; 
    float _max_deviation_ratio = 0.3f; // 默认允许 30% 的偏差
};