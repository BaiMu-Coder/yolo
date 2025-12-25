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
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <type_traits>
#include <utility>



#include "pose_estimator_lm.hpp"

// 输出结构体 (使用 shared_ptr)
struct InferOut {
    uint64_t frame_id;
    std::shared_ptr<image_process> proc;
    object_detect_result_list result;
};

// 输入任务结构体
struct InferJob {
    uint64_t frame_id;
    std::unique_ptr<image_process> proc;
};

// 椭圆拟合结果结构体
struct EllipseResult {
    cv::RotatedRect rect;
    bool is_from_mask; // true=掩码拟合, false=Box保底
};

class npu_infer_pool
{
public:
    // 业务回调类型
    using BusinessCallback = std::function<void(uint64_t, std::shared_ptr<image_process>, object_detect_result_list&)>;

    npu_infer_pool(std::string model_path, int woeker_size = 1, int bussiness_size = 5)
        : _woeker_size(woeker_size), _model_path(std::move(model_path))
    {
        // 1. 加载模型
        try {
            for (int i = 0; i < _woeker_size; ++i) {
                _models.push_back(std::make_unique<yolov8seg>(_model_path));
            }
        } catch (const std::bad_alloc&) {
            LOG_ERROR("Out of memory while creating models");
            exit(EXIT_FAILURE);
        }

        // 2. 初始化模型
        for (int i = 0; i < _woeker_size; ++i) {
            int err = 0;
            if (i == 0) err = _models[i]->init();
            else err = _models[i]->init(_models[0]->get_rknn_context());
            if (err != 0) { LOG_ERROR("Init rknn model failed!"); exit(err); }

            switch (i % 3) {
            case 0: _models[i]->set_npu_core(RKNN_NPU_CORE_0); break;
            case 1: _models[i]->set_npu_core(RKNN_NPU_CORE_1); break;
            case 2: _models[i]->set_npu_core(RKNN_NPU_CORE_2); break;
            }
        }

        // 3. 初始化姿态解算默认参数（与你 Python 默认一致）
        // 相机：FX=1639.6 FY=2165.4 CX=960 CY=540；D=0
        _K = (cv::Mat_<double>(3,3) << 1639.6, 0, 960,  0, 2165.4, 540,  0,0,1);
        _D = cv::Mat::zeros(4,1,CV_64F);

        // 物理模型：k=20/45.5；radius_cls0=1200*k, radius_cls1=980*k, radius_hole=120*k, length_L=920*k
        {
            const double k = 20.0 / 45.5;
            _drogue.radius_cls0_mm = 1200.0 * k;
            _drogue.radius_cls1_mm =  980.0 * k;
            _drogue.radius_hole_mm =  120.0 * k;
            _drogue.length_L_mm    =  920.0 * k;
        }
        _pose_estimator.Reset(_K, _D, _drogue);

        // 4. 启动线程池
        bussiness_pool = std::make_unique<ThreadPool>(bussiness_size);
        _workers.reserve(_woeker_size);
        for (int i = 0; i < _woeker_size; ++i) _workers.emplace_back([this, i] { worker_loop(i); });
    }

    ~npu_infer_pool() { Stop(); }

    void set_business_callback(BusinessCallback cb) { _biz_callback = std::move(cb); }
    void set_expect_id_ptr(std::shared_ptr<std::atomic<uint64_t>> ptr) { _min_expect_id_ptr = std::move(ptr); }

    // ===== 你原来已有的控制项 =====
    // 类别2是否优先使用掩码拟合 (true=尝试拟合, false=强制Box/内切圆)
    void set_class2_mask_fit_mode(bool enable) { _enable_mask_fit_for_class2 = enable; }

    // 允许的圆心最大偏差（相对短边比例），默认 0.3
    void set_deviation_threshold(float ratio) { _max_deviation_ratio = ratio; }

    // ===== 新增：姿态/深度 双轨显示控制 =====
    // display_fixed=true：画面显示固定距离解算；false：显示自动估计 tz
    void set_pose_display_fixed(bool display_fixed) { _display_fixed_mode = display_fixed; }
    void set_pose_fixed_distance_mm(double fixed_mm) { _fixed_distance_mm = fixed_mm; }

    // 可选：如果你要改相机参数/物理模型（建议你初始化后调用一次）
    void set_camera_params(const cv::Mat& K, const cv::Mat& D) {
        _K = K.clone(); _D = D.clone();
        _pose_estimator.Reset(_K, _D, _drogue);
    }
    void set_drogue_model(const DrogueModel& m) {
        _drogue = m;
        _pose_estimator.Reset(_K, _D, _drogue);
    }

    void AddInferenceTask(std::unique_ptr<image_process> image_processor) {
        uint64_t id = _frame_seq.fetch_add(1, std::memory_order_relaxed);
        _in_queue.push(InferJob{id, std::move(image_processor)});
    }

    BlockingQueue<InferOut>& get_npu_infer_out() { return _out_queue; }

    void Stop() {
        _in_queue.stop();
        for (auto &t : _workers) { if (t.joinable()) t.join(); }
        _workers.clear();
        _out_queue.stop();
    }

private:
    // ============================================================
    // score 提取：兼容不同 object_detect_result 字段名
    // ============================================================
    template<class T, class = void> struct has_score : std::false_type {};
    template<class T> struct has_score<T, std::void_t<decltype(std::declval<T>().score)>> : std::true_type {};
    template<class T, class = void> struct has_prop : std::false_type {};
    template<class T> struct has_prop<T, std::void_t<decltype(std::declval<T>().prop)>> : std::true_type {};
    template<class T, class = void> struct has_conf : std::false_type {};
    template<class T> struct has_conf<T, std::void_t<decltype(std::declval<T>().conf)>> : std::true_type {};
    template<class T, class = void> struct has_probability : std::false_type {};
    template<class T> struct has_probability<T, std::void_t<decltype(std::declval<T>().probability)>> : std::true_type {};

  // ============================================================
// 置信度提取：你的 object_detect_result 只有 prop
// ============================================================
static inline double det_score(const object_detect_result& d) {
    return (double)d.prop;
}


    static int pick_best_idx_by_class(const object_detect_result_list& result, int cls_id) {
        int best = -1;
        double best_s = -1e18;
        for (int i = 0; i < result.count; ++i) {
            const auto& d = result.results_box[i];
            if (d.cls_id != cls_id) continue;
            double s = det_score(d);
            if (best < 0 || s > best_s) { best = i; best_s = s; }
        }
        return best;
    }

    // ============================================================
    // 椭圆拟合（保留你已有的 deviation 校验，并修复你文件里的 else {56）
    // 同时 Box 保底改成 “内切圆/正方形” 更贴近你的 Python 逻辑
    // ============================================================
    static EllipseResult calculate_best_ellipse(const cv::Mat& frame,
                                                const object_detect_result& det_result,
                                                uint8_t* mask_data,
                                                bool force_box_mode,
                                                float deviation_threshold_ratio)
    {
        EllipseResult result{};
        bool fit_success = false;

        cv::Point2f box_center(det_result.x + det_result.w / 2.0f,
                               det_result.y + det_result.h / 2.0f);

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

                    if ((int)max_itr->size() >= 5) {
                        cv::RotatedRect local_ellipse = cv::fitEllipse(*max_itr);

                        result.rect = local_ellipse;
                        result.rect.center.x += (float)x;
                        result.rect.center.y += (float)y;

                        // 偏差校验：mask 椭圆中心不能离 box_center 太远
                        float dist = std::hypot(result.rect.center.x - box_center.x,
                                                result.rect.center.y - box_center.y);

                        float limit_dist = std::min((float)det_result.w, (float)det_result.h) * deviation_threshold_ratio;

                        if (dist <= limit_dist) {
                            result.is_from_mask = true;
                            fit_success = true;
                        }
                    }
                }
            }
        }

        // --- 策略 2: Box 保底（内切圆/正方形）---
        if (!fit_success) {
            float min_s = std::min((float)det_result.w, (float)det_result.h);
            cv::Size2f size(min_s, min_s);
            result.rect = cv::RotatedRect(box_center, size, 0.0f);
            result.is_from_mask = false;
        }

        return result;
    }

    // 像素混合
    static inline void blend_pixel(uchar* b, uchar* g, uchar* r, const int* color_weight, float alpha_beta) {
        *b = cv::saturate_cast<uchar>(*b * alpha_beta + color_weight[0]);
        *g = cv::saturate_cast<uchar>(*g * alpha_beta + color_weight[1]);
        *r = cv::saturate_cast<uchar>(*r * alpha_beta + color_weight[2]);
    }

    static inline void draw_txt(cv::Mat& img, const std::string& text, int y,
                                const cv::Scalar& color, double scale=1.2, int thickness=2)
    {
        cv::putText(img, text, {20,y}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0,0,0), thickness + 3);
        cv::putText(img, text, {20,y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
    }

    // ============================================================
    // 绘制结果：你原来的 det + mask 绘制保留
    // 在最后追加：cls0/1/2 择优 + PoseSolve + Dist
    // ============================================================
    void draw_results_on_frame(cv::Mat& frame,
                               const object_detect_result_list& result,
                               bool use_mask_fit_class2,
                               float deviation_threshold_ratio)
    {
        // 颜色
        const cv::Scalar C_GRAY(100,100,100);
        const cv::Scalar C_CYAN(255,255,0);
        const cv::Scalar C_GRN(0,255,0);
        const cv::Scalar C_YEL(0,255,255);
        const cv::Scalar C_WHT(255,255,255);

        // ====== A) 你原来的 det + mask + 单体椭圆绘制 ======
        float alpha = 0.5f;
        float beta = 1.0f - alpha;
        int color_weights[3][3];
        color_weights[0][0] = 0;              color_weights[0][1] = 0;              color_weights[0][2] = (int)(255 * alpha); // 红
        color_weights[1][0] = 0;              color_weights[1][1] = (int)(255 * alpha); color_weights[1][2] = 0;              // 绿
        color_weights[2][0] = (int)(255*alpha); color_weights[2][1] = (int)(255*alpha); color_weights[2][2] = 0;              // 青

        const auto& seg_result = result.results_mask[0];

        for (int i = 0; i < result.count; i++) {
            const auto& det_box = result.results_box[i];

            int x = std::max(0, det_box.x);
            int y = std::max(0, det_box.y);
            int w = std::min((int)det_box.w, frame.cols - x);
            int h = std::min((int)det_box.h, frame.rows - y);
            cv::Rect box(x, y, w, h);
            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);

            // mask 指针
            uint8_t* raw_mask_ptr = nullptr;
            int mask_idx = -1;
            int class_id = det_box.cls_id;
            if (class_id >= 0 && class_id < (int)seg_result.each_of_mask.size()) {
                if (seg_result.each_of_mask[class_id]) {
                    raw_mask_ptr = seg_result.each_of_mask[class_id].get();
                    mask_idx = class_id % 3;
                }
            }

            // 类别2是否强制 box
            bool force_box = false;
            if (class_id == 2 && !use_mask_fit_class2) force_box = true;

            EllipseResult ellipse_res = calculate_best_ellipse(frame, det_box, raw_mask_ptr, force_box, deviation_threshold_ratio);

            cv::Scalar e_color = ellipse_res.is_from_mask ? cv::Scalar(0,255,0) : cv::Scalar(0,255,255);
            cv::ellipse(frame, ellipse_res.rect, e_color, 2);
            cv::circle(frame, ellipse_res.rect.center, 2, cv::Scalar(0,0,255), -1);

            // ROI mask 上色
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
                            blend_pixel(&ptr_img[3*abs_col], &ptr_img[3*abs_col+1], &ptr_img[3*abs_col+2],
                                        color_weights[mask_idx], beta);
                        }
                    }
                }
            }
        }

        // ====== B) 追加：cls0/1/2 择优 + 姿态解算 + 深度 ======
        const int CLS0_ID = 0; // 外圈
        const int CLS1_ID = 1; // 中圈
        const int CLS2_ID = 2; // 内孔

        int idx0 = pick_best_idx_by_class(result, CLS0_ID);
        int idx1 = pick_best_idx_by_class(result, CLS1_ID);
        int idx2 = pick_best_idx_by_class(result, CLS2_ID);

        // Python 同逻辑：必须有孔 + (外圈或中圈)
        if (idx2 < 0 || (idx0 < 0 && idx1 < 0)) {
            draw_txt(frame, "Pose: --", 190, C_CYAN, 1.2);
            draw_txt(frame, "Dist: --", 240, C_CYAN, 1.2);
            return;
        }

        auto mask_ptr_of = [&](int cls_id)->uint8_t* {
            if (cls_id < 0 || cls_id >= (int)seg_result.each_of_mask.size()) return nullptr;
            if (!seg_result.each_of_mask[cls_id]) return nullptr;
            return seg_result.each_of_mask[cls_id].get();
        };

        // cand0 / cand1
        bool has0 = (idx0 >= 0);
        bool has1 = (idx1 >= 0);

        EllipseResult cand0{}, cand1{};
        if (has0) cand0 = calculate_best_ellipse(frame, result.results_box[idx0], mask_ptr_of(CLS0_ID), false, deviation_threshold_ratio);
        if (has1) cand1 = calculate_best_ellipse(frame, result.results_box[idx1], mask_ptr_of(CLS1_ID), false, deviation_threshold_ratio);

        // hole（cls2）：可选用 mask 拟合；如果开关关了就强制 box（和你现有代码一致）
        bool force_box2 = !use_mask_fit_class2;
        EllipseResult hole_e = calculate_best_ellipse(frame, result.results_box[idx2], mask_ptr_of(CLS2_ID), force_box2, deviation_threshold_ratio);
        cv::Point2f hole_center = hole_e.rect.center;
        float hole_radius = 0.5f * std::min(hole_e.rect.size.width, hole_e.rect.size.height);

        // 择优策略：优先 cand0=Mask，否则 cand1=Mask，否则 cand0（按你 Python）
        cv::RotatedRect target;
        bool use_cls1 = false;
        bool ok_target = false;

        if (has0 && has1) {
            if (cand0.is_from_mask) { target = cand0.rect; use_cls1=false; ok_target=true; }
            else if (cand1.is_from_mask) { target = cand1.rect; use_cls1=true; ok_target=true; }
            else { target = cand0.rect; use_cls1=false; ok_target=true; }
        } else if (has0) { target = cand0.rect; use_cls1=false; ok_target=true; }
        else if (has1) { target = cand1.rect; use_cls1=true; ok_target=true; }

        if (!ok_target) return;

        // 备选灰色
        if (has0) cv::ellipse(frame, cand0.rect, C_GRAY, 1);
        if (has1) cv::ellipse(frame, cand1.rect, C_GRAY, 1);
        if (hole_radius > 1) cv::circle(frame, hole_center, (int)hole_radius, C_GRAY, 1);

        // 正主高亮
        cv::ellipse(frame, target, C_CYAN, 4);
        cv::circle(frame, hole_center, 6, C_CYAN, -1);
        if (hole_radius > 1) cv::circle(frame, hole_center, (int)hole_radius, C_CYAN, 3);
        cv::line(frame, target.center, hole_center, C_CYAN, 3);

        // 双轨解算
        Pose6D pose_auto = _pose_estimator.Solve(target, hole_center, use_cls1, std::nullopt);
        Pose6D pose_fix  = _pose_estimator.Solve(target, hole_center, use_cls1, _fixed_distance_mm);
        Pose6D pose_final = _display_fixed_mode ? pose_fix : pose_auto;

        // 画轴
        _pose_estimator.DrawAxis(frame, pose_final, use_cls1);

        // UI 文本
        {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "Ref(%s): (%.0f, %.0f)",
                          use_cls1 ? "Mid" : "Out",
                          target.center.x, target.center.y);
            draw_txt(frame, buf, 100, C_WHT, 0.8);

            std::snprintf(buf, sizeof(buf), "Hole: (%.0f, %.0f)", hole_center.x, hole_center.y);
            draw_txt(frame, buf, 140, C_WHT, 0.8);

            if (_display_fixed_mode) {
                std::snprintf(buf, sizeof(buf), "Yaw:%.1f Pit:%.1f", pose_fix.yaw_deg, pose_fix.pitch_deg);
                draw_txt(frame, buf, 190, C_YEL, 1.3);
                std::snprintf(buf, sizeof(buf), "Dist: %.2fm (Fixed)", pose_fix.tz_mm / 1000.0);
                draw_txt(frame, buf, 240, C_YEL, 1.3);
            } else {
                std::snprintf(buf, sizeof(buf), "Yaw:%.1f Pit:%.1f", pose_auto.yaw_deg, pose_auto.pitch_deg);
                draw_txt(frame, buf, 190, C_GRN, 1.3);
                std::snprintf(buf, sizeof(buf), "Dist: %.2fm (Auto)", pose_auto.tz_mm / 1000.0);
                draw_txt(frame, buf, 240, C_GRN, 1.3);
            }
        }
    }

    // ============================================================
    // NPU 工作线程（结构保持你原样，只把 draw 调用从 static 改成成员函数调用）
    // ============================================================
    void worker_loop(int model_id)
    {
        while (true)
        {
            auto jobOpt = _in_queue.pop();
            if (!jobOpt) break;

            auto job = std::move(*jobOpt);
            auto proc = std::move(job.proc);

            // NPU 前处理 & 推理
            proc->image_preprocessing(640, 640);
            int image_len = 0;
            uint8_t* buffer = proc->get_image_buffer(&image_len);

            if (_models[model_id]->set_input_data(buffer, image_len) != RKNN_SUCC) break;
            if (_models[model_id]->rknn_model_inference() != RKNN_SUCC) break;
            if (_models[model_id]->get_output_data() != RKNN_SUCC) break;

            // NPU 后处理
            object_detect_result_list result;
            letterbox letter_box = proc->get_letterbox();
            if (_models[model_id]->post_process(result, letter_box) != RKNN_SUCC) break;

            // 反压丢帧检查
            if (_min_expect_id_ptr) {
                uint64_t min_needed = _min_expect_id_ptr->load(std::memory_order_relaxed);
                if (job.frame_id < min_needed) continue;
            }

            std::shared_ptr<image_process> shared_proc = std::move(proc);

            bussiness_pool->enqueue(
                [this, fid = job.frame_id, p = shared_proc, r = std::move(result)]() mutable
                {
                    cv::Mat& frame_to_draw = *(p->_src_image_frame);

                    // 业务线程绘制（含 Pose/Depth）
                    this->draw_results_on_frame(frame_to_draw, r,
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

    // ===== 你原来的控制参数 =====
    bool _enable_mask_fit_for_class2 = true;
    float _max_deviation_ratio = 0.3f;

    // ===== 新增：Pose/Depth 控制参数 =====
    bool _display_fixed_mode = false;
    double _fixed_distance_mm = 3000.0;

    // ===== 新增：PoseEstimator =====
    cv::Mat _K, _D;
    DrogueModel _drogue;
    PoseEstimatorLM _pose_estimator;
};
