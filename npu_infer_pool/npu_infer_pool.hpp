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
#include <random>
#include <mutex>

#include "pose_estimator_lm.hpp"
#include "ellipse_fitter.hpp"

// 输出结构体 (使用 shared_ptr)
struct InferOut
{
    uint64_t frame_id;
    std::shared_ptr<image_process> proc;
    object_detect_result_list result;
};

// 输入任务结构体
struct InferJob
{
    uint64_t frame_id;
    std::unique_ptr<image_process> proc;
};

class npu_infer_pool
{
public:
    // 业务回调类型
    using BusinessCallback = std::function<void(uint64_t, std::shared_ptr<image_process>, object_detect_result_list &)>;

    npu_infer_pool(std::string model_path, int woeker_size = 3, int bussiness_size = 3)
        : _woeker_size(woeker_size), _model_path(std::move(model_path))
    {
        // 1. 加载模型
        try
        {
            for (int i = 0; i < _woeker_size; ++i)
            {
                _models.push_back(std::make_unique<yolov8seg>(_model_path));
            }
        }
        catch (const std::bad_alloc &)
        {
            LOG_ERROR("Out of memory while creating models");
            exit(EXIT_FAILURE);
        }

        // 2. 初始化模型
        for (int i = 0; i < _woeker_size; ++i)
        {
            int err = 0;
            if (i == 0)
                err = _models[i]->init();
            else
                err = _models[i]->init(_models[0]->get_rknn_context());
            if (err != 0)
            {
                LOG_ERROR("Init rknn model failed!");
                exit(err);
            }

            switch (i % 3)
            {
            case 0:
                _models[i]->set_npu_core(RKNN_NPU_CORE_0);
                break;
            case 1:
                _models[i]->set_npu_core(RKNN_NPU_CORE_1);
                break;
            case 2:
                _models[i]->set_npu_core(RKNN_NPU_CORE_2);
                break;
            }
        }

        // 3. 初始化姿态解算默认参数（与 Python 默认一致）
        // 相机：FX=1639.6 FY=2165.4 CX=960 CY=540；D=0     //_K相机内参 和 _D畸变系数
        _K = (cv::Mat_<double>(3, 3) << 1639.6, 0, 960, 0, 2165.4, 540, 0, 0, 1); // cv::Mat_<double>(3,3)  创建一个3x3的矩阵 。  <<：后面数字依次填入矩阵，行优先
        _D = cv::Mat::zeros(4, 1, CV_64F);                                        // 创建一个4行1列的全零矩阵，64位浮点数

        // 物理模型：k=20/45.5；radius_cls0=1200*k, radius_cls1=980*k, radius_hole=120*k, length_L=920*k
        {
            const double k = 20.0 / 45.5; // 缩放倍数，用于调节真实大小
            _drogue.radius_cls0_mm = 1200.0 * k;
            _drogue.radius_cls1_mm = 980.0 * k;
            _drogue.radius_hole_mm = 120.0 * k;
            _drogue.length_L_mm = 920.0 * k;
        }
        _pose_estimator.Reset(_K, _D, _drogue);

        // 4. 启动业务线程池
        bussiness_pool = std::make_unique<ThreadPool>(bussiness_size);
        _workers.reserve(_woeker_size);
        for (int i = 0; i < _woeker_size; ++i)
            _workers.emplace_back([this, i]
                                  { worker_loop(i); });
    }

    ~npu_infer_pool() { Stop(); }

    void set_business_callback(BusinessCallback cb)
    {
        _biz_callback = std::move(cb);
    }

    // 把一个“外部可更新的期望帧号（expect id）”共享给 npu_infer_pool，用来在 pool 内部丢弃过期帧
    void set_expect_id_ptr(std::shared_ptr<std::atomic<uint64_t>> ptr)
    {
        _min_expect_id_ptr = std::move(ptr);
    }

    // 参考外圈（cls0/cls1）是否强制使用检测框内切圆。
    // 验收现场图像质量较差时可一键切换；默认 false，仍优先使用 Mask 拟合。
    void set_reference_ring_force_box_mode(bool force_box)
    {
        _force_box_for_reference_ring = force_box;
    }

    // 分类别控制参考圈。true=尝试 Mask 拟合，false=强制检测框内切圆。
    // 当 set_reference_ring_force_box_mode(true) 时，这两个开关会被全局强制模式覆盖。
    void set_class0_mask_fit_mode(bool enable)
    {
        _enable_mask_fit_for_class0 = enable;
    }

    void set_class1_mask_fit_mode(bool enable)
    {
        _enable_mask_fit_for_class1 = enable;
    }

    // 类别2是否使用掩码拟合 (true=尝试拟合, false=强制Box/内切圆)
    // 内孔 Mask 边界误差相对较大，默认 false。
    void set_class2_mask_fit_mode(bool enable)
    {
        _enable_mask_fit_for_class2 = enable;
    }

    // 允许的圆心最大偏差（相对短边比例），默认 0.3
    void set_deviation_threshold(float ratio)
    {
        _max_deviation_ratio = ratio;
    }

    void set_temporal_filter_enabled(bool enable)
    {
        _enable_temporal_filter = enable;
        if (!enable)
        {
            std::lock_guard<std::mutex> lock(_ellipse_filter_mutex);
            _ellipse_temporal_filter.Reset();
            _last_filtered_frame_id = 0;
            _has_filtered_frame = false;
        }
    }

    // =====姿态/深度 双轨显示控制 =====
    // display_fixed=true：画面显示固定距离解算；false：显示自动估计 tz
    void set_pose_display_fixed(bool display_fixed)
    {
        _display_fixed_mode = display_fixed;
    }

    void set_pose_fixed_distance_mm(double fixed_mm)
    {
        _fixed_distance_mm = fixed_mm;
    }

    // 改相机参数/物理模型（建议初始化后调用一次）
    void set_camera_params(const cv::Mat &K, const cv::Mat &D)
    {
        _K = K.clone();
        _D = D.clone();
        _pose_estimator.Reset(_K, _D, _drogue);
    }

    // 改变锥套的形状大小 接口
    void set_drogue_model(const DrogueModel &m)
    {
        _drogue = m;
        _pose_estimator.Reset(_K, _D, _drogue);
    }

    void AddInferenceTask(std::unique_ptr<image_process> image_processor)
    {
        uint64_t id = _frame_seq.fetch_add(1, std::memory_order_relaxed);
        _in_queue.push(InferJob{id, std::move(image_processor)});
    }

    BlockingQueue<InferOut> &get_npu_infer_out() { return _out_queue; }

    void Stop()
    {
        _in_queue.stop();
        for (auto &t : _workers)
        {
            if (t.joinable())
                t.join();
        }
        _workers.clear();
        _out_queue.stop();
    }

private:
    // ============================================================
    // score 提取：兼容不同 object_detect_result 字段名
    // ============================================================
    template <class T, class = void>
    struct has_score : std::false_type
    {
    };

    template <class T>
    struct has_score<T, std::void_t<decltype(std::declval<T>().score)>> : std::true_type
    {
    };

    template <class T, class = void>
    struct has_prop : std::false_type
    {
    };

    template <class T>
    struct has_prop<T, std::void_t<decltype(std::declval<T>().prop)>> : std::true_type
    {
    };

    template <class T, class = void>
    struct has_conf : std::false_type
    {
    };

    template <class T>
    struct has_conf<T, std::void_t<decltype(std::declval<T>().conf)>> : std::true_type
    {
    };

    template <class T, class = void>
    struct has_probability : std::false_type
    {
    };

    template <class T>
    struct has_probability<T, std::void_t<decltype(std::declval<T>().probability)>> : std::true_type
    {
    };

    // ============================================================
    // 置信度提取：你的 object_detect_result 只有 prop
    // ============================================================
    static inline double det_score(const object_detect_result &d)
    {
        return (double)d.prop;
    }

    // 主最用就是找出最优的目标进行椭圆拟合姿态解算
    static int pick_best_idx_by_class(const object_detect_result_list &result,
                                      const std::vector<EllipseFitResult> &ellipses,
                                      int cls_id)
    { // 找结果里面 类别为cls_id 分最高的那一个检测框
        int best = -1;
        double best_s = -1e18;
        for (int i = 0; i < result.count; ++i)
        {
            const auto &d = result.results_box[i];
            if (d.cls_id != cls_id)
                continue;
            if (i >= static_cast<int>(ellipses.size()) || !ellipses[i].valid)
                continue;
            double s = i < static_cast<int>(ellipses.size())
                           ? EllipseSelectionScore(ellipses[i], static_cast<float>(det_score(d)))
                           : det_score(d);
            if (best < 0 || s > best_s)
            {
                best = i;
                best_s = s;
            }
        }
        return best;
    }

    // 像素混合  alpha_beta为上色深度的占比
    static inline void blend_pixel(uchar *b, uchar *g, uchar *r, const int *color_weight, float alpha_beta)  //对一个像素的 BGR 三个通道做线性混合，常用于 mask 上色叠加 
    {
        //saturate_cast 是OpenCV 的“安全类型转换”： 把结果截断到0~255
        *b = cv::saturate_cast<uchar>(*b * alpha_beta + color_weight[0]);
        *g = cv::saturate_cast<uchar>(*g * alpha_beta + color_weight[1]);
        *r = cv::saturate_cast<uchar>(*r * alpha_beta + color_weight[2]);
    }


    //在图上画文字，带黑色描边，更清晰； scale是字体大小倍率  thickness是字体线条粗细
    static inline void draw_txt(cv::Mat &img, const std::string &text, int y,
                                const cv::Scalar &color, double scale = 1.2, int thickness = 2)
    {
        cv::putText(img, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), thickness + 3);
        cv::putText(img, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
    }

    // ============================================================
    // 绘制结果： det + mask 绘制保留
    // 在最后追加：cls0/1/2 择优 + PoseSolve + Dist
    // ============================================================
    void draw_results_on_frame(uint64_t frame_id,
                               cv::Mat &frame,   //要绘制的图像帧
                               const object_detect_result_list &result,
                               bool use_mask_fit_class0,
                               bool use_mask_fit_class1,
                               bool use_mask_fit_class2,   //对于最里面的小圆 是采用哪种方法拟合椭圆
                               bool force_box_for_reference_ring,
                               float deviation_threshold_ratio) //mask 拟合椭圆的合理性门槛(中心偏差比例),用来判断mask拟合的椭圆离检测框的中心是否太远
    //直接把检测框、mask 着色、椭圆、姿态轴、文字信息画到 frame 上;
    //如果缺少关键目标，会在画面上显示 "Pose: --" "Dist: --" 然后 return;
     {
        //颜色bgr
        const cv::Scalar C_GRAY(100, 100, 100);  //灰色
        const cv::Scalar C_CYAN(255, 255, 0);
        const cv::Scalar C_GRN(0, 255, 0);
        const cv::Scalar C_YEL(0, 255, 255);
        const cv::Scalar C_WHT(255, 255, 255);

        // ====== A)  det + mask + 单体椭圆绘制 ======
        float alpha = 0.5f;
        float beta = 1.0f - alpha;
        int color_weights[3][3];

        color_weights[0][0] = 0;
        color_weights[0][1] = 0;
        color_weights[0][2] = (int)(255 * alpha); // 红

        color_weights[1][0] = 0;
        color_weights[1][1] = (int)(255 * alpha);
        color_weights[1][2] = 0; // 绿

        color_weights[2][0] = (int)(255 * alpha);
        color_weights[2][1] = (int)(255 * alpha);
        color_weights[2][2] = 0; // 青

        const auto &seg_result = result.results_mask[0];
        const cv::Mat fitting_image = frame.clone();

        EllipseFitConfig ellipse_config;
        ellipse_config.center_deviation_ratio = deviation_threshold_ratio;
        const EllipseFitter ellipse_fitter(ellipse_config);

        //存储拟合的每个圆，共姿态结算使用，避免多次拟合
        std::vector<EllipseFitResult> ellipse_results;
        ellipse_results.reserve(result.count);

        for (int i = 0; i < result.count; i++)
        {
            const auto &det_box = result.results_box[i];

            int x = std::max(0, det_box.x);
            int y = std::max(0, det_box.y);
            int w = std::min((int)det_box.w, frame.cols - x);
            int h = std::min((int)det_box.h, frame.rows - y);
            if (w > 0 && h > 0)
            {
                cv::Rect box(x, y, w, h);
                cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);  //画检测框 2个像素宽
            }

            // mask 指针
            uint8_t *raw_mask_ptr = nullptr;
            uint8_t *mask_probability_ptr = nullptr;
            int class_id = det_box.cls_id;
            if (i < static_cast<int>(seg_result.each_of_mask.size()) &&
                seg_result.each_of_mask[i])
            {
                raw_mask_ptr = seg_result.each_of_mask[i].get();  //每个框和它的掩码一一对应
            }
            if (i < static_cast<int>(seg_result.each_of_mask_probability.size()) &&
                seg_result.each_of_mask_probability[i])
            {
                mask_probability_ptr = seg_result.each_of_mask_probability[i].get();
            }
            
            
            
            // 类别到拟合策略的映射（在线、批处理、单帧入口保持完全一致）：
            //   cls0 外圈、cls1 中圈：
            //     默认 PreferMaskNoEdge，即 Mask 拟合失败直接用框内切圆。
            //     不使用灰度 Edge，是为了避免验收现场的反光、阴影和背景纹理产生伪边缘。
            //   cls2 内孔：
            //     默认 ForceBox；只有显式打开 use_mask_fit_class2 才使用 PreferMask。
            //   force_box_for_reference_ring：
            //     验收保底总开关，优先级最高，可同时强制 cls0/cls1 使用内切圆。
            const bool force_box =
                (class_id == 0 && (force_box_for_reference_ring || !use_mask_fit_class0)) ||
                (class_id == 1 && (force_box_for_reference_ring || !use_mask_fit_class1)) ||
                (class_id == 2 && !use_mask_fit_class2);


                
            const cv::Rect detection_rect(det_box.x, det_box.y, det_box.w, det_box.h);
            const EllipseFitMode fit_mode = force_box
                                                ? EllipseFitMode::ForceBox
                                                : (class_id == 0 || class_id == 1
                                                       ? EllipseFitMode::PreferMaskNoEdge
                                                       : EllipseFitMode::PreferMask);
            EllipseFitResult ellipse_result = ellipse_fitter.Fit(
                fitting_image, detection_rect, raw_mask_ptr, fit_mode,
                mask_probability_ptr);
            // 橙色明确标记“部分可见但仍可解”；红色表示可见弧不足、仅保留诊断结果。
            const cv::Scalar e_color = !ellipse_result.valid
                                           ? cv::Scalar(0, 0, 255)
                                       : ellipse_result.partial_visibility
                                           ? cv::Scalar(0, 165, 255)
                                       : ellipse_result.source == EllipseSource::Mask
                                           ? cv::Scalar(0, 255, 0)
                                           : (ellipse_result.source == EllipseSource::Edge
                                                  ? cv::Scalar(255, 0, 255)
                                                  : cv::Scalar(0, 255, 255));
            cv::ellipse(frame, ellipse_result.ellipse, e_color, 2);
            cv::circle(frame, ellipse_result.ellipse.center, 2, cv::Scalar(0, 0, 255), -1);
            if (ellipse_result.partial_visibility)
            {
                for (size_t point_index = 0;
                     point_index < ellipse_result.visible_arc_points.size();
                     point_index += 4)
                    cv::circle(frame, ellipse_result.visible_arc_points[point_index],
                               1, cv::Scalar(0, 165, 255), -1);
            }

            ellipse_results.push_back(std::move(ellipse_result));

            // ROI mask 上色
            if (raw_mask_ptr && w > 0 && h > 0 && class_id >= 0 && class_id < 3)
            {
                cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, raw_mask_ptr);  //CV_8UC1单通道灰度图
                #pragma omp parallel for  //开启多线程并行加速 告诉编译器，把下面的for循环拆开分给cpu的多个核心同时执行
                for (int r = 0; r < h; r++)
                {
                    int abs_row = y + r;
                    uchar *ptr_img = frame.ptr<uchar>(abs_row);  //获取图像的 abs_row 行首地址
                    uchar *ptr_mask = full_mask.ptr<uchar>(abs_row); //行首地址
                    for (int c = 0; c < w; c++)
                    {
                        int abs_col = x + c;
                        if (ptr_mask[abs_col] > 0)
                        {
                            blend_pixel(&ptr_img[3 * abs_col], &ptr_img[3 * abs_col + 1], &ptr_img[3 * abs_col + 2],
                                        color_weights[class_id], beta);
                        }
                    }
                }
            }
        }

        // ====== B) cls0/1/2 择优 + 姿态解算 + 深度 ======
        const int CLS0_ID = 0; // 外圈
        const int CLS1_ID = 1; // 中圈
        const int CLS2_ID = 2; // 内孔

        // 联合检测置信度、拟合质量和双圆几何一致性选择最优目标组合。
        std::vector<int> class_ids(result.count);
        std::vector<float> detection_confidences(result.count);
        for (int i = 0; i < result.count; ++i)
        {
            class_ids[i] = result.results_box[i].cls_id;
            detection_confidences[i] = result.results_box[i].prop;
        }
        int idx2 = pick_best_idx_by_class(result, ellipse_results, CLS2_ID);
        int outer_count = 0, middle_count = 0;
        for (int i = 0; i < result.count; ++i)
        {
            if (!ellipse_results[i].valid) continue;
            outer_count += class_ids[i] == CLS0_ID;
            middle_count += class_ids[i] == CLS1_ID;
        }
        std::function<float(int, int)> pose_pair_score;
        if (idx2 >= 0 && outer_count * middle_count > 1 &&
            outer_count * middle_count <= 4)
        {
            pose_pair_score = [&](int outer_index, int middle_index)
            {
                const EllipseFitResult &outer_fit = ellipse_results[outer_index];
                const EllipseFitResult &middle_fit = ellipse_results[middle_index];
                const PoseEllipseObservation outer_obs{
                    outer_fit.ellipse, EllipseObservationSigmaPx(outer_fit), true,
                    outer_fit.visible_arc_points, outer_fit.partial_visibility,
                    outer_fit.visible_arc_ratio};
                const PoseEllipseObservation middle_obs{
                    middle_fit.ellipse, EllipseObservationSigmaPx(middle_fit), true,
                    middle_fit.visible_arc_points, middle_fit.partial_visibility,
                    middle_fit.visible_arc_ratio};
                const float reprojection = static_cast<float>(
                    _pose_estimator.EvaluateDualReprojectionScore(
                    outer_obs, middle_obs, ellipse_results[idx2].ellipse.center,
                    EllipseObservationSigmaPx(ellipse_results[idx2])));
                return reprojection * std::sqrt(
                    std::max(0.0f, outer_fit.quality * middle_fit.quality));
            };
        }
        const RingPairSelection ring_pair = _ring_pair_refiner.SelectAndRefine(
            class_ids, detection_confidences, ellipse_results, CLS0_ID, CLS1_ID,
            pose_pair_score);
        int idx0 = ring_pair.outer_index;
        int idx1 = ring_pair.middle_index;

        // Python 同逻辑：必须有孔 + (外圈或中圈)  才能进行姿态解算
        if (idx2 < 0 || (idx0 < 0 && idx1 < 0))
        {
            draw_txt(frame, "Pose: --", 190, C_CYAN, 1.2);
            draw_txt(frame, "Dist: --", 240, C_CYAN, 1.2);
            return;
        }
        // 实时业务线程可能乱序：只允许更新 frame_id 更新的帧进入时序滤波状态。
        if (_enable_temporal_filter)
        {
            std::lock_guard<std::mutex> lock(_ellipse_filter_mutex);
            if (!_has_filtered_frame || frame_id > _last_filtered_frame_id)
            {
                if (idx0 >= 0) ellipse_results[idx0] = _ellipse_temporal_filter.Update(0, ellipse_results[idx0]);
                if (idx1 >= 0) ellipse_results[idx1] = _ellipse_temporal_filter.Update(1, ellipse_results[idx1]);
                ellipse_results[idx2] = _ellipse_temporal_filter.Update(2, ellipse_results[idx2]);
                _last_filtered_frame_id = frame_id;
                _has_filtered_frame = true;
            }
        }

        // cand0 / cand1
        bool has0 = (idx0 >= 0);
        bool has1 = (idx1 >= 0);

        EllipseFitResult cand0{}, cand1{};
        if (has0)
        {
            cand0 = ellipse_results[idx0];
        }
        if (has1)
        {
            cand1 = ellipse_results[idx1];
        }

        const EllipseFitResult &hole_ellipse = ellipse_results[idx2];

        cv::Point2f hole_center = hole_ellipse.ellipse.center;
       

        // 在已通过联合约束的外/中圈中按综合质量选择位姿参考。
        cv::RotatedRect target;
        bool use_cls1 = false;
        bool ok_target = false;

        if (has0 && has1)
        {
            const float outer_score = EllipseSelectionScore(cand0, result.results_box[idx0].prop);
            const float middle_score = EllipseSelectionScore(cand1, result.results_box[idx1].prop);
            if (outer_score >= middle_score)
            {
                target = cand0.ellipse;
                use_cls1 = false;
                ok_target = true;
            }
            else
            {
                target = cand1.ellipse;
                use_cls1 = true;
                ok_target = true;
            }
        }
        else if (has0)
        {
            target = cand0.ellipse;
            use_cls1 = false;
            ok_target = true;
        }
        else if (has1)
        {
            target = cand1.ellipse;
            use_cls1 = true;
            ok_target = true;
        }

        if (!ok_target)
            return;


        // 正主高亮  C_CYAN青色
        cv::ellipse(frame, target, C_CYAN, 4);
        cv::ellipse(frame, hole_ellipse.ellipse, C_CYAN, 4);
        cv::circle(frame, hole_center, 6, C_CYAN, -1);
        cv::line(frame, target.center, hole_center, C_CYAN, 3);  //绘制姿态轴线，画一条直线连接 两个椭圆的中心


        std::optional<PoseEllipseObservation> outer_observation;
        std::optional<PoseEllipseObservation> middle_observation;
        if (has0)
            outer_observation = PoseEllipseObservation{cand0.ellipse,
                                                       EllipseObservationSigmaPx(cand0), true,
                                                       cand0.visible_arc_points,
                                                       cand0.partial_visibility,
                                                       cand0.visible_arc_ratio};
        if (has1)
            middle_observation = PoseEllipseObservation{cand1.ellipse,
                                                        EllipseObservationSigmaPx(cand1), true,
                                                        cand1.visible_arc_points,
                                                        cand1.partial_visibility,
                                                        cand1.visible_arc_ratio};
        const double hole_sigma = EllipseObservationSigmaPx(hole_ellipse);

        // 在线画面每帧只消费一种位姿模式。以前无条件同时求 auto/fixed，相当于
        // 完整执行两次 LM；改为只计算当前显示模式，直接节省约一半位姿耗时。
        const Pose6D pose_final = _display_fixed_mode
            ? _pose_estimator.SolveDual(
                  outer_observation, middle_observation, hole_center,
                  hole_sigma, _fixed_distance_mm)
            : _pose_estimator.SolveDual(
                  outer_observation, middle_observation, hole_center,
                  hole_sigma, std::nullopt);

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

            if (_display_fixed_mode)
            {
                std::snprintf(buf, sizeof(buf), "Yaw:%.1f Pit:%.1f", pose_final.yaw_deg, pose_final.pitch_deg);
                draw_txt(frame, buf, 190, C_YEL, 1.3);
                std::snprintf(buf, sizeof(buf), "Dist: %.2fm (Fixed)", pose_final.tz_mm / 1000.0);
                draw_txt(frame, buf, 240, C_YEL, 1.3);
            }
            else
            {
                std::snprintf(buf, sizeof(buf), "Yaw:%.1f Pit:%.1f", pose_final.yaw_deg, pose_final.pitch_deg);
                draw_txt(frame, buf, 190, C_GRN, 1.3);
                std::snprintf(buf, sizeof(buf), "Dist: %.2fm (Auto)", pose_final.tz_mm / 1000.0);
                draw_txt(frame, buf, 240, C_GRN, 1.3);
            }
        }
    }

    // ============================================================
    // NPU 工作线程
    // ============================================================
    void worker_loop(int model_id)
    {
        while (true)
        {
            auto jobOpt = _in_queue.pop();
            if (!jobOpt)
                break;

            auto job = std::move(*jobOpt);

            //  1) 提前丢帧（省 NPU/CPU） +  仍然 push token，避免外部 inflight 卡死
            if (_min_expect_id_ptr)
            {
                uint64_t min_needed = _min_expect_id_ptr->load(std::memory_order_relaxed);
                if (job.frame_id < min_needed)
                {
                    // 关键：push 一个“空结果”，让外部 consumer 能 pop 到并做 inflight--
                    _out_queue.push(InferOut{job.frame_id, nullptr, object_detect_result_list{}});
                    continue;
                }
            }

            auto proc = std::move(job.proc);

            // NPU 前处理 & 推理（这里做一次就够了）
            proc->image_preprocessing(640, 640);
            int image_len = 0;
            uint8_t *buffer = proc->get_image_buffer(&image_len);

            if (_models[model_id]->set_input_data(buffer, image_len) != RKNN_SUCC)
                break;
            if (_models[model_id]->rknn_model_inference() != RKNN_SUCC)
                break;
            if (_models[model_id]->get_output_data() != RKNN_SUCC)
                break;

            // 后处理
            object_detect_result_list result;
            letterbox letter_box = proc->get_letterbox();
            if (_models[model_id]->post_process(result, letter_box) != RKNN_SUCC)
                break;

            std::shared_ptr<image_process> shared_proc = std::move(proc);

            bussiness_pool->enqueue(
                [this, fid = job.frame_id, p = shared_proc, r = std::move(result)]() mutable
                {
                    cv::Mat &frame_to_draw = *(p->_src_image_frame);

                    this->draw_results_on_frame(fid, frame_to_draw, r,
                                                this->_enable_mask_fit_for_class0,
                                                this->_enable_mask_fit_for_class1,
                                                this->_enable_mask_fit_for_class2,
                                                this->_force_box_for_reference_ring,
                                                this->_max_deviation_ratio);

                    if (this->_biz_callback)
                        this->_biz_callback(fid, p, r);

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

    // ===== 控制参数 =====
    bool _enable_mask_fit_for_class0 = true;
    bool _enable_mask_fit_for_class1 = true;
    bool _enable_mask_fit_for_class2 = false;
    bool _force_box_for_reference_ring = false;
    float _max_deviation_ratio = 0.3f;
    bool _enable_temporal_filter = true;
    RingPairRefiner _ring_pair_refiner;
    EllipseTemporalFilter _ellipse_temporal_filter;
    std::mutex _ellipse_filter_mutex;
    uint64_t _last_filtered_frame_id = 0;
    bool _has_filtered_frame = false;

    // ===== 新增：Pose/Depth 控制参数 =====
    bool _display_fixed_mode = false;
    double _fixed_distance_mm = 3000.0;

    // ===== 新增：PoseEstimator =====
    cv::Mat _K, _D; // 相机内参 和 畸变系数
    DrogueModel _drogue;
    PoseEstimatorLM _pose_estimator;
};
