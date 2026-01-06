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

#include "pose_estimator_lm.hpp"

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

// 椭圆拟合结果结构体
struct EllipseResult
{
    cv::RotatedRect rect; // 表示拟合出来的椭圆（中心、长短轴、角度）
    bool is_from_mask;    // true=掩码拟合, false=Box保底
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

    // 类别2是否优先使用掩码拟合 (true=尝试拟合, false=强制Box/内切圆)
    void set_class2_mask_fit_mode(bool enable)
    {
        _enable_mask_fit_for_class2 = enable;
    }

    // 允许的圆心最大偏差（相对短边比例），默认 0.3
    void set_deviation_threshold(float ratio)
    {
        _max_deviation_ratio = ratio;
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
    static int pick_best_idx_by_class(const object_detect_result_list &result, int cls_id)
    { // 找结果里面 类别为cls_id 分最高的那一个检测框
        int best = -1;
        double best_s = -1e18;
        for (int i = 0; i < result.count; ++i)
        {
            const auto &d = result.results_box[i];
            if (d.cls_id != cls_id)
                continue;
            double s = det_score(d);
            if (best < 0 || s > best_s)
            {
                best = i;
                best_s = s;
            }
        }
        return best;
    }

    // ============================================================
    // 椭圆拟合（保留 deviation 校验
    //  Box 保底改成 “内切圆/正方形” 更贴近 Python 逻辑
    // ============================================================

    ///////////////////////////////////////////////////////////////////////轻量RANSAC椭圆拟合工具
    // 点到椭圆边界的“近似像素误差”
    // 思路：把点变换到椭圆坐标系，r=sqrt((x/a)^2+(y/b)^2)，理想边界 r=1
    // err_px ≈ |r-1| * min(a,b)
    static inline float ellipse_radial_error_px(const cv::RotatedRect &e, const cv::Point2f &p)
    {
        float a = e.size.width * 0.5f;
        float b = e.size.height * 0.5f;
        if (a < 1e-3f || b < 1e-3f)
            return 1e9f;

        float x = p.x - e.center.x;
        float y = p.y - e.center.y;

        float ang = -e.angle * (float)CV_PI / 180.0f;
        float c = std::cos(ang), s = std::sin(ang);
        float xr = c * x - s * y;
        float yr = s * x + c * y;

        float r = std::sqrt((xr * xr) / (a * a) + (yr * yr) / (b * b));
        float err = std::fabs(r - 1.0f) * std::min(a, b);
        return err;
    }

    struct RansacEllipseFit
    {
        bool ok = false;
        cv::RotatedRect ellipse;
        int inliers = 0;
        float mean_err = 1e9f;
    };

    // 轻量 RANSAC：
    // - points：建议 <=150（实时要求）
    // - iters：80~150
    // - inlier_th_px：2~4 像素
    // - min_inlier_ratio：0.35~0.6（越大越严格）
    static inline RansacEllipseFit fit_ellipse_ransac(
        const std::vector<cv::Point> &points,
        int iters = 120,
        float inlier_th_px = 3.0f,
        float min_inlier_ratio = 0.45f,
        float max_axis_ratio = 6.0f // 轴比上限，防止退化
    )
    {
        RansacEllipseFit best;
        const int N = (int)points.size();
        if (N < 20)
            return best; // 太少点别RANSAC，直接普通拟合更合适

        std::mt19937 rng(12345); // 固定种子：可复现；想随机可用 std::random_device
        std::uniform_int_distribution<int> uni(0, N - 1);

        std::vector<cv::Point> sample(5);
        std::vector<int> idx(5);

        for (int t = 0; t < iters; ++t)
        {
            // 随机取5个不同点
            for (int k = 0; k < 5;)
            {
                int r = uni(rng);
                bool dup = false;
                for (int j = 0; j < k; ++j)
                    if (idx[j] == r)
                    {
                        dup = true;
                        break;
                    }
                if (dup)
                    continue;
                idx[k] = r;
                sample[k] = points[r];
                ++k;
            }

            cv::RotatedRect e;
            try
            {
                e = cv::fitEllipse(sample);
            }
            catch (...)
            {
                continue;
            }

            float a = e.size.width * 0.5f, b = e.size.height * 0.5f;
            if (a < 2.f || b < 2.f)
                continue;

            float axis_ratio = std::max(a, b) / std::max(1e-3f, std::min(a, b));
            if (axis_ratio > max_axis_ratio)
                continue; // 过扁过长通常是坏解

            // 统计inliers
            int inl = 0;
            float err_sum = 0.f;
            for (int i = 0; i < N; ++i)
            {
                float err = ellipse_radial_error_px(e, (cv::Point2f)points[i]);
                if (err <= inlier_th_px)
                {
                    ++inl;
                    err_sum += err;
                }
            }

            // 更新best（优先inliers，其次平均误差）
            if (inl > best.inliers || (inl == best.inliers && inl > 0 && err_sum / inl < best.mean_err))
            {
                best.ok = true;
                best.ellipse = e;
                best.inliers = inl;
                best.mean_err = (inl > 0) ? (err_sum / inl) : 1e9f;
            }
        }

        if (!best.ok)
            return best;

        // inlier比例门槛
        const int min_inliers = (int)std::ceil(min_inlier_ratio * N);
        if (best.inliers < std::max(20, min_inliers)) // 至少20个inlier
        {
            best.ok = false;
            return best;
        }

        // 用inliers再拟合一次（关键：稳定）
        std::vector<cv::Point> inlier_pts;
        inlier_pts.reserve(best.inliers);
        for (int i = 0; i < N; ++i)
        {
            float err = ellipse_radial_error_px(best.ellipse, (cv::Point2f)points[i]);
            if (err <= inlier_th_px)
                inlier_pts.push_back(points[i]);
        }

        if ((int)inlier_pts.size() >= 5)
        {
            try
            {
                best.ellipse = cv::fitEllipse(inlier_pts);
                best.ok = true;
                best.inliers = (int)inlier_pts.size();
            }
            catch (...)
            {
                best.ok = false;
            }
        }
        else
        {
            best.ok = false;
        }
        return best;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////工具代码结尾

    static EllipseResult calculate_best_ellipse(const cv::Mat &frame,                   // 原图像帧
                                                const object_detect_result &det_result, // 检测框
                                                uint8_t *mask_data,                     // 掩码数据
                                                bool force_box_mode,                    // 标志位，是否强制走检测框的内切圆
                                                float deviation_threshold_ratio)        // 偏差阈值比例：用于判断 mask 拟合的椭圆中心是否离 box 中心太远
    {
        EllipseResult result{};
        bool fit_success = false;

        cv::Point2f box_center(det_result.x + det_result.w / 2.0f, // 检测框中心点
                               det_result.y + det_result.h / 2.0f);

        // --- 策略 1: 尝试 Mask 拟合 ---
        if (mask_data && !force_box_mode)
        {
            int x = std::max(0, det_result.x);
            int y = std::max(0, det_result.y);
            int w = std::min((int)det_result.w, frame.cols - x);
            int h = std::min((int)det_result.h, frame.rows - y);

            if (w > 0 && h > 0)
            {
                cv::Rect roi_rect(x, y, w, h);                                 // 一个矩形区域
                cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, mask_data); // 用mask区域包装了一个Mat
                cv::Mat roi_mask = full_mask(roi_rect);                        // 仍然是视图，没有拷贝，意思就是 用一个roi_mask 表示了 full_mask中roi_rect 的这个区域   （引用同一块内存的一部分）
                cv::Mat contour_input = roi_mask.clone();                      // 深拷贝了一份

                // 找轮廓   有多少个"白色孤岛"就扫描多少个轮廓
                /*
                段代码的核心函数 cv::findContours 基于著名的 Suzuki85 算法（Suzuki & Abe, 1985）。 findContours 的工作原理就像是一个机器人在遍历图像：
                1.它逐行扫描图像，直到遇到一个非零像素（通常是白色，代表物体）。
                2.一旦遇到，它就沿着这块白色区域的边缘行走，直到回到原点。
                3.它把沿途经过的坐标点记录下来，这就形成了一个轮廓。
                */
                std::vector<std::vector<cv::Point>> contours;                                          // 轮廓点集
                cv::findContours(contour_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // RETR_EXTERNAL：只找外轮廓（忽略洞/内轮廓）  CHAIN_APPROX_SIMPLE：压缩轮廓点（减少点数）

                if (!contours.empty())
                {
                    // ------- 更鲁棒：合并多个轮廓 + 轻量RANSAC拟合椭圆 -------
                    std::vector<cv::Point> all_pts;
                    all_pts.reserve(1000);

                    // ROI坐标系下 box_center（因为 contour 点在 ROI 内）   坐标变换一下
                    const float cx_roi = box_center.x - (float)x;
                    const float cy_roi = box_center.y - (float)y;

                    // 过滤阈值（代价很低，但鲁棒性提升巨大）
                    const double roi_area = (double)w * (double)h;
                    const double area_min = 0.005 * roi_area;                      // ROI面积的0.5% 过滤小噪声
                    const float dist_limit = 0.80f * std::min((float)w, (float)h); // 轮廓重心离box中心太远就不要

                    for (auto &c : contours)
                    {
                        if (c.size() < 5)
                            continue;
                        double area = std::fabs(cv::contourArea(c));  //格林公式计算面积
                        if (area < area_min)
                            continue;

                        cv::Moments mu = cv::moments(c);  //计算轮廓的图像距
                        if (std::fabs(mu.m00) < 1e-6)   //m00代表0阶距，对于二值图像或轮廓来说，就是轮廓面积
                            continue;
                        float ccx = (float)(mu.m10 / mu.m00);
                        float ccy = (float)(mu.m01 / mu.m00);  //计算重心位置        重心=力矩/总质量
                        float dist = std::hypot(ccx - cx_roi, ccy - cy_roi);    //计算欧氏距离
                        if (dist > dist_limit)
                            continue;

                        all_pts.insert(all_pts.end(), c.begin(), c.end());
                    }

                    // 点太少就算失败（走box保底）
                    if ((int)all_pts.size() >= 20)
                    {
                        // downsample 到 <=150，保证实时性    下采样设置
                        const int MAX_PTS = 150;
                        std::vector<cv::Point> ds;
                        ds.reserve(std::min((int)all_pts.size(), MAX_PTS));

                        int step = std::max(1, (int)all_pts.size() / MAX_PTS);  //计算步长   就是每隔step间距取一个点
                        for (int i = 0; i < (int)all_pts.size(); i += step)
                            ds.push_back(all_pts[i]);


                        // 轻量RANSAC参数：可以按机器性能调
                        // iters 80~150；inlier阈值 2~4px
                        auto rf = fit_ellipse_ransac(ds,
                                                     120,   // iters   迭代次数
                                                     3.0f,  // inlier_th_px    阈值。点距离椭圆边 3 像素以内算“自己人”。
                                                     0.45f, // min_inlier_ratio   比例。至少 45% 的点要在椭圆上，才算拟合成功
                                                     6.0f); // max_axis_ratio     形状限制。长轴/短轴不能超过6（防止拟合成一根长面条）。

                        if (rf.ok)  //如果拟合出椭圆了，进行坐标还原
                        {
                            cv::RotatedRect local_ellipse = rf.ellipse;

                            result.rect = local_ellipse;
                            // ROI坐标 -> 全图坐标
                            result.rect.center.x += (float)x;
                            result.rect.center.y += (float)y;

                            // 偏差校验：mask椭圆中心不能离box_center太远
                            float distc = std::hypot(result.rect.center.x - box_center.x,
                                                     result.rect.center.y - box_center.y);

                            float limit_dist = std::min((float)det_result.w, (float)det_result.h) * deviation_threshold_ratio;

                            if (distc <= limit_dist)
                            {
                                result.is_from_mask = true;
                                fit_success = true;
                            }
                        }
                    }
                }
            }
        }

        // --- 策略 2: Box 保底（内切圆/正方形）---
        if (!fit_success)
        {
            float min_s = std::min((float)det_result.w, (float)det_result.h);
            cv::Size2f size(min_s, min_s);
            result.rect = cv::RotatedRect(box_center, size, 0.0f);
            result.is_from_mask = false;
        }


        //这里Cany方法单独罗列出来  如有需要自行添加, 对于我们本项目里面的处理逻辑，如果yolo都检测失败了，那也就没必要去拟合椭圆了，
        //强行Cany拟合效果肯定很差，不如保持上一帧结果输出就行.


        return result;
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
    void draw_results_on_frame(cv::Mat &frame,   //要绘制的图像帧
                               const object_detect_result_list &result,
                               bool use_mask_fit_class2,   //对于最里面的小圆 是采用哪种方法拟合椭圆
                               float deviation_threshold_ratio) //mask 拟合椭圆的合理性门槛(中心偏差比例),用来判断mask拟合的椭圆离检测框的中心是否太远
    //直接把检测框、mask 着色、椭圆、姿态轴、文字信息画到 frame 上;
    //如果缺少关键目标，会在画面上显示 "Pose: --" "Dist: --" 然后 return;
     {
        //颜色bgr
        const cv::Scalar C_GRAY(100, 100, 100);
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

        for (int i = 0; i < result.count; i++)
        {
            const auto &det_box = result.results_box[i];

            int x = std::max(0, det_box.x);
            int y = std::max(0, det_box.y);
            int w = std::min((int)det_box.w, frame.cols - x);
            int h = std::min((int)det_box.h, frame.rows - y);
            cv::Rect box(x, y, w, h);
            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);  //画检测框 2个像素宽

            // mask 指针
            uint8_t *raw_mask_ptr = nullptr;
            int class_id = det_box.cls_id;
            raw_mask_ptr = seg_result.each_of_mask[i].get();  //每个框和他的掩码是一一对应的
            
            
            
            // 类别2是否强制 box
            bool force_box = false;
            if (class_id == 2 && !use_mask_fit_class2)
                force_box = true;


                
            EllipseResult ellipse_res = calculate_best_ellipse(frame, det_box, raw_mask_ptr, force_box, deviation_threshold_ratio);

            cv::Scalar e_color = ellipse_res.is_from_mask ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
            cv::ellipse(frame, ellipse_res.rect, e_color, 2);
            cv::circle(frame, ellipse_res.rect.center, 2, cv::Scalar(0, 0, 255), -1);

            // ROI mask 上色
            if (raw_mask_ptr && w > 0 && h > 0 && mask_idx >= 0)
            {
                cv::Mat full_mask(frame.rows, frame.cols, CV_8UC1, raw_mask_ptr);
                #pragma omp parallel for
                for (int r = 0; r < h; r++)
                {
                    int abs_row = y + r;
                    uchar *ptr_img = frame.ptr<uchar>(abs_row);
                    uchar *ptr_mask = full_mask.ptr<uchar>(abs_row);
                    for (int c = 0; c < w; c++)
                    {
                        int abs_col = x + c;
                        if (ptr_mask[abs_col] > 0)
                        {
                            blend_pixel(&ptr_img[3 * abs_col], &ptr_img[3 * abs_col + 1], &ptr_img[3 * abs_col + 2],
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
        if (idx2 < 0 || (idx0 < 0 && idx1 < 0))
        {
            draw_txt(frame, "Pose: --", 190, C_CYAN, 1.2);
            draw_txt(frame, "Dist: --", 240, C_CYAN, 1.2);
            return;
        }

        auto mask_ptr_of = [&](int cls_id) -> uint8_t *
        {
            if (cls_id < 0 || cls_id >= (int)seg_result.each_of_mask.size())
                return nullptr;
            if (!seg_result.each_of_mask[cls_id])
                return nullptr;
            return seg_result.each_of_mask[cls_id].get();
        };

        // cand0 / cand1
        bool has0 = (idx0 >= 0);
        bool has1 = (idx1 >= 0);

        EllipseResult cand0{}, cand1{};
        if (has0)
            cand0 = calculate_best_ellipse(frame, result.results_box[idx0], mask_ptr_of(CLS0_ID), false, deviation_threshold_ratio);
        if (has1)
            cand1 = calculate_best_ellipse(frame, result.results_box[idx1], mask_ptr_of(CLS1_ID), false, deviation_threshold_ratio);

        // hole（cls2）：可选用 mask 拟合；如果开关关了就强制 box（和你现有代码一致）
        bool force_box2 = !use_mask_fit_class2;
        EllipseResult hole_e = calculate_best_ellipse(frame, result.results_box[idx2], mask_ptr_of(CLS2_ID), force_box2, deviation_threshold_ratio);
        cv::Point2f hole_center = hole_e.rect.center;
        float hole_radius = 0.5f * std::min(hole_e.rect.size.width, hole_e.rect.size.height);

        // 择优策略：优先 cand0=Mask，否则 cand1=Mask，否则 cand0（按你 Python）
        cv::RotatedRect target;
        bool use_cls1 = false;
        bool ok_target = false;

        if (has0 && has1)
        {
            if (cand0.is_from_mask)
            {
                target = cand0.rect;
                use_cls1 = false;
                ok_target = true;
            }
            else if (cand1.is_from_mask)
            {
                target = cand1.rect;
                use_cls1 = true;
                ok_target = true;
            }
            else
            {
                target = cand0.rect;
                use_cls1 = false;
                ok_target = true;
            }
        }
        else if (has0)
        {
            target = cand0.rect;
            use_cls1 = false;
            ok_target = true;
        }
        else if (has1)
        {
            target = cand1.rect;
            use_cls1 = true;
            ok_target = true;
        }

        if (!ok_target)
            return;

        // 备选灰色
        if (has0)
            cv::ellipse(frame, cand0.rect, C_GRAY, 1);
        if (has1)
            cv::ellipse(frame, cand1.rect, C_GRAY, 1);
        if (hole_radius > 1)
            cv::circle(frame, hole_center, (int)hole_radius, C_GRAY, 1);

        // 正主高亮
        cv::ellipse(frame, target, C_CYAN, 4);
        cv::circle(frame, hole_center, 6, C_CYAN, -1);
        if (hole_radius > 1)
            cv::circle(frame, hole_center, (int)hole_radius, C_CYAN, 3);
        cv::line(frame, target.center, hole_center, C_CYAN, 3);

        // 双轨解算
        Pose6D pose_auto = _pose_estimator.Solve(target, hole_center, use_cls1, std::nullopt);
        Pose6D pose_fix = _pose_estimator.Solve(target, hole_center, use_cls1, _fixed_distance_mm);
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

            if (_display_fixed_mode)
            {
                std::snprintf(buf, sizeof(buf), "Yaw:%.1f Pit:%.1f", pose_fix.yaw_deg, pose_fix.pitch_deg);
                draw_txt(frame, buf, 190, C_YEL, 1.3);
                std::snprintf(buf, sizeof(buf), "Dist: %.2fm (Fixed)", pose_fix.tz_mm / 1000.0);
                draw_txt(frame, buf, 240, C_YEL, 1.3);
            }
            else
            {
                std::snprintf(buf, sizeof(buf), "Yaw:%.1f Pit:%.1f", pose_auto.yaw_deg, pose_auto.pitch_deg);
                draw_txt(frame, buf, 190, C_GRN, 1.3);
                std::snprintf(buf, sizeof(buf), "Dist: %.2fm (Auto)", pose_auto.tz_mm / 1000.0);
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

                    this->draw_results_on_frame(frame_to_draw, r,
                                                this->_enable_mask_fit_for_class2,
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
    bool _enable_mask_fit_for_class2 = true;
    float _max_deviation_ratio = 0.3f;

    // ===== 新增：Pose/Depth 控制参数 =====
    bool _display_fixed_mode = false;
    double _fixed_distance_mm = 3000.0;

    // ===== 新增：PoseEstimator =====
    cv::Mat _K, _D; // 相机内参 和 畸变系数
    DrogueModel _drogue;
    PoseEstimatorLM _pose_estimator;
};
