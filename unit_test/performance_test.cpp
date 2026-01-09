#include "yolov8seg.hpp"
#include "image_process.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <easy_timer.h>
#include "visual.hpp"
#include "pose_estimator_lm.hpp"
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <type_traits>
#include <utility>
#include <random>
#include <fstream>
using namespace std;

TIMER ttt;
TIMER zzz;

// 椭圆拟合结果结构体
struct EllipseResult
{
    cv::RotatedRect rect; // 表示拟合出来的椭圆（中心、长短轴、角度）
    bool is_from_mask;    // true=掩码拟合, false=Box保底
};


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




int main(int argc, char *argv[])
{
if(argc<3)
{
       cout << "Usage:  " << argv[0] <<"     " <<"model.rknn path"<<"     "<< "video path" <<endl;
        return -1;
}

 // 打开视频文件
    cv::VideoCapture cap(argv[2]);
  if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open video file." << std::endl;
        return -1;
    }

    yolov8seg yolo(argv[1]);
    yolo.init();
  DrogueModel drogue;
PoseEstimatorLM pose_estimator;

       cv::Mat K = (cv::Mat_<double>(3, 3) << 1639.6, 0, 960, 0, 2165.4, 540, 0, 0, 1); // cv::Mat_<double>(3,3)  创建一个3x3的矩阵 。  <<：后面数字依次填入矩阵，行优先
       cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);                                        // 创建一个4行1列的全零矩阵，64位浮点数

        // 物理模型：k=20/45.5；radius_cls0=1200*k, radius_cls1=980*k, radius_hole=120*k, length_L=920*k
        {
            const double k = 20.0 / 45.5; // 缩放倍数，用于调节真实大小
            drogue.radius_cls0_mm = 1200.0 * k;
            drogue.radius_cls1_mm = 980.0 * k;
            drogue.radius_hole_mm = 120.0 * k;
            drogue.length_L_mm = 920.0 * k;
        }
        pose_estimator.Reset(K, D, drogue);


        vector<double> pre_processing;
        vector<double> rknn_infer;
        vector<double> post_processing;

        vector<double> yuan;
        vector<double> zitai;

        vector<double> full_frame;

        pre_processing.reserve(3000);
        rknn_infer.reserve(3000);
        post_processing.reserve(3000);
        yuan.reserve(3000);
        zitai.reserve(3000);
        full_frame.reserve(3000);



    cv::Mat frame;
    while (cap.read(frame))
    {
        zzz.tik();
        int err = 0;
        string path("day.jpg");
        cv::Mat frame11=frame.clone();
        if (frame11.empty())
        {
            cout << "imread error" << endl;
            return -1;
        }

        ttt.tik();
        image_process image(frame11);
        image.image_preprocessing(640, 640);
        ttt.tok();
        ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_图像预处理用时:");
        pre_processing.push_back(ttt.get_time());


        int image_len;
        void *image_data = image.get_image_buffer(&image_len);

        ttt.tik();
        yolo.set_input_data(image_data, image_len);
        ttt.tok();
        ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_yolo设置输入用时:");

        ttt.tik();
        err = yolo.rknn_model_inference();
        if (err)
        {
            cout << "rknn_model_inference error" << endl;
            return -1;
        }
        ttt.tok();
        ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_模型rknn推理用时");
        rknn_infer.push_back(ttt.get_time());


        err = yolo.get_output_data();
        if (err)
        {
            cout << "get_output_data error" << endl;
            return -1;
        }

        ttt.tik();
        object_detect_result_list result;
        letterbox letter_box = image.get_letterbox();

        yolo.post_process(result, letter_box);
        ttt.tok();
        ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_模型输出后处理用时");
        post_processing.push_back(ttt.get_time());




        ttt.tik();
        // 存储拟合的每个圆，共姿态结算使用，避免多次拟合
        std::vector<EllipseResult> store_EllipseResult;
        store_EllipseResult.reserve(result.count);
        const auto &seg_result = result.results_mask[0];
        for (int i = 0; i < result.count; i++)
        {
            const auto &det_box = result.results_box[i];

            int x = std::max(0, det_box.x);
            int y = std::max(0, det_box.y);
            int w = std::min((int)det_box.w, frame.cols - x);
            int h = std::min((int)det_box.h, frame.rows - y);
        
            // mask 指针
            uint8_t *raw_mask_ptr = nullptr;
            int class_id = det_box.cls_id;
            raw_mask_ptr = seg_result.each_of_mask[i].get(); // 每个框和他的掩码是一一对应的

            // 类别2是否强制 box
            bool force_box = false;
                force_box = true;

            EllipseResult ellipse_res = calculate_best_ellipse(frame, det_box, raw_mask_ptr, force_box, 0.3f);
           
            store_EllipseResult.push_back(std::move(ellipse_res));

        }
        ttt.tok();
        ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_拟合椭圆用时:");
         yuan.push_back(ttt.get_time());
        



         ttt.tik();
        // ====== B) cls0/1/2 择优 + 姿态解算 + 深度 ======
        const int CLS0_ID = 0; // 外圈
        const int CLS1_ID = 1; // 中圈
        const int CLS2_ID = 2; // 内孔

        //找预测结果里面最优的目标（置信度最高）
        int idx0 = pick_best_idx_by_class(result, CLS0_ID);
        int idx1 = pick_best_idx_by_class(result, CLS1_ID);
        int idx2 = pick_best_idx_by_class(result, CLS2_ID);

        // Python 同逻辑：必须有孔 + (外圈或中圈)  才能进行姿态解算
        if (idx2 < 0 || (idx0 < 0 && idx1 < 0))
        {
            break;
        }



        //定义在函数内部的“临时小函数”，Lambda表达式
        //[&] : 按引用捕获外部作用域的所有变量
        //(int cls_id) : 参数列表
        //->uint8_t *  : 尾置返回类型
        auto mask_ptr_of = [&](int index) -> uint8_t *
        {
            if (index < 0 || index >= (int)seg_result.each_of_mask.size())
                return nullptr;
            if (!seg_result.each_of_mask[index])
                return nullptr;
            return seg_result.each_of_mask[index].get();
        };
        ///////////Lambda表达式结束



        // cand0 / cand1
        bool has0 = (idx0 >= 0);
        bool has1 = (idx1 >= 0);

        EllipseResult cand0{}, cand1{};
        if (has0)
        {
            // cand0 = calculate_best_ellipse(frame, result.results_box[idx0], mask_ptr_of(idx0), false, deviation_threshold_ratio);
            cand0 = store_EllipseResult[idx0];
        }
        if (has1)
        {
            // cand1 = calculate_best_ellipse(frame, result.results_box[idx1], mask_ptr_of(idx1), false, deviation_threshold_ratio);
            cand1 = store_EllipseResult[idx1];
        }

        //// hole（cls2）：可选用 mask 拟合；如果开关关了就强制 box
        // bool force_box2 = !use_mask_fit_class2;
        // EllipseResult hole_e = calculate_best_ellipse(frame, result.results_box[idx2], mask_ptr_of(idx2), force_box2, deviation_threshold_ratio);
          EllipseResult& hole_e=store_EllipseResult[idx2];

        cv::Point2f hole_center = hole_e.rect.center;
       

        // 择优策略：优先 cand0=Mask，否则 cand1=Mask，否则 cand0
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
            return 0;

        ttt.tok();
        ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_杂七杂八:");
       

         ttt.tik();
        // 双轨解算
        Pose6D pose_auto = pose_estimator.Solve(target, hole_center, use_cls1, std::nullopt);   //自己解算深度信息
    
        ttt.tok();
        ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_姿态解算用时:");
         zitai.push_back(ttt.get_time());
     

        yolo.release_output_data();

        zzz.tok();
        zzz.print_time_app("xxxxxxxxxxxxxxxxxxxx_整个处理一帧用时（包含上述打印用时，以及读视频，清理资源任务）:");
         full_frame.push_back(zzz.get_time());

        cout<<endl;
        cout<<endl;
        cout<<endl;

      
    };

        cout<<endl;
        cout<<endl;
        cout<<endl;
        cout<<"一共处理"<<pre_processing.size()<<"帧图像,平均用时统计如下："<<endl;
double tem=0;
int n=pre_processing.size();
double max=0;
double min=10000;
    for(const auto& x: pre_processing)
     {
        if(x>max) max=x;
        if(x<min) min=x;
        tem+=x;
    }
cout<<"图像预处理平均用时:"<<tem/n<<"ms"<<endl;
cout<<"图像预处理最大用时:"<<max<<"ms"<<endl;
cout<<"图像预处理最小用时:"<<min<<"ms"<<endl;
cout<<endl;



 tem=0;max=0; min=10000;
    for(const auto& x: rknn_infer)
     {
        if(x>max) max=x;
        if(x<min) min=x;
        tem+=x;
    }
cout<<"模型推理平均用时:"<<tem/n<<"ms"<<endl;
cout<<"模型推理最大用时:"<<max<<"ms"<<endl;
cout<<"模型推理最小用时:"<<min<<"ms"<<endl;
cout<<endl;



 tem=0;max=0; min=10000;
    for(const auto& x: post_processing)
     {
        if(x>max) max=x;
        if(x<min) min=x;
        tem+=x;
    }
cout<<"模型输出后处理平均用时:"<<tem/n<<"ms"<<endl;
cout<<"模型输出后处理最大用时:"<<max<<"ms"<<endl;
cout<<"模型输出后处理最小用时:"<<min<<"ms"<<endl;
cout<<endl;



 tem=0;max=0; min=10000;
    for(const auto& x: yuan)
     {
        if(x>max) max=x;
        if(x<min) min=x;
        tem+=x;
    }
cout<<"椭圆拟合平均用时:"<<tem/n<<"ms"<<endl;
cout<<"椭圆拟合最大用时:"<<max<<"ms"<<endl;
cout<<"椭圆拟合最小用时:"<<min<<"ms"<<endl;
cout<<endl;


 tem=0;max=0; min=10000;
    for(const auto& x: zitai)
     {
        if(x>max) max=x;
        if(x<min) min=x;
        tem+=x;
    }
cout<<"姿态解算平均用时:"<<tem/n<<"ms"<<endl;
cout<<"姿态解算最大用时:"<<max<<"ms"<<endl;
cout<<"姿态解算最小用时:"<<min<<"ms"<<endl;
cout<<endl;



 tem=0;max=0; min=10000;
    for(const auto& x: full_frame)
     {
        if(x>max) max=x;
        if(x<min) min=x;
        tem+=x;
    }
cout<<"完整处理一帧平均用时:"<<tem/n<<"ms"<<endl;
cout<<"完整处理一帧最大用时:"<<max<<"ms"<<endl;
cout<<"完整处理一帧最小用时:"<<min<<"ms"<<endl;
cout<<endl;





//写进txt文本文件，进行画图处理
std::ofstream outFile1("./huatu/pre_processing.txt");
std::ofstream outFile2("./huatu/rknn_infer.txt");
std::ofstream outFile3("./huatu/post_processing.txt");
std::ofstream outFile4("./huatu/yuan.txt");
std::ofstream outFile5("./huatu/zitai.txt");
std::ofstream outFile6("./huatu/full_frame.txt");
 for(const auto& x: pre_processing)
  outFile1 << x << std::endl;
   for(const auto& x: rknn_infer)
  outFile2 << x << std::endl;
   for(const auto& x: post_processing)
  outFile3 << x << std::endl;
   for(const auto& x: yuan)
  outFile4 << x << std::endl;
   for(const auto& x: zitai)
  outFile5 << x << std::endl;
   for(const auto& x: full_frame)
  outFile6 << x << std::endl;


    return 0;
}