#include <iostream>
#include <string>
#include "yolov8seg.hpp"
#include "image_process.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <unistd.h>

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



using namespace std;


struct EllipseResult {
    cv::RotatedRect rect;
    bool is_from_mask; 
};



    unsigned char class_colors[][3] = {
        {255, 56, 56},   // 'FF3838'
        {255, 157, 151}, // 'FF9D97'
        {255, 112, 31},  // 'FF701F'
        {255, 178, 29},  // 'FFB21D'
        {207, 210, 49},  // 'CFD231'
        {72, 249, 10},   // '48F90A'
        {146, 204, 23},  // '92CC17'
        {61, 219, 134},  // '3DDB86'
        {26, 147, 52},   // '1A9334'
        {0, 212, 187},   // '00D4BB'
        {44, 153, 168},  // '2C99A8'
        {0, 194, 255},   // '00C2FF'
        {52, 69, 147},   // '344593'
        {100, 115, 255}, // '6473FF'
        {0, 24, 236},    // '0018EC'
        {132, 56, 255},  // '8438FF'
        {82, 0, 133},    // '520085'
        {203, 56, 255},  // 'CB38FF'
        {255, 149, 200}, // 'FF95C8'
        {255, 55, 199}   // 'FF37C7'
    };






 // 1. 核心算法：椭圆拟合 (含偏差校验)
    EllipseResult calculate_best_ellipse(const cv::Mat& frame, 
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
                        } else { 
                            fit_success = false;
                            // 偏离太远，fit_success 保持 false，自动掉入下面的保底逻辑
                            // std::cout << "Mask ellipse rejected! Deviation: " << dist << " > " << limit_dist << std::endl;
                        }
                    }
                }
            }
        }

// fit_success = false;
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








int main(int argc, char **argv)
{

int err=0;
string path(argv[1]);
cv::Mat frame11=cv::imread(path);
cv::Mat frame1=cv::imread(path);
cv::Mat frame=cv::imread(path);
if(frame11.empty())
{
  cout<<"imread error"<<endl;
    return -1;
}

image_process image(frame11);
image.image_preprocessing(640,640);
int image_len;
void* image_data=image.get_image_buffer(&image_len);

yolov8seg yolo("new_yolov8_seg.rknn");
yolo.init();
yolo.set_input_data(image_data,image_len);
err=yolo.rknn_model_inference();
 if(err)
{
  cout<<"rknn_model_inference error"<<endl;
    return -1;
}
err=yolo.get_output_data();
 if(err)
{
  cout<<"get_output_data error"<<endl;
    return -1;
}

object_detect_result_list result;
letterbox letter_box=image.get_letterbox();


yolo.post_process(result,letter_box);

cout<<result.count<<endl;




// for(int i=0; i<result.count; ++i)
// {
//     int x1=result.results_box[i].x;
//     int y1=result.results_box[i].y;
//     int x2=result.results_box[i].x+result.results_box[i].w;
//     int y2=result.results_box[i].y+result.results_box[i].h;
//     printf("第%d个\n",i);
//     cout<<x1<<endl;
//     cout<<y1<<endl;
//     cout<<x2<<endl;
//     cout<<y2<<endl;
// cv::Point pt1(result.results_box[i].x, result.results_box[i].y);   // 左上角
// cv::Point pt2(result.results_box[i].x+result.results_box[i].w ,  result.results_box[i].y+result.results_box[i].h);   // 右下角
// cv::Scalar color(0, 0, 255);
// cv::rectangle(frame1, pt1, pt2, color, 0);  // thickness<0 就填充
// }




//    double alpha = 0.5;  // 透明度，可以自己调 0~1
//         // draw mask
//      // draw mask
//     // 掩码：尺寸与 frame 一样，0 = 背景，>0 = 前景

// if(result.results_mask[0].each_of_mask[0])
// {
//     cv::Mat mask(frame.rows, frame.cols, CV_8UC1,
//                  result.results_mask[0].each_of_mask[0].get());

//     // 只要 > 0 就认为是前景
//     cv::Mat fgMask = (mask > 0);

//     // 做一个 overlay
//     cv::Mat overlay = frame.clone();

//     // 想要的掩码颜色（BGR）这里用红色
//     cv::Scalar maskColor(0, 0, 255);

//     // 把前景区域涂成红色
//     overlay.setTo(maskColor, fgMask);

//     // 半透明叠加到原图上

//     cv::addWeighted(overlay, alpha, frame1, 1.0 - alpha, 0, frame1);
//   }




//          // draw mask
//     // 掩码：尺寸与 frame 一样，0 = 背景，>0 = 前景
// if(result.results_mask[0].each_of_mask[1]){
//     cv::Mat mask1(frame.rows, frame.cols, CV_8UC1,
//                  result.results_mask[0].each_of_mask[1].get());

//     // 只要 > 0 就认为是前景
//     cv::Mat fgMask1 = (mask1 > 0);

//     // 做一个 overlay
//     cv::Mat overlay1 = frame.clone();

//     // 想要的掩码颜色（BGR）这里用红色
//     cv::Scalar maskColor1(0, 255, 0);

//     // 把前景区域涂成红色
//     overlay1.setTo(maskColor1, fgMask1);

//     // 半透明叠加到原图上
//     cv::addWeighted(overlay1, alpha, frame1, 1.0 - alpha, 0, frame1);
//     }


    
//              // draw mask
//     // 掩码：尺寸与 frame 一样，0 = 背景，>0 = 前景
// if(result.results_mask[0].each_of_mask[2])
// {
//     cv::Mat mask2(frame.rows, frame.cols, CV_8UC1,
//                  result.results_mask[0].each_of_mask[2].get());

//     // 只要 > 0 就认为是前景
//     cv::Mat fgMask2 = (mask2 > 0);

//     // 做一个 overlay
//     cv::Mat overlay2 = frame.clone();

//     // 想要的掩码颜色（BGR）这里用红色
//     cv::Scalar maskColor2(255, 255, 0);

//     // 把前景区域涂成红色
//     overlay2.setTo(maskColor2, fgMask2);

//     // 半透明叠加到原图上
//     cv::addWeighted(overlay2, alpha, frame1, 1.0 - alpha, 0, frame1);

//     }




 draw_results_on_frame(frame1, result, true, 0.3);







cv::imshow("1",frame1);
cv::waitKey(0);



    
  sleep(20);
yolo.release_output_data();
    return 0;
}