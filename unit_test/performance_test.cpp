#include "yolov8seg.hpp"
#include "image_process.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <easy_timer.h>
#include "visual.hpp"


using namespace std;


static uint8_t clamp(uint8_t val, uint8_t min, uint8_t max)
{
    return val > min ? (val < max ? val : max) : min;
}



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



#define N_CLASS_COLORS 20 





TIMER ttt;

int main(int argc , char*argv[])
{

 int i=5;
while(i--)
{

int err=0;
string path("day.jpg");
cv::Mat frame11=cv::imread(path);
cv::Mat frame1=cv::imread(path);
cv::Mat frame=cv::imread(path);
if(frame11.empty())
{
  cout<<"imread error"<<endl;
    return -1;
}



ttt.tik();
image_process image(frame11);
image.image_preprocessing(640,640);
ttt.tok();
ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_image_preprocessing:");

int image_len;
void* image_data=image.get_image_buffer(&image_len);

ttt.tik();
yolov8seg yolo(argv[1]);
yolo.init();
yolo.set_input_data(image_data,image_len);
ttt.tok();
ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_yolo init:");



ttt.tik();
err=yolo.rknn_model_inference();
 if(err)
{
  cout<<"rknn_model_inference error"<<endl;
    return -1;
}
ttt.tok();
ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_rknn_model_inference:");


err=yolo.get_output_data();
 if(err)
{
  cout<<"get_output_data error"<<endl;
    return -1;
}


ttt.tik();
object_detect_result_list result;
letterbox letter_box=image.get_letterbox();


yolo.post_process(result,letter_box);
ttt.tok();
ttt.print_time_app("xxxxxxxxxxxxxxxxxxxx_post_process:");

cout<<result.count<<endl;



     // draw boxes
        for (int i = 0; i < result.count; i++)
        {      
cv::Point pt1(result.results_box[i].x, result.results_box[i].y);   // 左上角
cv::Point pt2(result.results_box[i].x+result.results_box[i].w ,  result.results_box[i].y+result.results_box[i].h);   // 右下角
cv::Scalar color(0, 0, 255);
cv::rectangle(frame, pt1, pt2, color, 2);  // thickness<0 就填充  
        }


     double alpha = 0.5;  // 透明度，可以自己调 0~1
        // draw mask
     // draw mask
    // 掩码：尺寸与 frame 一样，0 = 背景，>0 = 前景

if(result.results_mask[0].each_of_mask[0])
{
    cv::Mat mask(frame.rows, frame.cols, CV_8UC1,
                 result.results_mask[0].each_of_mask[0].get());

    // 只要 > 0 就认为是前景
    cv::Mat fgMask = (mask > 0);

    // 做一个 overlay
    cv::Mat overlay = frame.clone();

    // 想要的掩码颜色（BGR）这里用红色
    cv::Scalar maskColor(0, 0, 255);

    // 把前景区域涂成红色
    overlay.setTo(maskColor, fgMask);

    // 半透明叠加到原图上

    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame);
  }




         // draw mask
    // 掩码：尺寸与 frame 一样，0 = 背景，>0 = 前景
if(result.results_mask[0].each_of_mask[1]){
    cv::Mat mask1(frame.rows, frame.cols, CV_8UC1,
                 result.results_mask[0].each_of_mask[1].get());

    // 只要 > 0 就认为是前景
    cv::Mat fgMask1 = (mask1 > 0);

    // 做一个 overlay
    cv::Mat overlay1 = frame.clone();

    // 想要的掩码颜色（BGR）这里用红色
    cv::Scalar maskColor1(0, 255, 0);

    // 把前景区域涂成红色
    overlay1.setTo(maskColor1, fgMask1);

    // 半透明叠加到原图上
    cv::addWeighted(overlay1, alpha, frame, 1.0 - alpha, 0, frame);
    }


    
             // draw mask
    // 掩码：尺寸与 frame 一样，0 = 背景，>0 = 前景
if(result.results_mask[0].each_of_mask[2])
{
    cv::Mat mask2(frame.rows, frame.cols, CV_8UC1,
                 result.results_mask[0].each_of_mask[2].get());

    // 只要 > 0 就认为是前景
    cv::Mat fgMask2 = (mask2 > 0);

    // 做一个 overlay
    cv::Mat overlay2 = frame.clone();

    // 想要的掩码颜色（BGR）这里用红色
    cv::Scalar maskColor2(255, 255, 0);

    // 把前景区域涂成红色
    overlay2.setTo(maskColor2, fgMask2);

    // 半透明叠加到原图上
    cv::addWeighted(overlay2, alpha, frame, 1.0 - alpha, 0, frame);

    }



cv::imshow("1",frame);
cv::waitKey(0);




yolo.release_output_data();

};

    return 0;
}