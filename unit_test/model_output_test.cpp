#include <iostream>
#include "common.hpp"
#include "yolov8seg.hpp"
#include "image_process.hpp"

#include <easy_timer.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>
#include "visual.hpp"
using namespace std;





static inline  int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static float deqnt_affine_to_f32(int8_t proto, int32_t zp, float scale) // 这种处理方式是唯一解
{
  return ((float)proto - (float)zp) * scale;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}




int main(int argv , char* argc[])
{

  TIMER T;


  int err=0;
string path((argc[1]));
cv::Mat frame11=cv::imread(path);
cv::Mat frame=cv::imread(path);
if(frame11.empty())
{
  cout<<"imread error"<<endl;
    return -1;
}

T.tik();
image_process image(frame11);
image.image_preprocessing(640,640);
T.tok();
T.print_time("image_preprocessing");



int image_len;
uint8_t* image_data=image.get_image_buffer(&image_len);
if(!image_data)
{
  cout<<"get_image_buffer error"<<endl;
  return -1;
}



T.tik();
yolov8seg yolo("best.rknn");
yolo.init();
T.tok();
T.print_time("init ");
T.tik();
err=yolo.set_input_data(image_data,image_len);
 if(err!=RKNN_SUCC)
{
  cout<<"set_input_data error"<<endl;
    return -1;
}
T.tok();
T.print_time("set_input_data");


T.tik();
err=yolo.rknn_model_inference();
 if(err)
{
  cout<<"rknn_model_inference error"<<endl;
    return -1;
}
T.tok();
T.print_time("rknn_model_inference");





err=yolo.get_output_data();
 if(err)
{
  cout<<"get_output_data error"<<endl;
    return -1;
}


object_detect_result_list result;
letterbox letter_box=image.get_letterbox();



T.tik();
err=yolo.post_process(result,letter_box);
 if(err<0)
{
  cout<<"post_process error"<<endl;
    return -1;
}
T.tok();
T.print_time("post_process");

cout<<"result.count: "<<result.count<<endl;



// for(int i=0; i<result.count; ++i)
// {
//   cout<<"qqqqqqqqqqqqqq"<<endl;
//   cout<<result.results_box[i].x<<" "<<result.results_box[i].y<<" "<<result.results_box[i].w<<" "<<result.results_box[i].h<<endl;
// }



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






Visual visual(frame,&result);
visual.show_box("1");
// visual.show_mask("2");
visual.show("3");
visual.show_mask("2");


cv::waitKey(0);
// sleep(20);





yolo.release_output_data();
   return 0;
}