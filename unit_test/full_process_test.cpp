#include "yolov8seg.hpp"
#include "image_process.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <unistd.h>

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






int main()
{
int err=0;
string path("bus.jpg");
cv::Mat frame11=cv::imread(path);
cv::Mat frame1=cv::imread(path);
if(frame11.empty())
{
  cout<<"imread error"<<endl;
    return -1;
}

image_process image(frame11);
image.image_preprocessing(640,640);
int image_len;
void* image_data=image.get_image_buffer(&image_len);

yolov8seg yolo("yolov8_seg.rknn");
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




for(int i=0; i<result.count; ++i)
{
    int x1=result.results_box[i].x;
    int y1=result.results_box[i].y;
    int x2=result.results_box[i].x+result.results_box[i].w;
    int y2=result.results_box[i].y+result.results_box[i].h;
    printf("第%d个\n",i);
    cout<<x1<<endl;
    cout<<y1<<endl;
    cout<<x2<<endl;
    cout<<y2<<endl;
cv::Point pt1(result.results_box[i].x, result.results_box[i].y);   // 左上角
cv::Point pt2(result.results_box[i].x+result.results_box[i].w ,  result.results_box[i].y+result.results_box[i].h);   // 右下角
cv::Scalar color(0, 0, 255);
cv::rectangle(frame1, pt1, pt2, color, 0);  // thickness<0 就填充
}


cv::imshow("1",frame1);
cv::waitKey(0);



    
  sleep(20);
yolo.release_output_data();
    return 0;
}