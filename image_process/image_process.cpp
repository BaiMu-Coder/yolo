#include "image_process.hpp"



image_process::image_process(std::unique_ptr<cv::Mat> frame):_src_image_frame(std::move(frame)){}

image_process::image_process(cv::Mat& frame):_src_image_frame(std::move(std::make_unique<cv::Mat>(frame))){}



int image_process::image_preprocessing(int target_size_x,int target_size_y)
{
    if(!_src_image_frame)
    {
        LOG_ERROR("_src_image_frame is nullptr");
        return -1;
    }
_letterbox.dst_w= target_size_x;
_letterbox.dst_h= target_size_y;
_letterbox.src_w=_src_image_frame->cols;
_letterbox.src_h=_src_image_frame->rows;
double tem_scale=1.0*target_size_x/_src_image_frame->cols;
double scale=1.0*target_size_y/_src_image_frame->rows;
if(tem_scale<scale)
scale=tem_scale;
_letterbox.scale=scale;

int resized_x=  static_cast<int>(_src_image_frame->cols*_letterbox.scale);  
int resized_y=  static_cast<int>(_src_image_frame->rows*_letterbox.scale);

_letterbox.upleft_pad_x=(target_size_x-resized_x)>>1;
_letterbox.upleft_pad_y=(target_size_y-resized_y)>>1;
_letterbox.lowright_pad_x=target_size_x-resized_x-_letterbox.upleft_pad_x;  //后处理要用到 处理上下左右填充不一致的情况，也就是小数情况
_letterbox.lowright_pad_y=target_size_y-resized_y-_letterbox.upleft_pad_y;



//先把原图像缩放到 上面计算出来的大小
auto tem_image=std::make_unique<cv::Mat>();
cv::resize(*_src_image_frame,*tem_image,cv::Size(resized_x,resized_y));

//创建目标大图,用144灰色填充,保持类型不变
auto resize_image=std::make_unique<cv::Mat>(target_size_x,target_size_y,_src_image_frame->type(),cv::Scalar(114,114,114));

//计算 把缩放的图像 放进大图里的 左上角位置
cv::Point position(_letterbox.upleft_pad_x, _letterbox.upleft_pad_y);  //设置缩放后的图像在大画布里放置的左上角位置

//定义ROI区域
cv::Rect roi(position.x, position.y, resized_x,resized_y);
//把缩放的图像拷贝进去
tem_image->copyTo((*resize_image)(roi));



// 做BGR转RGB
cv::cvtColor(*resize_image, *resize_image, cv::COLOR_BGR2RGB);


//判断输入图片类型是不是 8位无符号+3通道
if (resize_image->type() != CV_8UC3) {
   LOG_ERROR("input image type is not CV_8UC3");
        return -1;
}


_dst_image_frame=std::move(resize_image);

return 0;
}


unsigned char* image_process::get_image_buffer(int* size)
{
    if(!_dst_image_frame || _dst_image_frame->empty())
    {
      std::cout<<"_dst_image_frame is nullptr"<<std::endl;
        return nullptr;
    }
 
    if(size)
    {
        *size=_dst_image_frame->channels()*_dst_image_frame->total(); // 3*640*640
    }
    return _dst_image_frame->data;
}



letterbox& image_process::get_letterbox()
{
    return _letterbox;
}




