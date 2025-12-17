#include "visual.hpp"
#include <algorithm>


static void show_screen(const cv::Mat& image,const std::string& name)
{
    double scale = std::min(
                (double)SCREEN_WIDTH / image.cols,
                (double)SCREEN_HEIGHT / image.rows);
    cv::Mat show;
    cv::resize(image, show, cv::Size(), scale,scale); // 等比例缩放到刚好放进屏幕  scaleb表示 宽 高 缩放的倍数
    // 显示这一帧
    cv::imshow(name, show);

}

static const std::vector<cv::Scalar> kClassColors = {
    cv::Scalar(0,   0, 255 ),   // 1: 红色
    cv::Scalar(0, 255,   0),   // 2: 绿色
    cv::Scalar(255, 0,   0),   // 3: 蓝色
    cv::Scalar(0, 255, 255),   // 4: 黄色
    cv::Scalar(255, 0, 255),   // 5: 品红
    cv::Scalar(255,255, 0),   // 6: 青色
    cv::Scalar(  0, 165, 255),  // 7: 橙
    cv::Scalar(128,   0, 128),  // 8: 紫
    cv::Scalar(  0, 128, 255),  // 9: 浅橙 / 金
    cv::Scalar(128, 255,   0),  //10: 黄绿
    cv::Scalar(255, 128,   0),  //11: 浅蓝
    cv::Scalar(  0, 128, 128),  //12: 青绿
    cv::Scalar(128,   0, 255),  //13: 紫红
    cv::Scalar(128, 128,   0),  //14: 暗青

    cv::Scalar(  0,  64, 255),  //15: 深橙红
    cv::Scalar(  0, 255, 128),  //16: 草绿色
    cv::Scalar(255,  64,   0),  //17: 深蓝
    cv::Scalar( 64,   0, 255),  //18: 亮紫
    cv::Scalar(255,   0, 128),  //19: 玫红
    cv::Scalar(128, 255, 255),  //20: 浅黄
    cv::Scalar(255, 128, 255),  //21: 浅粉
    cv::Scalar(255, 255, 128),  //22: 浅青
    cv::Scalar(128, 128, 255),  //23: 浅紫蓝
    cv::Scalar(128, 255, 128),  //24: 浅黄绿
    cv::Scalar(255, 128, 128),  //25: 浅红
    
};



Visual::Visual(const cv::Mat& image , object_detect_result_list* result):_result(result),_have_box(false),_have_mask(false)
{
_image=std::make_unique<cv::Mat>(image.clone());
_show_size=std::min(SCREEN_WIDTH,SCREEN_HEIGHT);
}


Visual::Visual(std::unique_ptr<cv::Mat> image , object_detect_result_list* result):_image(std::move(image)),_result(result),_have_box(false),_have_mask(false)
{
_show_size=std::max(SCREEN_WIDTH,SCREEN_HEIGHT);
}




void Visual::show(std::string show_name)
{
if(!_on_off)   return ;
if(_have_mask)
{
 // draw boxes
for (int i = 0; i < _result->count; i++)
        {      
cv::Point pt1(_result->results_box[i].x, _result->results_box[i].y);   // 左上角
cv::Point pt2(_result->results_box[i].x+_result->results_box[i].w , _result->results_box[i].y+_result->results_box[i].h);   // 右下角
cv::Scalar color(0, 0, 255); //设置框的颜色BGR
cv::rectangle(*_image_mask, pt1, pt2, color, 0);  // thickness<0 就填充  
        }
show_screen(*_image_mask,show_name); 
return ;
}
_have_mask=true;
_image_mask=std::make_unique<cv::Mat>(*_image);  //浅拷贝直接拿原图像存的地方来用

   //建一张“纯颜色的 mask 图”，大小和原图一样
    cv::Mat color_mask(_image_mask->size(), _image_mask->type(), cv::Scalar::all(0));

        for (int y = 0; y < _image_mask->rows; ++y)        //rows表示高度，垂直方向多少个像素；  cols表示宽度，水平方向有多少个像素
    {
        cv::Vec3b*   pColor = color_mask.ptr<cv::Vec3b>(y);   //返回第y行起始地址

        for (int x = 0; x < _image_mask->cols; ++x)
        {
            uint cls_id = _result->results_mask->seg_mask[y*_image_mask->cols+x];
            // std::cout<<cls_id;
            if (cls_id == 0 )
                continue;  // 背景保持黑色(0,0,0)，之后叠加不会影响

            const cv::Scalar& c = kClassColors[cls_id%kClassColors.size()];
            pColor[x] = cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]);
        }
        // std::cout<<std::endl;
    }

  // 3. 把 color_mask 半透明叠加到 vis 上
    //    vis = alpha * color_mask + (1-alpha) * image
    float alpha=0.5;
// show_screen(color_mask,"3333333333");

cv::addWeighted(color_mask, alpha, *_image, 1.0f - alpha, 0.0, *_image_mask);

//box
for (int i = 0; i < _result->count; i++)
        {      
cv::Point pt1(_result->results_box[i].x, _result->results_box[i].y);   // 左上角
cv::Point pt2(_result->results_box[i].x+_result->results_box[i].w , _result->results_box[i].y+_result->results_box[i].h);   // 右下角
cv::Scalar color(0, 0, 255); //设置框的颜色BGR
cv::rectangle(*_image_mask, pt1, pt2, color, 0);  // thickness<0 就填充  
        }

show_screen(*_image_mask,show_name);
return ;





} 



void Visual::show_box(std::string show_name)
{
    if(!_on_off)   return ;
   if(_have_box)
   {
     show_screen(*_image_box,show_name);
    return ;
   }
   _have_box=true;

//_image_box=std::make_unique<cv::Mat>(*_image); //这样写是浅拷贝,共用一块内存
// _image_mask = std::make_unique<cv::Mat>(_image->clone()); //深拷贝

     _image_box=std::make_unique<cv::Mat>();
     _image->copyTo(*_image_box);
 // draw boxes
for (int i = 0; i < _result->count; i++)
        {      
cv::Point pt1(_result->results_box[i].x, _result->results_box[i].y);   // 左上角
cv::Point pt2(_result->results_box[i].x+_result->results_box[i].w , _result->results_box[i].y+_result->results_box[i].h);   // 右下角
cv::Scalar color(0, 0, 255); //设置框的颜色BGR
cv::rectangle(*_image_box, pt1, pt2, color, 0);  // thickness<0 就填充  
        }
show_screen(*_image_box,show_name);

return ;
}



void Visual::show_mask(std::string show_name)
{
    if(!_on_off)   return ;
   if(_have_mask)
   {
 show_screen(*_image_mask,show_name);
    return;
   }
   _have_mask=true;
_image_mask=std::make_unique<cv::Mat>(*_image);  //浅拷贝直接拿原图像存的地方来用

   //建一张“纯颜色的 mask 图”，大小和原图一样
    cv::Mat color_mask(_image_mask->size(), _image_mask->type(), cv::Scalar::all(0));

        for (int y = 0; y < _image_mask->rows; ++y)        //rows表示高度，垂直方向多少个像素；  cols表示宽度，水平方向有多少个像素
    {
        cv::Vec3b*   pColor = color_mask.ptr<cv::Vec3b>(y);   //返回第y行起始地址

        for (int x = 0; x < _image_mask->cols; ++x)
        {
            uint cls_id = _result->results_mask->seg_mask[y*_image_mask->cols+x];
            // std::cout<<cls_id;
            if (cls_id == 0 )
                continue;  // 背景保持黑色(0,0,0)，之后叠加不会影响

            const cv::Scalar& c = kClassColors[cls_id%kClassColors.size()];
            pColor[x] = cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]);
        }
        // std::cout<<std::endl;
    }

  // 3. 把 color_mask 半透明叠加到 vis 上
    //    vis = alpha * color_mask + (1-alpha) * image
    float alpha=0.5;
// show_screen(color_mask,"3333333333");

cv::addWeighted(color_mask, alpha, *_image, 1.0f - alpha, 0.0, *_image_mask);
show_screen(*_image_mask,show_name);
return ;
}





