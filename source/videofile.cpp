#include "videofile.hpp"
#include "common.hpp"
#include <iostream>
#include <cstdio>


videofile::videofile( std::string path):_videofile_path(std::move(path))
{
_cap=std::make_unique<cv::VideoCapture>(_videofile_path);
if(!_cap->isOpened())
{
 LOG_ERROR("open video failed: %s", _videofile_path.c_str());
 throw std::runtime_error("open video failed");  //打开视频失败 抛异常出去  走到这里就跳出去了
}
}



std::unique_ptr<cv::Mat> videofile::get_next_frame()
{
//  auto frame=std::make_unique<cv::Mat>(); 
 std::unique_ptr<cv::Mat> frame=std::make_unique<cv::Mat>(); 
 *_cap>>(*frame);
 if(frame->empty())
 {
 LOG_ERROR("read video frame failed ");
 throw std::runtime_error("read video frame failed");  
 }
 return move(frame);
}