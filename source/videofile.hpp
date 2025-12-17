#pragma once 

#include <string>
#include <opencv2/opencv.hpp>
#include <memory>


class videofile
{
public:
videofile( std::string path);


std::unique_ptr<cv::Mat> get_next_frame();


private:
std::string _videofile_path;
std::unique_ptr<cv::VideoCapture> _cap;
};