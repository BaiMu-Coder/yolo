#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include "common.hpp"


class image_process
{
public:
image_process(std::unique_ptr<cv::Mat> frame);
image_process(cv::Mat& frame);

int image_preprocessing(int target_size_x,int target_size_y);
unsigned char* get_image_buffer(int* size=nullptr);


int image_postprocessing();


letterbox& get_letterbox();


private:
public:
std::unique_ptr<cv::Mat>  _src_image_frame;
std::unique_ptr<cv::Mat>  _dst_image_frame;
letterbox  _letterbox;


};