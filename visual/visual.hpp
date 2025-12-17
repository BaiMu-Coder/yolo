#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include "common.hpp"

#define SCREEN_WIDTH  780
#define SCREEN_HEIGHT 1080


class Visual
{
public:
Visual(const cv::Mat& image , object_detect_result_list* result);
Visual(std::unique_ptr<cv::Mat>image , object_detect_result_list* result);

void show(std::string show_name); //box+mask

void show_box(std::string show_name);

void show_mask(std::string show_name);



private:
uint8_t _select;
std::unique_ptr<cv::Mat> _image;

object_detect_result_list* _result;

bool _have_box;
std::unique_ptr<cv::Mat> _image_box;

bool _have_mask;
std::unique_ptr<cv::Mat> _image_mask;

int _show_size;


bool _on_off=true;
};