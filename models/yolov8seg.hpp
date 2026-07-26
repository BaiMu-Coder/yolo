#pragma once
#include "rknn_api.h"
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <utility>
#include "common.hpp"
#include "post_process_seg.hpp"
#include <opencv2/opencv.hpp>
#include "Float16.h"
#include "easy_timer.h"


class yolov8seg
{
 public:
  yolov8seg(std::string model_path);   
  
  ~yolov8seg();

 

  rknn_context* get_rknn_context();

  int set_npu_core(rknn_core_mask core_mask);


  int init(rknn_context* _ctx=nullptr);
  int exit();

 int set_input_data(void* image_data,int size);
 int rknn_model_inference();
 int get_output_data();
 int release_output_data();


 int post_process(object_detect_result_list& result , letterbox& letter_box);

 /**
  * @brief 从 RKNN 张量属性解析出的模型规格。
  *
  * 业务层只读取这里的尺寸，不再写死 640、80/40/20 或 Proto 160。
  * 因此启动时更换 640/1024 静态 RKNN 文件即可自动切换预处理和后处理。
  */
 struct ModelSpec
 {
   int input_width = 0;
   int input_height = 0;
   int input_channels = 0;
   int proto_width = 0;
   int proto_height = 0;
   int proto_channels = 0;
   int proto_output_index = -1;
   std::vector<cv::Size> detection_grids;
 };

 const ModelSpec& model_spec() const noexcept { return _model_spec; }
 cv::Size model_input_size() const noexcept
 {
   return {_model_spec.input_width, _model_spec.input_height};
 }

 private:
 public:
  int number_of_model_thread;
  rknn_context _ctx;
  std::string _model_path;

  //模型输入输出相关信息
  std::unique_ptr<rknn_tensor_attr[]> _input_tensor;
  std::unique_ptr<rknn_tensor_attr[]> _output_tensor;
  //输入输出的个数
  int _input_number;
  int _output_number;

  //模型输入
  int _model_channel;
  int _model_width;
  int _model_height;
  //模型是否量化
  bool _is_quant;


  //模型输入输出变量
  std::unique_ptr<rknn_input[]> _input;       //输入只有1个
  std::unique_ptr<rknn_output[]> _output;   //输出有13个


  // Proto数据(mask相关)
  int _proto_channel;
  int _proto_width;
  int _proto_height;
  int _proto_output_index = -1;
  std::vector<int> _branch_output_indices;
  ModelSpec _model_spec;

  //Proto反量化查表数据
   rknpu2::float16 _proto_table[256];
};
