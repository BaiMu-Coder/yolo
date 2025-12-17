#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "Float16.h"
#include "common.hpp"
#include <cstring>
 #include <iostream>
 #include <opencv2/opencv.hpp>
 #include <easy_timer.h>

template <typename T>  void swap(T&a,T&b)
{T c=a;a=b;b=c;}




int process_i8(std::unique_ptr<rknn_output[]>& output,std::unique_ptr<rknn_tensor_attr[]>& output_tensor,int index,         //输出信息  和  索引
     int grid_w, int grid_h, int model_w, int model_h, int stride,                    //格子大小（输出的大小）  ;  模型输入的大小  stride（步长/下采样倍数） ——也就是该输出特征图上 1 个网格格子对应输入图像上多少个像素。
     std::vector<float>& candidate_box ,std::vector<float>& box_score,std::vector<int>& class_id,  //候选框  置信度  类别
    std::unique_ptr<rknpu2::float16[]>& proto,std::vector<rknpu2::float16>& box_mask_coefficient,
    int proto_channel,int proto_width,int proto_height ,    //掩码系数部分
     float nms_threshold ) ;          //NMS阈值                               




//处理方法和上面类似
int process_fp32(std::unique_ptr<rknn_output[]>& output,std::unique_ptr<rknn_tensor_attr[]>& output_tensor,int index,         
     int grid_w, int grid_h, int model_w, int model_h, int stride ,                     
     std::vector<float>& candidate_box ,std::vector<float>& box_score,std::vector<int>& class_id, 
    std::unique_ptr<rknpu2::float16[]>& proto,std::vector<rknpu2::float16>& box_mask_coefficient,
    int proto_channel,int proto_width,int proto_height ,    
     float nms_threshold )   ;          






void quick_sort_desend_order( std::vector<float>& box_score,int left,int right, std::vector<int>& index_flag);


void nms(int valid_count,std::vector<int>& index_flag,std::vector<float>& candidate_box,std::vector<int>& class_id,int c,float nms_thresh);



std::pair<int,int> box_reverse(letterbox& letter_box , float x,float y) ;

int matrix_mult_by_npu_fp32(std::vector<rknpu2::float16>& filter_box_mask_coefficient,std::unique_ptr<rknpu2::float16[]>& proto,std::unique_ptr<float[]>& matrix_mult_result,int ROWS_A,int COLS_A,int COLS_B,rknn_context& ctx);
void matrix_mult_by_cpu_fp32(std::vector<float>& A,std::unique_ptr<float[]>& B,std::unique_ptr<float[]>& C, int ROWS_A, int COLS_A, int COLS_B);


void conbine_mak(std::unique_ptr<float[]>& mask_matrix_mult_result,std::unique_ptr<int8_t[]>& all_mask_in_one,std::vector<float>& filter_candidate_box_mask_conbine,std::vector<int>& mask_classid,int last_count,int proto_width,int proto_height);
void conbine_mak2(std::unique_ptr<float[]>& all_mask,std::unique_ptr<uint8_t[]>& all_mask_in_one, object_detect_result_list& result ,letterbox& letter_box);



void resize_by_opencv_fp(std::unique_ptr<float[]>& mask_matrix_mult_result,int last_count,int proto_width,int proto_height,
                    std::unique_ptr<float[]>& all_mask,letterbox& letter_box);
void resize_by_opencv_fp1(std::unique_ptr<float[]>& mask_matrix_mult_result,int last_count,int proto_width,int proto_height,
                    std::unique_ptr<float[]>& all_mask,letterbox& letter_box); //优化后