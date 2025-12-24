#pragma once
#include <cstdio>
#include <memory>
#include <vector>


#define NMS_THRESH 0.45f   //框重复的比例
#define BOX_THRESH 0.5f    //置信度阈值
#define OBJ_NUMB_MAX_SIZE 256   //最大检测的数目



// 最基础版，输出到 stderr
#define LOG_ERROR(fmt, ...)                                        \
    do                                                             \
    {                                                              \
        std::fprintf(stderr, "[ERROR] %s:%d %s() : " fmt "\n",     \
                     __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    } while (0)

#define LOG_INFO(fmt, ...)                                         \
    do                                                             \
    {                                                              \
        std::fprintf(stdout, "[INFO ] %s:%d %s() : " fmt "\n",     \
                     __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    } while (0)


typedef struct letterbox
{
//原始图尺寸
int src_w;
int src_h;

//目标输入尺寸
int dst_w;
int dst_h;

//缩放比例
double  scale;

//左右和上下的padding
int upleft_pad_x;
int upleft_pad_y;
int lowright_pad_x;
int lowright_pad_y;

}letterbox;




typedef struct
{
    int x,y,w,h;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct
{
  std::unique_ptr<uint8_t[]> seg_mask;
  std::vector<std::unique_ptr<uint8_t[]>> each_of_mask;   //此项目定制 记录每个类别的掩码
} object_segment_result;

typedef struct
{
    int count=0;
    object_detect_result results_box[OBJ_NUMB_MAX_SIZE];
    object_segment_result results_mask[1];  //所有mask合并为1个 放在这里
} object_detect_result_list;





