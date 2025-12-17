#pragma once 
#include "yolov8seg.hpp"
#include "thread_pool.hpp"
#include <string>
#include <memory>
#include <vector>

class npu_infer_pool
{
public:

npu_infer_pool(std::string model_path  , int pool_size=1):_pool_size(pool_size),_model_path(model_path)
{
try{ 
//配置线程池
_pool=std::make_unique<ThreadPool>(_pool_size);
//每一个线程都需要加载一个模型
for(int i=0; i<_pool_size; ++i)
{
   _models.push_back(std::make_unique<yolov8seg>(_model_path)); 
}
}catch (const std::bad_alloc &e) {
    LOG_ERROR("Out of memory: {}", e.what());
    exit(EXIT_FAILURE);
}

//yolo模型的初始化
for(int i=0; i<_pool_size ; ++i)
{
int err=0;
if(i==0)
err=_models[i]->init();
else
err=_models[i]->init(_models[0]->get_rknn_context());

//设置每个模型运行在哪个NPU核心
switch(i%3)
{
 case 0 :
 _models[i]->set_npu_core(RKNN_NPU_CORE_0);
 break;
 case 1 :
 _models[i]->set_npu_core(RKNN_NPU_CORE_1);
 break;
 case 2 :
 _models[i]->set_npu_core(RKNN_NPU_CORE_2);
 break;
}

if(err!=0)
{
    LOG_ERROR("Init rknn model failed!");
    exit(err); 
}
}   
}


~npu_infer_pool()
{}



// void AddInferenceTask(std::shared_ptr<cv::Mat> src,ImageProcess &image_process);











private:
int _pool_size;
std::string _model_path;

std::unique_ptr<ThreadPool> _pool;
std::vector<std::unique_ptr<yolov8seg>> _models;

};

