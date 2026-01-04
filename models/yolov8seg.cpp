#include "yolov8seg.hpp"
#include "common.hpp"
#include <memory>
#include <cstdio>
#include <iostream>
#include <cstring>


static std::unique_ptr<char[]> read_data_from_file(const std::string &path, int &len)
{
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        LOG_ERROR("fopen model file error, %s", path.c_str());
        len = -1;
        fclose(fp);
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    len = ftell(fp);
    if (len <= 0)
    {
        fclose(fp);
        LOG_ERROR("model file size error");
        return nullptr;
    }
    fseek(fp, 0, SEEK_SET);

    std::unique_ptr<char[]> model = std::make_unique<char[]>(len);
    // auto model = std::make_unique<char[]>(len);

    size_t n = fread(model.get(), 1, len, fp);
    fclose(fp);

    if (n != (size_t)len)
    {
        len = -1;
        LOG_ERROR("read model file size error");
        return nullptr;
    }

    return move(model);
}

static void printf_rknn_tensor_attr(const rknn_tensor_attr *attr)
{
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "index:  " << attr->index << std::endl;
    std::cout << "n_elems:  " << attr->n_elems << std::endl;
    std::cout << "n_dims:  " << attr->n_dims << std::endl;
    std::cout << "dims: [";
    for (int i = 0; i < RKNN_MAX_DIMS; ++i)
        std::cout << " " << (attr->dims)[i];
    std::cout << " ]" << std::endl;

    std::cout << "name:  ";
    for (int i = 0; i < RKNN_MAX_NAME_LEN; ++i)
        std::cout << (attr->name)[i];
    std::cout << std::endl;

    std::cout << "fmt:  " << get_format_string(attr->fmt) << std::endl;
    std::cout << "type:  " << get_type_string(attr->type) << std::endl;             // 表示这个 tensor 在内存里用什么“数据类型”存的
    std::cout << "qnt_type:  " << get_qnt_type_string(attr->qnt_type) << std::endl; // 示这个 tensor 有没有做“量化”，以及用的什么量化方式

    std::cout << "fl:  " << attr->fl << std::endl;
    std::cout << "zp:  " << attr->zp << std::endl;
    std::cout << "scale:  " << attr->scale << std::endl;
    std::cout << "w_stride:  " << attr->w_stride << std::endl;
    std::cout << "size_with_stride:  " << attr->size_with_stride << std::endl;
    std::cout << "pass_through:  " << attr->pass_through << std::endl;
    std::cout << "h_stride:  " << attr->h_stride << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
}

yolov8seg::yolov8seg(std::string model_path) : _model_path(model_path),_ctx(0) ,_output(nullptr),_input(nullptr){}


yolov8seg::~yolov8seg()
{
    if(_ctx)
    {
    int err=rknn_destroy(_ctx);
    if(err!=RKNN_SUCC)
    {
        std::cout<<"rknn_destroy error, errno:"<<err<<std::endl;
    }
    }
}





int yolov8seg::init(rknn_context *ctx )
{
    int err = 0;

    // 创建RKNN对象   因为线程池里面的每个模型都是一样的  所以后面的模型直接复制
    if (ctx)
    {
        err = rknn_dup_context(ctx, &_ctx);
        if (err != RKNN_SUCC)
        {
            LOG_ERROR("rknn_dup_context error,errno:%d", err);
            return err;
        }
    }
    else
    {
        // 读取RKNN文件
        int model_len = 0;
        auto model_file_buffer = read_data_from_file(_model_path, model_len);
        if (!model_file_buffer || model_len <= 0)
        {
            LOG_ERROR("read_data_from_file function error");
            return -1;
        }

        // 创建RKNN对象
        err = rknn_init(&_ctx, model_file_buffer.get(), model_len, 0, NULL);
        if (err != RKNN_SUCC)
        {
            LOG_ERROR("rknn_init error,errno:%d", err);
            return err;
        }
    }

    // 设置运行这个模型的线程 运行在哪个NPU核心上面
    err = rknn_set_core_mask(_ctx, RKNN_NPU_CORE_AUTO);
    if (err != RKNN_SUCC)
    {
        LOG_ERROR("rknn_set_core_mask error,errno:%d", err);
        return err;
    }

    // 获取RKNN 的SDK版本信息
    rknn_sdk_version version;
    err = rknn_query(_ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (err != RKNN_SUCC)
    {
        LOG_ERROR("rknn_query SDK_VERSION error,errno:%d", err);
        return err;
    }
    printf("sdk api version: %s\n", version.api_version);
    printf("driver version: %s\n", version.drv_version);

    // 查询输入输出tensor个数
    rknn_input_output_num io_number;
    err = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &io_number, sizeof(io_number));
    if (err != RKNN_SUCC)
    {
        LOG_ERROR("rknn_query IN_OUT_NUM error,errno:%d", err);
        return err;
    }
    _input_number= io_number.n_input;
    _output_number=io_number.n_output;
    printf("model input num: %d, output num: %d\n", io_number.n_input, io_number.n_output);

    // 查询输入信息
    std::cout << "input tensors: " << std::endl;
    auto input_tensor = std::make_unique<rknn_tensor_attr[]>(io_number.n_input);
    memset(input_tensor.get(), 0, sizeof(rknn_tensor_attr) * io_number.n_input);
    for (int i = 0; i < io_number.n_input; ++i)
    {
        input_tensor[i].index = i;
        err = rknn_query(_ctx, RKNN_QUERY_INPUT_ATTR, &(input_tensor[i]), sizeof(rknn_tensor_attr));
        if (err != RKNN_SUCC)
        {
            LOG_ERROR("rknn_query RKNN_QUERY_INPUT_ATTR error,errno:%d", err);
            return err;
        }
        // printf_rknn_tensor_attr(&(input_tensor[i]));
    }

    // 查询输出信息
    std::cout << "output tensors: " << std::endl;
    auto output_tensor = std::make_unique<rknn_tensor_attr[]>(io_number.n_output);
    memset(output_tensor.get(), 0, sizeof(rknn_tensor_attr) * io_number.n_output);
    if (io_number.n_output != 13)
    {
        LOG_ERROR("The output is not 13 (io_number.n_output)"); // 经过 瑞芯微模型转换  分割模型的数据是13个
        return -1;
    }
    for (int i = 0; i < io_number.n_output; ++i)
    {
        output_tensor[i].index = i;
        err = rknn_query(_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_tensor[i]), sizeof(rknn_tensor_attr));
        if (err != RKNN_SUCC)
        {
            LOG_ERROR("rknn_query RKNN_QUERY_OUTPUT_ATTR error,errno:%d", err);
            return err;
        }
        // printf_rknn_tensor_attr(&(output_tensor[i]));
    }
     

    //从上面获取的数据里面设置一些相关信息
    if (output_tensor[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_tensor[0].type == RKNN_TENSOR_INT8)
    {
        _is_quant = true;
    }
    else
    {
        _is_quant = false;
    }


    if (input_tensor[0].fmt == RKNN_TENSOR_NCHW)
  {
    std::cout<<"model is NCHW input fmt"<<std::endl;
    _model_channel = input_tensor[0].dims[1];
    _model_height = input_tensor[0].dims[2];
    _model_width = input_tensor[0].dims[3];
  }
  else if(input_tensor[0].fmt == RKNN_TENSOR_NHWC)
  {
   std::cout<<"model is NHWC input fmt"<<std::endl;
    _model_height = input_tensor[0].dims[1];
    _model_width = input_tensor[0].dims[2];
    _model_channel = input_tensor[0].dims[3];
  }
  else
  {
    std::cout<<"input_tensor[0].fmt doesn't exist"<<std::endl;
  }



    if (output_tensor[io_number.n_output-1].fmt == RKNN_TENSOR_NCHW)
  {
    std::cout<<"model proto is NCHW fmt"<<std::endl;
    _proto_channel = output_tensor[io_number.n_output-1].dims[1];
    _proto_height = output_tensor[io_number.n_output-1].dims[2];
    _proto_width = output_tensor[io_number.n_output-1].dims[3];
  }
  else if(input_tensor[io_number.n_output-1].fmt == RKNN_TENSOR_NHWC)
  {
   std::cout<<"model proto is NHWC fmt"<<std::endl;
    _proto_height = output_tensor[io_number.n_output-1].dims[1];
    _proto_width = output_tensor[io_number.n_output-1].dims[2];
    _proto_channel = output_tensor[io_number.n_output-1].dims[3];
  }
  else
  {
    std::cout<<"output_tensor[12].fmt doesn't exist"<<std::endl;
  }


  _output_tensor=std::move(output_tensor);
  _input_tensor=std::move(input_tensor);


  process_i8_index12_init(_output_tensor,_proto_table);


return 0;
}


int yolov8seg::set_npu_core(rknn_core_mask core_mask)
{
    return rknn_set_core_mask(_ctx, core_mask);
}



 int yolov8seg::set_input_data(void* image_data,int size)
 {
    if(!image_data)
    {
LOG_ERROR("image_data is nullptr");
    return -1;
    }
 
    if(!_input)
  _input=std::make_unique<rknn_input[]>(_input_number);
  memset(&(_input[0]),0,sizeof(rknn_input)*_input_number);

  _input[0].buf=image_data;
  _input[0].index=0;
  _input[0].size=size;
  _input[0].pass_through=0;  //让RKNN做量化预处理
  _input[0].type = RKNN_TENSOR_UINT8;   //这里就填输入数据是什么格式填什么格式就行,老老实实告诉RKNN让他帮你做预处理
  _input[0].fmt=RKNN_TENSOR_NHWC;    
  
  return rknn_inputs_set(_ctx,1,&(_input[0]));
 }



 int yolov8seg::rknn_model_inference()
 {
     return rknn_run(_ctx,NULL);
 }


int yolov8seg::get_output_data()
{
if(!_output)
 _output=std::make_unique<rknn_output[]>(_output_number);
 
 memset(&(_output[0]),0,sizeof(rknn_output)*_output_number);   //rknn_output一共五个成员变量，下面三个需要自己设置，另两个返回设置
 for(int i=0; i<_output_number; ++i)
 {
    _output[i].index=i;                       //索引位置
    _output[i].is_prealloc=0;                 //标识存放输出数据是否是预分配，该字段由用户设置。 0未分配
    _output[i].want_float=(!_is_quant);       //标识是否需要将数据转为float类型输出
     //is_quant == 0（非量化模型）→ want_float = 1（无所谓，输出本来就是 float）
     //is_quant == 1（量化模型）→ want_float = 0（你将拿到 INT8/UINT8 原始输出，需要自己反量化或后处理能直接用量化值）
 } 
 return rknn_outputs_get(_ctx,_output_number,&(_output[0]),NULL);
}


 int yolov8seg::release_output_data()
 {
return rknn_outputs_release(_ctx,_output_number,&(_output[0]));
 }



int yolov8seg::post_process(object_detect_result_list& result , letterbox& letter_box)
 {

  TIMER xxx;

    std::vector<float> candidate_box;  //保存候选框  四个一组  x,y,w,h  
    std::vector<float> box_score;     //每个候选框的分类置信度
    std::vector<int> class_id;        //每个候选框的id

    std::vector< rknpu2::float16> box_mask_coefficient;  //每个候选框对应的mask 系数（长度 PROTO_CHANNEL）
    auto proto=std::unique_ptr< rknpu2::float16[]>( new rknpu2::float16[_proto_channel * _proto_width * _proto_height]); //Proto 原型掩码（大小 C*Hp*Wp），只会在处理到 proto 输出的那一次被填充。
    std::vector< rknpu2::float16> filter_box_mask_coefficient;  //经过 NMS 筛选剩下的那些候选的系数，用于最终的 matmul也就是生成掩码
   
    int valid_count=0;



     int dfl_len =_output_tensor[0].dims[1] / 4;//用回归张量的通道数反推出 DFL 的桶数（= reg_max+1），后处理时用于 DFL 解码。 dims[1]就是物品的类别数
     //YOLOv8 / YOLOv5（从 6.x 开始）用的是DLF 回归框：
    // 不是直接输出 4 个值（cx, cy, w, h）
    // 而是输出 4 × (reg_max+1) 个值
    // 每个边界都用一个分布来学习（类似分类概率）

     int output_per_branch = _output_number / 3;// default 3 branch    输出有 3 个尺度分支（stride=8/16/32），每个尺度对应一个网格（80×80、40×40、20×20）。
      //在 RKNN 优化后的模型里，这三块经常被拆成 3 个（或 4 个）独立的输出张量，而不是一个大张量。
       //再加上一个单独的 Proto 张量，所以你会在 io_num.n_output 里看到总输出数 ≈ 3个尺度 × 每尺度的输出个数 + 1(proto)，常见是 13 个（12 + 1）



    xxx.tik();
      //4+4+4+1）一共13个输出  
      for(int i=0; i<_output_number ; ++i)
      {
       int grid_h , grid_w;
      // int stride = _model_height / grid_h;   //这行在算 feature stride（步长/下采样倍数） ——也就是该输出特征图上 1 个网格格子对应输入图像上多少个像素。
       if (_output_tensor[i].fmt == RKNN_TENSOR_NCHW)
       {
       grid_h = _output_tensor[i].dims[2];
       grid_w = _output_tensor[i].dims[3];
         }
         else if(_output_tensor[i].fmt == RKNN_TENSOR_NHWC)
         {
         grid_h = _output_tensor[i].dims[1];
         grid_w = _output_tensor[i].dims[2];
        }
        else
        {
            LOG_ERROR("_output_tensor[i].fmt is error");
            return -1;
        }

        int stride = _model_height / grid_h;
   
       if(_is_quant)
       {
        valid_count+=process_i8(_output,_output_tensor,i,grid_w,grid_h,_model_width,_model_height,stride,candidate_box,box_score,class_id,proto,box_mask_coefficient,_proto_channel,_proto_width,_proto_height,BOX_THRESH,_proto_table);
       }
       else
       {
        valid_count+=process_fp32(_output,_output_tensor,i,grid_w,grid_h,_model_width,_model_height,stride,candidate_box,box_score,class_id,proto,box_mask_coefficient,_proto_channel,_proto_width,_proto_height,BOX_THRESH);
       }
      }



xxx.tok();
xxx.print_time("process_i8");

// std::cout<<"validCount size :"<<valid_count<<std::endl;
      if(valid_count<=0)
      {
       return 0;   //未检测到物体
      }

 
      //nms,进行同一个类别重复框的过滤（覆盖超过我们设定的阈值就过滤掉）
       
         //首先进行置信度（评分的排序） ,因为过滤框,肯定是保留高分，过滤低分
          std::vector<int> index_flag;
          for(int i=0; i<valid_count; ++i)
             index_flag.push_back(i);         //这个的作用就是，保存 预测框-分数-类别-掩码 这几个vector的对应关系,因为排序了分数，你还要能对应上其他的信息，所以这里放一个来存这些信息
      
         
         //按分数进行降序排序，把index_flag同步调整，就复制分数的调换数据的步骤就行
         quick_sort_desend_order(box_score, 0, valid_count-1, index_flag);
    
         //进行按类别筛选   把不要的index_flag里面对应的地方置-1,也就是断开他们的联系
         std::set<int> class_id_set(class_id.begin(),class_id.end());
      
          for(const auto& c:class_id_set)
          {
            nms(valid_count,index_flag,candidate_box,class_id,c,NMS_THRESH);  
          }

        


          //最后：把框筛选出来  以及 把mask系数也提取出来
int last_count = 0;//记录最终的检测数量
  std::vector<float> filter_candidate_box;//存储 筛选后的框xywh
  std::vector<float> filter_candidate_box_mask_conbine;
  std::vector<int> mask_classid;
          for(int i=0; i<valid_count; ++i)
          {
             if (index_flag[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)   continue;
      
        int n=index_flag[i];
        float x1 = candidate_box[n * 4 + 0];
        float y1 = candidate_box[n * 4 + 1];
        float x2 = x1 + candidate_box[n * 4 + 2];
        float y2 = y1 + candidate_box[n * 4 + 3];
       

        int id = class_id[n];
        float obj_conf = box_score[i];

        result.results_box[last_count].cls_id=id;
        result.results_box[last_count].prop=obj_conf;
        filter_candidate_box.push_back(x1);
        filter_candidate_box.push_back(y1);
        filter_candidate_box.push_back(x2);
        filter_candidate_box.push_back(y2); //这里坐标先放里面暂存一下  后面还有转换回原图坐标系里面的(这里的坐标是640*640下的坐标)
           
        filter_candidate_box_mask_conbine.push_back(x1/4);
        filter_candidate_box_mask_conbine.push_back(y1/4);
        filter_candidate_box_mask_conbine.push_back(x2/4);
        filter_candidate_box_mask_conbine.push_back(y2/4); //这里的检测框给掩码使用，掩码层的分辨率是160*160 所以这里要缩小4倍与之对应
        mask_classid.push_back(id);
        //mask系数提取出来过滤后的
        for(int j=0; j<_proto_channel; ++j)
        filter_box_mask_coefficient.push_back(box_mask_coefficient[n*_proto_channel+j]);
         last_count++;
          }
result.count=last_count;
 //nms部分结束
 



    //框坐标转换，由放缩填充后的转换为原图坐标系
    for(int i=0;i<last_count;++i)
    {
      std::pair<int,int> x=box_reverse(letter_box,filter_candidate_box[i*4],filter_candidate_box[i*4+1]);
      std::pair<int,int> y=box_reverse(letter_box,filter_candidate_box[i*4+2],filter_candidate_box[i*4+3]);
       result.results_box[i].x=x.first;
       result.results_box[i].y=x.second;
       result.results_box[i].w=y.first-x.first;
       result.results_box[i].h=y.second-x.second;
    }




  
    //计算mask掩码信息，用mask系数和proto来计算  （矩阵乘法）
    int ROWS_A=last_count;   //行数
    int COLS_A=_proto_channel;  //列数
    int COLS_B=_proto_height*_proto_width;
  

auto mask_matrix_mult_result=std::unique_ptr<float[]>(new float[ROWS_A*COLS_B]);




   xxx.tik();
int err=matrix_mult_by_npu_fp32(filter_box_mask_coefficient,proto,mask_matrix_mult_result,ROWS_A,COLS_A,COLS_B); //直接拿浮点数进行计算，整体体量小,量化int8提升也很小
 if(err!=RKNN_SUCC)
 {
   LOG_ERROR("matrix_mult_by_npu_fp32 fail, errno:%d", err);
  return err;
 }


// matrix_mult_by_cpu_fp32(filter_box_mask_coefficient,proto,mask_matrix_mult_result,ROWS_A,COLS_A,COLS_B);
xxx.tok();
xxx.print_time("matrix_mult_by_npu_fp32");







#ifdef XXX
 /*     方案1  先逐张合并在整体放缩      */
 /*主要区别就是 先合并的话mask掩码先变为整数了，会导致后面放缩的时候 掩码边界地方处理的不是很平滑*/
 /*效率会快一点，当检测结果越多越显著*/
 //把所有掩码合成写到一张图上面 160*160
 auto all_mask_in_one=std::make_unique<int8_t[]>(_proto_height*_proto_width);
 memset(all_mask_in_one.get(),0,sizeof(int8_t)*_proto_height*_proto_width);
 conbine_mak(mask_matrix_mult_result,all_mask_in_one,filter_candidate_box_mask_conbine,mask_classid,last_count,_proto_width,_proto_height);


 //得到真实mask,处理为原图尺寸
    int tem_leftx=letter_box.upleft_pad_x/4;
    int tem_rightx=letter_box.lowright_pad_x/4;
    int tem_lefty=letter_box.upleft_pad_y/4;
    int tem_righty=letter_box.lowright_pad_y/4;
    std::cout<<tem_leftx<<" "<<tem_rightx<<" "<<tem_lefty<<" "<<tem_righty<<std::endl;

    int padx= (letter_box.lowright_pad_x+letter_box.upleft_pad_x)/4;
    int pady= (letter_box.lowright_pad_y+letter_box.upleft_pad_y)/4;
    int conbine_width = _proto_width - padx;  //
    int conbine_height= _proto_height- pady;
    int real_width = letter_box.src_w; //原始输入图像尺寸
    int real_height = letter_box.src_h;
  auto conbine_mask_crop_pad=std::make_unique<int8_t[]>(conbine_width*conbine_height);
  auto real_mask=std::make_unique<uint8_t[]>(real_width*real_height);
  int cropped_index=0;
  for(int i=0; i<_proto_height;++i)
  {
  for(int j=0; j<_proto_width;++j)
 { 
    if(j >= tem_leftx && j < _proto_width-tem_rightx && i >= tem_lefty && i < _proto_height - tem_righty)
     conbine_mask_crop_pad[cropped_index++] = all_mask_in_one[i*_proto_width+j];        //把上面合并出来的mask减去填充部分得到新的mask ， 来进行缩放
 }
 }

    cv::Mat src_image(conbine_height, conbine_width, CV_8U, conbine_mask_crop_pad.get());
    cv::Mat dst_image;
    cv::resize(src_image,dst_image,cv::Size(real_width, real_height), 0, 0,cv::INTER_LINEAR);


    memcpy(real_mask.get(),dst_image.data,real_width*real_height*sizeof(int8_t));
    result.results_mask->seg_mask=std::move(real_mask);
/*************************************************************/

#else

/*   方案2 先逐张放缩  在进行整体合并为一张图     */  
//效果经测试比上述效果好
//每张图进行逐行放缩
   xxx.tik();
 auto all_mask=std::unique_ptr<float[]>(new float[last_count * letter_box.src_w * letter_box.src_h]); 
  xxx.tok();
xxx.print_time("all_mask ");
//    xxx.tik();
//  resize_by_opencv_fp(mask_matrix_mult_result,last_count,_proto_width,_proto_height,
//                     all_mask,letter_box); 
//  xxx.tok();
// xxx.print_time("方案2----1");

xxx.tik();
 resize_by_opencv_fp1(mask_matrix_mult_result,last_count,_proto_width,_proto_height,
                    all_mask,letter_box);
 xxx.tok();
xxx.print_time("方案2----1");


//  xxx.tik();
// //整体掩码合并
//  auto all_mask_in_one=std::make_unique<uint8_t[]>(letter_box.src_w*letter_box.src_h);  //这个会自动帮清零，会有开销，其他的地方要注意
//  conbine_mak2(all_mask,all_mask_in_one,result,letter_box);
//  result.results_mask->seg_mask=std::move(all_mask_in_one);
//  xxx.tok();
// xxx.print_time("方案2----2");


 xxx.tik();
 //小优化不合并整个掩码直接每个单独保存就行
conbine_mak22(all_mask,result,letter_box);

 xxx.tok();
xxx.print_time("方案2----3");

/**************************************** */
#endif


return 0;
 }
 


rknn_context* yolov8seg::get_rknn_context()
  {
    return &_ctx;
  }