#include "yolov8seg.hpp"
#include "common.hpp"
#include <memory>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <cstring>


static std::unique_ptr<char[]> read_data_from_file(const std::string &path, int &len)
{
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        LOG_ERROR("fopen model file error, %s", path.c_str());
        len = -1;
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

static bool tensor_spatial_size(const rknn_tensor_attr &attr,
                                int &width,
                                int &height)
{
    if (attr.n_dims < 4) return false;
    if (attr.fmt == RKNN_TENSOR_NCHW)
    {
        height = attr.dims[2];
        width = attr.dims[3];
        return true;
    }
    if (attr.fmt == RKNN_TENSOR_NHWC)
    {
        height = attr.dims[1];
        width = attr.dims[2];
        return true;
    }
    return false;
}

static int tensor_channel_count(const rknn_tensor_attr &attr)
{
    if (attr.n_dims < 4) return 0;
    if (attr.fmt == RKNN_TENSOR_NCHW) return attr.dims[1];
    if (attr.fmt == RKNN_TENSOR_NHWC) return attr.dims[3];
    return 0;
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
    if (io_number.n_input != 1)
    {
        LOG_ERROR("only one image input tensor is supported, actual=%d",
                  io_number.n_input);
        return -1;
    }

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
    // 当前 RKNN 导出拓扑为：
    //   3 × [box, class, class_sum, mask_coeff] + 1 × Proto。
    // 输入分辨率可以是640、1024或其他合法尺寸，但输出排列必须保持该结构。
    if (io_number.n_output < 5 || (io_number.n_output - 1) % 4 != 0)
    {
        LOG_ERROR("unsupported YOLO-seg output topology, output count=%d",
                  io_number.n_output);
        return -1;
    }
    const int branch_count = (io_number.n_output - 1) / 4;
    if (branch_count != 3)
    {
        LOG_ERROR("expected 3 detection branches, actual=%d", branch_count);
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


  if (input_tensor[0].n_dims < 4)
  {
    LOG_ERROR("model input tensor must have 4 dimensions, actual=%d",
              input_tensor[0].n_dims);
    return -1;
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
    LOG_ERROR("unsupported model input tensor format");
    return -1;
  }
  if (_model_width <= 0 || _model_height <= 0 || _model_channel != 3)
  {
    LOG_ERROR("invalid model input shape: %dx%dx%d, expected 3 channels",
              _model_width, _model_height, _model_channel);
    return -1;
  }

  _proto_output_index = io_number.n_output - 1;
  if (output_tensor[_proto_output_index].n_dims < 4)
  {
    LOG_ERROR("Proto tensor must have 4 dimensions, actual=%d",
              output_tensor[_proto_output_index].n_dims);
    return -1;
  }
  if (output_tensor[_proto_output_index].fmt == RKNN_TENSOR_NCHW)
  {
    std::cout<<"model proto is NCHW fmt"<<std::endl;
    _proto_channel = output_tensor[_proto_output_index].dims[1];
    _proto_height = output_tensor[_proto_output_index].dims[2];
    _proto_width = output_tensor[_proto_output_index].dims[3];
  }
  else if(output_tensor[_proto_output_index].fmt == RKNN_TENSOR_NHWC)
  {
   std::cout<<"model proto is NHWC fmt"<<std::endl;
    _proto_height = output_tensor[_proto_output_index].dims[1];
    _proto_width = output_tensor[_proto_output_index].dims[2];
    _proto_channel = output_tensor[_proto_output_index].dims[3];
  }
  else
  {
    LOG_ERROR("unsupported Proto tensor format at output %d",
              _proto_output_index);
    return -1;
  }

  _branch_output_indices.clear();
  _model_spec.detection_grids.clear();
  for (int branch = 0; branch < branch_count; ++branch)
  {
    const int output_index = branch * 4;
    int grid_width = 0;
    int grid_height = 0;
    if (!tensor_spatial_size(output_tensor[output_index],
                             grid_width, grid_height) ||
        grid_width <= 0 || grid_height <= 0)
    {
      LOG_ERROR("invalid detection grid at output %d", output_index);
      return -1;
    }
    for (int member = 1; member < 4; ++member)
    {
      int member_width = 0;
      int member_height = 0;
      if (!tensor_spatial_size(output_tensor[output_index + member],
                               member_width, member_height) ||
          member_width != grid_width || member_height != grid_height)
      {
        LOG_ERROR("branch %d tensor %d grid mismatch", branch,
                  output_index + member);
        return -1;
      }
    }
    const int box_channels =
        tensor_channel_count(output_tensor[output_index]);
    const int mask_coefficient_channels =
        tensor_channel_count(output_tensor[output_index + 3]);
    if (box_channels <= 0 || box_channels % 4 != 0 ||
        mask_coefficient_channels != _proto_channel)
    {
      LOG_ERROR("branch %d channel mismatch: box=%d mask_coeff=%d proto=%d",
                branch, box_channels, mask_coefficient_channels,
                _proto_channel);
      return -1;
    }
    if (_model_width % grid_width != 0 ||
        _model_height % grid_height != 0)
    {
      LOG_ERROR("model input %dx%d is not divisible by grid %dx%d",
                _model_width, _model_height, grid_width, grid_height);
      return -1;
    }
    _branch_output_indices.push_back(output_index);
    _model_spec.detection_grids.emplace_back(grid_width, grid_height);
  }

  // 当前 INT8 快速后处理直接读取 RKNN 的原始量化输出。若某个输出不是
  // affine-asymmetric INT8，继续 reinterpret_cast 会得到没有意义的数据，
  // 因此在启动阶段立即拒绝不兼容模型，而不是运行时静默输出错误结果。
  if (_is_quant)
  {
    for (int index = 0; index < io_number.n_output; ++index)
    {
      if (output_tensor[index].type != RKNN_TENSOR_INT8 ||
          output_tensor[index].qnt_type !=
              RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
      {
        LOG_ERROR("quantized output %d must be affine INT8", index);
        return -1;
      }
    }
  }

  _model_spec.input_width = _model_width;
  _model_spec.input_height = _model_height;
  _model_spec.input_channels = _model_channel;
  _model_spec.proto_width = _proto_width;
  _model_spec.proto_height = _proto_height;
  _model_spec.proto_channels = _proto_channel;
  _model_spec.proto_output_index = _proto_output_index;


  _output_tensor=std::move(output_tensor);
  _input_tensor=std::move(input_tensor);

  if (_is_quant)
    process_i8_proto_table_init(_output_tensor, _proto_output_index,
                                _proto_table);

  std::cout << "Model input: " << _model_width << "x" << _model_height
            << "x" << _model_channel << "\nDetection grids:";
  for (size_t index = 0; index < _model_spec.detection_grids.size(); ++index)
  {
    const cv::Size &grid = _model_spec.detection_grids[index];
    std::cout << " " << grid.width << "x" << grid.height
              << "(stride "
              << static_cast<double>(_model_width) / grid.width << "x"
              << static_cast<double>(_model_height) / grid.height << ")";
  }
  std::cout << "\nProto: " << _proto_channel << "x"
            << _proto_width << "x" << _proto_height << std::endl;


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
    const int expected_size = _model_width * _model_height * _model_channel;
    if (size != expected_size)
    {
      LOG_ERROR("input byte size mismatch: actual=%d expected=%d (%dx%dx%d)",
                size, expected_size, _model_width, _model_height,
                _model_channel);
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
    if (letter_box.dst_w != _model_width ||
        letter_box.dst_h != _model_height)
    {
      LOG_ERROR("letterbox/model size mismatch: letterbox=%dx%d model=%dx%d",
                letter_box.dst_w, letter_box.dst_h,
                _model_width, _model_height);
      return -1;
    }
    if (letter_box.src_w <= 0 || letter_box.src_h <= 0 ||
        letter_box.scale <= 0.0)
    {
      LOG_ERROR("invalid letterbox metadata");
      return -1;
    }

    std::vector<float> candidate_box;  //保存候选框  四个一组  x,y,w,h  
    std::vector<float> box_score;     //每个候选框的分类置信度
    std::vector<int> class_id;        //每个候选框的id

    std::vector< rknpu2::float16> box_mask_coefficient;  //每个候选框对应的mask 系数（长度 PROTO_CHANNEL）
    auto proto=std::unique_ptr< rknpu2::float16[]>( new rknpu2::float16[_proto_channel * _proto_width * _proto_height]); //Proto 原型掩码（大小 C*Hp*Wp），只会在处理到 proto 输出的那一次被填充。
    std::vector< rknpu2::float16> filter_box_mask_coefficient;  //经过 NMS 筛选剩下的那些候选的系数，用于最终的 matmul也就是生成掩码
   
    int valid_count=0;



    xxx.tik();
    // Proto 与检测分支分开处理，避免遍历辅助张量时误把通道维当作网格。
    if (_is_quant)
      process_i8(_output, _output_tensor, _proto_output_index,
                  _proto_width, _proto_height, 0.0f, 0.0f,
                  candidate_box, box_score, class_id, proto,
                  box_mask_coefficient, _proto_channel, _proto_width,
                  _proto_height, BOX_THRESH, _proto_table,
                  _proto_output_index);
    else
      process_fp32(_output, _output_tensor, _proto_output_index,
                   _proto_width, _proto_height, 0.0f, 0.0f,
                   candidate_box, box_score, class_id, proto,
                   box_mask_coefficient, _proto_channel, _proto_width,
                   _proto_height, BOX_THRESH, _proto_output_index);

    for (const int output_index : _branch_output_indices)
    {
      int grid_width = 0;
      int grid_height = 0;
      if (!tensor_spatial_size(_output_tensor[output_index],
                               grid_width, grid_height))
      {
        LOG_ERROR("invalid detection tensor format at output %d",
                  output_index);
        return -1;
      }
      const float stride_x =
          static_cast<float>(_model_width) / grid_width;
      const float stride_y =
          static_cast<float>(_model_height) / grid_height;
      if (_is_quant)
        valid_count += process_i8(
            _output, _output_tensor, output_index, grid_width, grid_height,
            stride_x, stride_y, candidate_box, box_score, class_id, proto,
            box_mask_coefficient, _proto_channel, _proto_width,
            _proto_height, BOX_THRESH, _proto_table,
            _proto_output_index);
      else
        valid_count += process_fp32(
            _output, _output_tensor, output_index, grid_width, grid_height,
            stride_x, stride_y, candidate_box, box_score, class_id, proto,
            box_mask_coefficient, _proto_channel, _proto_width,
            _proto_height, BOX_THRESH, _proto_output_index);
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
        filter_candidate_box.push_back(y2); // 暂存模型输入坐标，后面统一反变换到原图。

        // 检测框位于模型输入坐标系，Mask裁剪位于Proto坐标系。
        // 比例由真实张量尺寸决定，不再假设固定 640/160=4。
        const float proto_scale_x =
            static_cast<float>(_proto_width) / _model_width;
        const float proto_scale_y =
            static_cast<float>(_proto_height) / _model_height;
        filter_candidate_box_mask_conbine.push_back(x1 * proto_scale_x);
        filter_candidate_box_mask_conbine.push_back(y1 * proto_scale_y);
        filter_candidate_box_mask_conbine.push_back(x2 * proto_scale_x);
        filter_candidate_box_mask_conbine.push_back(y2 * proto_scale_y);
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
 //把所有掩码合成写到Proto特征图上
 auto all_mask_in_one=std::make_unique<int8_t[]>(_proto_height*_proto_width);
 memset(all_mask_in_one.get(),0,sizeof(int8_t)*_proto_height*_proto_width);
 conbine_mak(mask_matrix_mult_result,all_mask_in_one,filter_candidate_box_mask_conbine,mask_classid,last_count,_proto_width,_proto_height);


 //得到真实mask,处理为原图尺寸
    const double proto_scale_x =
        static_cast<double>(_proto_width) / letter_box.dst_w;
    const double proto_scale_y =
        static_cast<double>(_proto_height) / letter_box.dst_h;
    int tem_leftx=std::lround(letter_box.upleft_pad_x*proto_scale_x);
    int tem_rightx=std::lround(letter_box.lowright_pad_x*proto_scale_x);
    int tem_lefty=std::lround(letter_box.upleft_pad_y*proto_scale_y);
    int tem_righty=std::lround(letter_box.lowright_pad_y*proto_scale_y);
    std::cout<<tem_leftx<<" "<<tem_rightx<<" "<<tem_lefty<<" "<<tem_righty<<std::endl;

    int padx= tem_leftx+tem_rightx;
    int pady= tem_lefty+tem_righty;
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
