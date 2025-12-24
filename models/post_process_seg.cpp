#include "post_process_seg.hpp"




inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static  float deqnt_affine_to_f32(int8_t proto, int32_t zp,  float scale) // 这种处理方式是唯一解
{
  return  (proto -  zp) * scale;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static void compute_dfl(std::vector<float> &before_dfl, const int dfl_len, float box[])
{
  for (int b = 0; b < 4; b++)
  {
    float exp_t[dfl_len];
    float exp_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i < dfl_len; i++)
    {
      exp_t[i] = exp(before_dfl[i + b * dfl_len]); // 这里是做softmax 来求每个概率  所以要做e的函数
      exp_sum += exp_t[i];
    }
  
    for (int i = 0; i < dfl_len; i++)
    {
      acc_sum += exp_t[i] / exp_sum * i; // 一直累加下去得到最终的预测值  也就是边框需要的那四个值；ltrb（left、top、right、bottom）  也就是从网格中心到左上右下的格数
    }
    box[b] = acc_sum;
  }
}





int process_i8(std::unique_ptr<rknn_output[]> &output, std::unique_ptr<rknn_tensor_attr[]> &output_tensor, int index, // 输出信息  和  索引
               int grid_w, int grid_h, int model_w, int model_h, int stride,                                          // 格子大小（输出的大小）  ;  模型输入的大小  stride（步长/下采样倍数） ——也就是该输出特征图上 1 个网格格子对应输入图像上多少个像素。
               std::vector<float> &candidate_box, std::vector<float> &box_score, std::vector<int> &class_id,          // 候选框  置信度  类别
               std::unique_ptr<rknpu2::float16[]> &proto, std::vector<rknpu2::float16> &box_mask_coefficient,
               int proto_channel, int proto_width, int proto_height, // 掩码系数部分
               float box_threshold)                                  // NMS阈值
{

  TIMER xx;

  // Skip if input_id is not 0, 4, 8, or 12
  if (index % 4 != 0)
  {
    return 0;
  }

  int volid_count = 0;
  int grid_len = grid_w * grid_h;


  if (index == 12)
  {
      xx.tik();
    /**
     * 获取第12层的输出数据，然后把量化后的数据还原到为原来的浮点型
     * 需要注意的是，这里没有使用比例关系，因为程序需要是INT的数据，不需要0~1的float数据
     */
    int8_t *input_proto = (int8_t *)output[index].buf;
    int32_t zp_proto = output_tensor[index].zp;
    float scale_proto = output_tensor[index].scale; //这两值是做反量化的
    
    //因为input_proto就-128到127种可能，直接把结果全算出来，然后查表找答案就行
    rknpu2::float16 table[256];
     for (int i = -128; i < 128; ++i) 
     {
        float f = deqnt_affine_to_f32(i, zp_proto, scale_proto);
        table[i+128] = static_cast<rknpu2::float16>(f);
    }

    // #pragma omp parallel for schedule(static)   //把下面这个 for 循环 分给多个 CPU 线程同时跑 (提升效果不明显)
    for (int i = 0; i < proto_channel * proto_width * proto_height; i++)
    {
      // proto[i] = (rknpu2::float16)deqnt_affine_to_f32(input_proto[i], zp_proto, scale_proto); // 反量化_仿射
        proto[i] = table[(input_proto[i])+128]; // 反量化_查表  做这个量化主要原因就是，这里循环太大了，每个循环提升一点，累计下来都很大
    }

    xx.tok();
xx.print_time("index == 12");
    return 0;
  }

  // 预测框部分  解码xywh
  int8_t *box_temsor = (int8_t *)output[index].buf;
  int32_t box_zp = output_tensor[index].zp;
  float box_scale = output_tensor[index].scale;

  // 分类分支，计算分类概率
  int8_t *score_tensor = (int8_t *)output[index + 1].buf;
  int32_t score_zp = output_tensor[index + 1].zp;
  float score_scale = output_tensor[index + 1].scale;

  // 分类归一化辅助  ReduceSum 辅助 tensor（为 softmax / exp 用） ； 作用：对第 2 个 tensor（80 个类别的分支）在 “类别维度” 上做 sum
  // 存的是当前网格 (i,j) 上，所有类别“整体的概率/激活强度” 的一个总和。
  int8_t *score_sum_tensor = (int8_t *)output[index + 2].buf;
  int32_t score_sum_zp = output_tensor[index + 2].zp;
  float score_sum_scale = output_tensor[index + 2].scale;

  // seg32通道mask特征，拼出最终instance mask   实例分割掩码特征
  int8_t *seg_tensor = (int8_t *)output[index + 3].buf;
  int32_t seg_zp = output_tensor[index + 3].zp;
  float seg_scale = output_tensor[index + 3].scale;

  int8_t score_threshold_u8 = qnt_f32_to_affine(box_threshold, score_zp, score_scale);             // 这里量化把 阈值（float → 量化值）后续在遍历 output 的时候，很多地方直接拿 量化后的 int8 跟阈值比较，就不需要每个元素都先反量化再比，节省计算：
  int8_t score_sum_threshold_u8 = qnt_f32_to_affine(box_threshold, score_sum_zp, score_sum_scale); // 快速过滤的一个前置值


  for (int j = 0; j < grid_h; ++j)
  {
     for (int i = 0; i < grid_w; ++i)
    {
      int offset = j * grid_w + i;
      int offset_seg = j * grid_w + i; // 这里存一下位置 给proto使用

      // for quick filtering through "score sum"
      if (score_sum_tensor)
      {
        // 如果得分总和(所有识别到的物体的置信度只和)少于设定阈值，直接放弃本次的目标
        if (score_sum_tensor[offset] < score_sum_threshold_u8)
          continue;
      }

      // 这里确定第一个检测框，属于是哪一个类别id；
      // 在第二个输出里面的一竖条找置信度最大的那个类别
      int8_t max_score = -128; // 这里给一个最小的初值保证 合法性
      int max_class_id = -1;
      int tem_class_number = 0;
      if (output_tensor[index + 1].fmt == RKNN_TENSOR_NCHW)
      {
        tem_class_number = output_tensor[index + 1].dims[1];
      }
      else if (output_tensor[index + 1].fmt == RKNN_TENSOR_NHWC)
      {
        tem_class_number = output_tensor[index + 1].dims[3];
      }


      for (int c = 0; c < tem_class_number; ++c)
      {
        if (score_tensor[offset] > score_threshold_u8 && score_tensor[offset] > max_score)
        {
          max_score = score_tensor[offset];
          max_class_id = c;
        }
        offset += grid_len;
      }
 

      if (max_class_id != -1 && max_score > score_threshold_u8) // 满足这两个说明这个检测框是有效的  则进行检测框的解码  以及proto的系数提取
      {

        // proto系数的提取
        int8_t *tem_seg_tensor = seg_tensor + offset_seg;
        for (int k = 0; k < proto_channel; ++k)
        {
          rknpu2::float16 tem_proto_coefficient = (rknpu2::float16)deqnt_affine_to_f32(tem_seg_tensor[k * grid_len], seg_zp, seg_scale);
          box_mask_coefficient.push_back(tem_proto_coefficient);
        }

        // 检测框解码
        //  DFL（Distribution Focal Loss）把 box 分支的输出解码成真正的 bbox，再从特征图坐标转成输入图像坐标 的核心代码
        float box[4];
        int dfl_number;
        if (output_tensor[index].fmt == RKNN_TENSOR_NCHW)
        {
          dfl_number = output_tensor[index].dims[1];
        }
        else if (output_tensor[index].fmt == RKNN_TENSOR_NHWC)
        {
          dfl_number = output_tensor[index].dims[3];
        }
        std::vector<float> before_dfl;

        offset = j * grid_w + i;
        for (int k = 0; k < dfl_number; ++k)
        {
          before_dfl.push_back(deqnt_affine_to_f32(box_temsor[offset + k * grid_len], box_zp, box_scale));
         
          // std::cout<<before_dfl[before_dfl.size()-1]<<std::endl;
        }
        int dfl_len = dfl_number / 4;
        compute_dfl(before_dfl, dfl_len, box); // ltrb 它们都是“以当前网格(i,j对应的那个网格)的中心为原点，向四个方向延伸多少格”。
                                               // left  top  right  below
        float x1, y1, x2, y2, w, h;
        x1 = (i + 0.5 - box[0]) * stride; // stride 用来网格数转像素坐标的
        y1 = (j + 0.5 - box[1]) * stride; //+0.5是要走到网格的中心
        x2 = (i + 0.5 + box[2]) * stride;
        y2 = (j + 0.5 + box[3]) * stride;
     
        w = x2 - x1;
        h = y2 - y1;
       

        candidate_box.push_back(x1);
        candidate_box.push_back(y1);
        candidate_box.push_back(w);
        candidate_box.push_back(h);
        // std::cout<<candidate_box[4*volid_count]<<" "<<candidate_box[4*volid_count+1]<<" "<<candidate_box[4*volid_count+2]<<" "<<candidate_box[4*volid_count+3]<<std::endl;
        // std::cout<<deqnt_affine_to_f32(max_score, score_zp, score_scale)<<std::endl;
        box_score.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
        // std::cout<<box_score[box_score.size()-1]<<std::endl;
        class_id.push_back(max_class_id);
        // std::cout<<class_id[class_id.size()-1]<<std::endl;

        volid_count++;
      }
    }
  }
  return volid_count;
}

int process_fp32(std::unique_ptr<rknn_output[]> &output, std::unique_ptr<rknn_tensor_attr[]> &output_tensor, int index,
                 int grid_w, int grid_h, int model_w, int model_h, int stride,
                 std::vector<float> &candidate_box, std::vector<float> &box_score, std::vector<int> &class_id,
                 std::unique_ptr<rknpu2::float16[]> &proto, std::vector<rknpu2::float16> &box_mask_coefficient,
                 int proto_channel, int proto_width, int proto_height,
                 float box_threshold)
{
  //Skip if input_id is not 0, 4, 8, or 12
  if (index % 4 != 0)
  {
    return 0;
  }

  int volid_count = 0;
  int grid_len = grid_w * grid_h;

  if (index == 12)
  {
    float *input_proto = (float *)output[index].buf;
    for (int i = 0; i < proto_channel * proto_width * proto_height; i++)
    {
      proto[i] = (rknpu2::float16)input_proto[i];
    }
    return 0;
  }

  // 预测框部分  解码xywh
  float *box_temsor = (float *)output[index].buf;

  // 分类分支，计算分类概率
  float *score_tensor = (float *)output[index + 1].buf;

  // 分类归一化辅助  ReduceSum 辅助 tensor（为 softmax / exp 用） ； 作用：对第 2 个 tensor（80 个类别的分支）在 “类别维度” 上做 sum
  // 存的是当前网格 (i,j) 上，所有类别“整体的概率/激活强度” 的一个总和。
  float *score_sum_tensor = (float *)output[index + 2].buf;

  // seg32通道mask特征，拼出最终instance mask   实例分割掩码特征
  float *seg_tensor = (float *)output[index + 3].buf;

  float score_threshold_u8 = box_threshold;     // 这里量化把 阈值（float → 量化值）后续在遍历 output 的时候，很多地方直接拿 量化后的 int8 跟阈值比较，就不需要每个元素都先反量化再比，节省计算：
  float score_sum_threshold_u8 = box_threshold; // 快速过滤的一个前置值

  for (int j = 0; j < grid_h; ++j)
  {
    for (int i = 0; i < grid_w; ++i)
    {
      int offset = j * grid_w + i;
      int offset_seg = offset; // 这里存一下位置 给proto使用

      // for quick filtering through "score sum"
      if (score_sum_tensor)
      {
        // 如果得分总和(所有识别到的物体的置信度只和)少于设定阈值，直接放弃本次的目标
        if (score_sum_tensor[offset] < score_sum_threshold_u8)
          continue;
      }

      // 这里确定第一个检测框，属于是哪一个类别id；
      // 在第二个输出里面的一竖条找置信度最大的那个类别
      float max_score = -99999; // 这里给一个最小的初值保证 合法性
      int max_class_id = -1;
      int tem_class_number = 0;
      if (output_tensor[index + 1].fmt == RKNN_TENSOR_NCHW)
      {
        tem_class_number = output_tensor[index + 1].dims[1];
      }
      else if (output_tensor[index + 1].fmt == RKNN_TENSOR_NHWC)
      {
        tem_class_number = output_tensor[index + 1].dims[3];
      }

      for (int c = 0; c < tem_class_number; ++c)
      {
        if (score_tensor[offset] > score_threshold_u8 && score_tensor[offset] > max_score)
        {
          max_score = score_tensor[offset];
          max_class_id = c;
        }
        offset += grid_len;
      }

      if (max_class_id != -1 && max_score > score_threshold_u8) // 满足这两个说明这个检测框是有效的  则进行检测框的解码  以及proto的系数提取
      {

        // proto系数的提取
        float *tem_seg_tensor = seg_tensor + offset_seg;
        for (int k = 0; k < proto_channel; ++k)
        {
          rknpu2::float16 tem_proto_coefficient = (rknpu2::float16)tem_seg_tensor[k * grid_len];
          box_mask_coefficient.push_back(tem_proto_coefficient);
        }

        // 检测框解码
        //  DFL（Distribution Focal Loss）把 box 分支的输出解码成真正的 bbox，再从特征图坐标转成输入图像坐标 的核心代码
        float box[4];
        int dfl_number;
        if (output_tensor[index].fmt == RKNN_TENSOR_NCHW)
        {
          dfl_number = output_tensor[index].dims[1];
        }
        else if (output_tensor[index].fmt == RKNN_TENSOR_NHWC)
        {
          dfl_number = output_tensor[index].dims[3];
        }
        std::vector<float> before_dfl;

        offset = j * grid_w + i;
        for (int k = 0; k < dfl_number; ++k)
        {
          before_dfl.push_back(box_temsor[offset + k * grid_len]);
        }
        int dfl_len = dfl_number / 4;
        compute_dfl(before_dfl, dfl_len, box); // ltrb 它们都是“以当前网格(i,j对应的那个网格)的中心为原点，向四个方向延伸多少格”。
                                               // left  top  right  below
        float x1, y1, x2, y2, w, h;
        x1 = (i + 0.5 - box[0]) * stride; // stride 用来网格数转像素坐标的
        y1 = (j + 0.5 - box[1]) * stride; //+0.5是要走到网格的中心
        x2 = (i + 0.5 + box[2]) * stride;
        y2 = (j + 0.5 + box[3]) * stride;
        w = x2 - x1;
        h = y2 - y1;
        candidate_box.push_back(x1);
        candidate_box.push_back(y1);
        candidate_box.push_back(w);
        candidate_box.push_back(h);
        box_score.push_back(max_score);
        class_id.push_back(max_class_id);
        volid_count++;
      }
    }
  }
  return volid_count;
}




void quick_sort_desend_order( std::vector<float>& box_score,int left,int right, std::vector<int>& index_flag)
{
  if(left>=right)  return ;
  float x=box_score[left];
  int i=left-1, j=right+1;
  while(i<j)
  {
    do{i++;}while(box_score[i]>x);
    do{j--;}while(box_score[j]<x);
     if(i<j) 
     {
      swap(box_score[i],box_score[j]);
      swap(index_flag[i],index_flag[j]);
     }
  }
 quick_sort_desend_order(box_score,left,j,index_flag);
 quick_sort_desend_order(box_score,j+1,right,index_flag);
}



static float CalculateOverlap(float x1,float y1, float x2,float y2 , float xx1, float yy1 , float xx2, float yy2)
{
    // 交集矩形的边界
    const float inter_left   = std::max(x1,  xx1);
    const float inter_top    = std::max(y1,  yy1);
    const float inter_right  = std::min(x2,  xx2);
    const float inter_bottom = std::min(y2,  yy2);

    // 没交集
    const float inter_w = inter_right  - inter_left;
    const float inter_h = inter_bottom - inter_top;
    if (inter_w <= 0.f || inter_h <= 0.f)
        return 0.f;

    const float inter_area = inter_w * inter_h;
    const float area1 = std::max(0.f, x2  - x1)  * std::max(0.f, y2  - y1);
    const float area2 = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);

    const float union_area = area1 + area2 - inter_area;
    return union_area > 0.f ? inter_area / union_area : 0.f;
}


void nms(int valid_count,std::vector<int>& index_flag,std::vector<float>& candidate_box,std::vector<int>& class_id,int c,float nms_thresh)
{
  for(int i=0; i<valid_count; ++i)
  {
    if(index_flag[i]==-1 || class_id[index_flag[i]]!=c)
      continue;

    int n=index_flag[i];
    float x1=candidate_box[n*4];
    float y1=candidate_box[n*4+1];
    float x2=candidate_box[n*4+2]+x1;
    float y2=candidate_box[n*4+3]+y1;
    for(int j=i+1;j<valid_count;++j)   // 满足条件的情况下  分高消分低
    {
      if(index_flag[j]==-1 || class_id[index_flag[j]]!=c)
      continue;
    int m=index_flag[j];
    float xx1=candidate_box[m*4];
    float yy1=candidate_box[m*4+1];
    float xx2=candidate_box[m*4+2]+xx1;
    float yy2=candidate_box[m*4+3]+yy1;
     if(CalculateOverlap(x1,y1,x2,y2,xx1,yy1,xx2,yy2)>nms_thresh){
          index_flag[j]=-1;
        // std::cout<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<"消去了"<<xx1<<" "<<yy1<<" "<<xx2<<" "<<yy2<<"重合比例为："<<CalculateOverlap(x1,y1,x2,y2,xx1,yy1,xx2,yy2)<<std::endl;

    }
  }
  }
  return ;
}



static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

 std::pair<int,int> box_reverse(letterbox& letter_box , float x,float y) 
 {
    int paddingx=letter_box.upleft_pad_x;
    int paddingy=letter_box.upleft_pad_y;
    float scale=letter_box.scale;
    float a=(x-paddingx)/scale;
    float b=(y-paddingy)/scale;
    int xx,yy;
    xx=(int)clamp(a,0,letter_box.src_w);
    yy=(int)clamp(b,0,letter_box.src_h);
    return std::move(std::make_pair(xx,yy));
 }





int matrix_mult_by_npu_fp32(std::vector<rknpu2::float16>& filter_box_mask_coefficient,std::unique_ptr<rknpu2::float16[]>& proto,std::unique_ptr<float[]>& matrix_mult_result,int ROWS_A,int COLS_A,int COLS_B )
{

TIMER xxx;
xxx.tik();
//1.初始化矩阵乘法的上下文
rknn_matmul_info info;  //传入矩阵的相关信息
memset(&info, 0, sizeof(rknn_matmul_info));
info.M=ROWS_A;
info.K=COLS_A;
info.N=COLS_B;
info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
info.B_layout=RKNN_MM_LAYOUT_NORM;
info.AC_layout=RKNN_MM_LAYOUT_NORM;


rknn_matmul_io_attr io_attr;
memset(&io_attr,0,sizeof(rknn_matmul_io_attr));

rknn_matmul_ctx ctx;
int err=rknn_matmul_create(&ctx,&info,&io_attr);  //单独建一个 matmul 专用的 ctx 变量
if(err!=RKNN_SUCC)
{
  LOG_ERROR("rknn_matmul_create fail,errno:%d",err);
  return err;
}

 xxx.tok();
  xxx.print_time("init---1");

xxx.tik();
//2.创建ABC三个矩阵的NPU内存
rknn_tensor_mem* A=rknn_create_mem(ctx,io_attr.A.size);
if(A==NULL)
{
   LOG_ERROR("rknn_create_mem A fail");
  return -1;
}
rknn_tensor_mem* B=rknn_create_mem(ctx,io_attr.B.size);
if(B==NULL)
{
   LOG_ERROR("rknn_create_mem B fail");
   rknn_destroy_mem(ctx,A);
  return -1;
}
rknn_tensor_mem* C=rknn_create_mem(ctx,io_attr.C.size);
if(C==NULL)
{
   LOG_ERROR("rknn_create_mem C fail");
   rknn_destroy_mem(ctx,A);
   rknn_destroy_mem(ctx,B);
  return -1;
}

 xxx.tok();
  xxx.print_time("init---2");

    //3.把输入float32 -> float16，填到 A/B buffer
  rknpu2::float16* A16 = (rknpu2::float16*)A->virt_addr;
  rknpu2::float16* B16 = (rknpu2::float16*)B->virt_addr;


  xxx.tik();
  //设置AB数据
  std::memcpy(A16, &(filter_box_mask_coefficient[0]), io_attr.A.size);
  std::memcpy(B16, &(proto[0]), io_attr.B.size);

    // for (int i = 0; i < ROWS_A * COLS_A; ++i) {
    //     A16[i] = filter_box_mask_coefficient[i];
    // }
    // for (int i = 0; i < COLS_A * COLS_B; ++i) {
    //     B16[i] = proto[i];
    // }

      // 4. 绑定 IO
    rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
    rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
    rknn_matmul_set_io_mem(ctx, C, &io_attr.C);

  xxx.tok();
  xxx.print_time("init---3");



   // 5. 运行 matmul
    err = rknn_matmul_run(ctx);
    if (err != RKNN_SUCC) {
        LOG_ERROR("rknn_matmul_run fail, errno:%d", err);
    if (A) rknn_destroy_mem(ctx, A);
    if (B) rknn_destroy_mem(ctx, B);
    if (C) rknn_destroy_mem(ctx, C);
       return err;
    }



   
    // 6. 拷贝 float32 输出到C
    float* C32 = (float*)C->virt_addr;
    std::memcpy(matrix_mult_result.get(), C32, sizeof(float)*ROWS_A*COLS_B);


    if (A) rknn_destroy_mem(ctx, A);
    if (B) rknn_destroy_mem(ctx, B);
    if (C) rknn_destroy_mem(ctx, C);

  rknn_destroy(ctx);

    return 0;
}

void matrix_mult_by_cpu_fp32(std::vector<rknpu2::float16>& A,std::unique_ptr<rknpu2::float16[]>& B,std::unique_ptr<float[]>& C, int ROWS_A, int COLS_A, int COLS_B)
{
    float temp = 0;
    for (int i = 0; i < ROWS_A; i++)
    {
        for (int j = 0; j < COLS_B; j++)
        {
            temp = 0;
            for (int k = 0; k < COLS_A; k++)
            {
                temp += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
            C[i * COLS_B + j] = temp;
        }
    }
}



 void conbine_mak(std::unique_ptr<float[]>& mask_matrix_mult_result,std::unique_ptr<int8_t[]>& all_mask_in_one,std::vector<float>& filter_candidate_box_mask_conbine,std::vector<int>& mask_classid,int last_count,int proto_width,int proto_height)
{
int len=proto_height*proto_width;
for(int i=0; i<last_count ; ++i)
{
float x1=filter_candidate_box_mask_conbine[4*i];
float y1=filter_candidate_box_mask_conbine[4*i+1];
float x2=filter_candidate_box_mask_conbine[4*i+2];
float y2=filter_candidate_box_mask_conbine[4*i+3];

for(int y=0; y<proto_height; ++y)
{
for(int x=0; x<proto_width; ++x)
{
  if(all_mask_in_one[y*proto_width+x]==0)
  if(x >= x1 && x < x2 && y >= y1 && y < y2 ) //掩码在检测框里面
  if(mask_matrix_mult_result[i*len+y*proto_width+x]>0)
    all_mask_in_one[y*proto_width+x]=mask_classid[i]+1;
}
}
}
}  

void conbine_mak2(std::unique_ptr<float[]>& all_mask,std::unique_ptr<uint8_t[]>& all_mask_in_one, object_detect_result_list& result ,letterbox& letter_box)
{
int real_x=letter_box.src_w;
int real_y=letter_box.src_h;
int len=real_x*real_y;
int n=result.count;

result.results_mask->each_of_mask.resize(n+3);
for(int i=0; i<n+3;++i)
result.results_mask->each_of_mask[i]=nullptr;

for(int i=0; i<n; ++i)
{
int x1=result.results_box[i].x;
int y1=result.results_box[i].y;
int x2=result.results_box[i].w+x1;
int y2=result.results_box[i].h+y1;
if (x2 <= x1 || y2 <= y1) continue;

auto tem_mask=std::make_unique<uint8_t[]>(real_x*real_y);

for(int y=y1; y<y2; ++y)
{
for(int x=x1; x<x2; ++x)
{
if(all_mask[i*len+y*real_x+x]>0)
 tem_mask[y*real_x+x]=result.results_box[i].cls_id+1;   //记录每一个单独的掩码

  if(all_mask_in_one[y*real_x+x]==0)
  if(all_mask[i*len+y*real_x+x]>0)
    all_mask_in_one[y*real_x+x]=result.results_box[i].cls_id+1;
}
}
result.results_mask->each_of_mask[result.results_box[i].cls_id]=std::move(tem_mask);  //因为我们最多预测出来三个种类嘛
}
}








void resize_by_opencv_fp(std::unique_ptr<float[]>& mask_matrix_mult_result,int last_count,int proto_width,int proto_height,
                    std::unique_ptr<float[]>& all_mask,letterbox& letter_box)
                    {

 //先把掩码中填充的部分裁掉，在进行放缩

  //1.裁剪
    int tem_leftx=letter_box.upleft_pad_x/4;
    int tem_rightx=letter_box.lowright_pad_x/4;
    int tem_lefty=letter_box.upleft_pad_y/4;
    int tem_righty=letter_box.lowright_pad_y/4;

    int padx= (letter_box.lowright_pad_x+letter_box.upleft_pad_x)/4;
    int pady= (letter_box.lowright_pad_y+letter_box.upleft_pad_y)/4;
    int conbine_width = proto_width - padx;  //掩码160*160裁去填充部分 的宽度
    int conbine_height= proto_height- pady;
    int real_width = letter_box.src_w; //原始输入图像尺寸
    int real_height = letter_box.src_h;

  auto every_mask_crop_pad=std::make_unique<float[]>(last_count*conbine_width*conbine_height);
  int cropped_index=0;
  for(int xx=0; xx<last_count; ++xx)
  {
  for(int i=0; i<proto_height;++i)
  {
  for(int j=0; j<proto_width;++j)
 { 
    if(j >= tem_leftx && j < proto_width-tem_rightx && i >= tem_lefty && i < proto_height - tem_righty)
    every_mask_crop_pad[cropped_index++] = mask_matrix_mult_result[xx*proto_width*proto_height+i*proto_width+j];        //每张mask减去填充部分得到新的mask ， 来进行缩放
 }
 }
} 
    //2.进行逐张放缩
                      for(int i=0; i<last_count; ++i)
                      {
                cv::Mat src_image(conbine_height, conbine_width, CV_32F, &every_mask_crop_pad[i * conbine_height * conbine_width]); //用外部内存包装乘Mat
                cv::Mat dst_image;
                cv::resize(src_image, dst_image, cv::Size(real_width, real_height), 0, 0, cv::INTER_LINEAR);
                memcpy(&all_mask[i * real_width * real_height], dst_image.data, real_width * real_height * sizeof(float));
                      }

                      return ;
 } 









 void resize_by_opencv_fp1(std::unique_ptr<float[]>& mask_matrix_mult_result,int last_count,int proto_width,int proto_height,
                    std::unique_ptr<float[]>& all_mask,letterbox& letter_box)
                    {

 //先把掩码中填充的部分裁掉，在进行放缩

  //1.裁剪
    int tem_leftx=letter_box.upleft_pad_x/4;
    int tem_rightx=letter_box.lowright_pad_x/4;
    int tem_lefty=letter_box.upleft_pad_y/4;
    int tem_righty=letter_box.lowright_pad_y/4;

    int padx= (letter_box.lowright_pad_x+letter_box.upleft_pad_x)/4;
    int pady= (letter_box.lowright_pad_y+letter_box.upleft_pad_y)/4;
    int conbine_width = proto_width - padx;  //掩码160*160裁去填充部分 的宽度
    int conbine_height= proto_height- pady;
    int real_width = letter_box.src_w; //原始输入图像尺寸
    int real_height = letter_box.src_h;




    // OpenCV 的 ROI 用 Rect 表示：x=left, y=top, 宽=crop_w, 高=crop_h
    const cv::Rect roi_rect(tem_leftx, tem_lefty, conbine_width, conbine_height);

    // 输出要放缩到原图大小
    const cv::Size dst_size(letter_box.src_w, letter_box.src_h);

    // 每张输入 mask 的元素个数
    const int proto_stride = proto_width * proto_height;
    // 每张输出 mask 的元素个数
    const int dst_stride   = letter_box.src_w * letter_box.src_h;

    // 取出 unique_ptr 的裸指针，方便做指针偏移
    float* src_ptr = mask_matrix_mult_result.get();
    float* dst_ptr = all_mask.get();


    //进行逐张放缩
                      for(int i=0; i<last_count; ++i)
                      {
                    cv::Mat proto(proto_height, proto_width, CV_32F, (void*)(src_ptr + i * proto_stride));  //原图包装乘一个Mat
                    cv::Mat cropped = proto(roi_rect);  // cropped 也是“视图”，指向 proto 的子区域

                    cv::Mat dst(letter_box.src_h, letter_box.src_w, CV_32F,(void*)(dst_ptr + i * dst_stride)); //让输出 Mat 直接指向 all_mask 对应区域 
 
                    cv::resize(cropped, dst, dst_size, 0, 0, cv::INTER_LINEAR); //// 把 cropped 放缩到 dst_size，并写入 dst（也就是 all_mask 的那段内存）

                      }

                      return ;
 } 













 