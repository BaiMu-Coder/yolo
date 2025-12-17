#include "yolov8seg.hpp"
#include "image_process.hpp"
#include <opencv2/opencv.hpp>


#include <iostream>


//这种写法 是唯一解  已经测试完成     模型输出 都是int8_t 型    zp也是int32_t   都是有符号的 

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}


static float deqnt_affine_to_f32(int8_t proto, int32_t zp, float scale)   //这种处理方式是唯一解
{
  return ((float)proto-(float)zp)*scale ;
}

static int8_t qnt_f32_to_affine(float f32 , int32_t zp , float scale)
{
  float dst_val=(f32/scale)+zp;
  int8_t res=(int8_t)__clip(dst_val,-128,127);
  return res;
}





int main()
{

cv::Mat frame=cv::imread("bus.jpg");
if(frame.empty())
{
    std::cout<<"open jpg error"<<std::endl;
    return -1;
}



image_process ip(frame);
ip.image_preprocessing(640,640);




yolov8seg model("yolov8_seg.rknn");
model.init();
int size;

void* buf=ip.get_image_buffer(&size);


model.set_input_data(buf,size);



model.rknn_model_inference();



model.get_output_data();



std::cout<<model._output[12].size<<std::endl;
std::cout<<model._proto_channel<<std::endl;

std::cout<<model._proto_width<<std::endl;
std::cout<<model._proto_height<<std::endl;


int8_t* input_proto = (int8_t*)model._output[12].buf; 
int32_t zp=model._output_tensor[12].zp;
float scale=model._output_tensor[12].scale;





//  std::cout<<"80*80"<<std::endl;
// input_proto= (float*)model._output[0].buf; 
//   for (int i = 0; i <200; i++) {
//         std::cout<<input_proto[i] <<" ";
//     }
//   std::cout<<std::endl;
//    std::cout<<std::endl;

// input_proto= (float*)model._output[1].buf; 
//      for (int i = 0; i <200; i++) {
//       std::cout<<input_proto[i] <<" ";
//     }
//   std::cout<<std::endl;
//    std::cout<<std::endl;


// input_proto= (float*)model._output[2].buf; 
//      for (int i = 0; i <200; i++) {
//      std::cout<<input_proto[i] <<" ";
//     }
//   std::cout<<std::endl;
//    std::cout<<std::endl;

//   input_proto= (float*)model._output[3].buf; 
//      for (int i = 0; i <200; i++) {
//   std::cout<<input_proto[i] <<" ";
//     }
//   std::cout<<std::endl;
//    std::cout<<std::endl;






    std::cout<<"80*80"<<std::endl;
input_proto= (int8_t*)model._output[0].buf; 
  for (int i = 0; i <200; i++) {
        std::cout<<static_cast<int>(input_proto[i]) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;

input_proto= (int8_t*)model._output[1].buf; 
     for (int i = 0; i <200; i++) {
      std::cout<<static_cast<int>(input_proto[i]) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


input_proto= (int8_t*)model._output[2].buf; 
     for (int i = 0; i <200; i++) {
     std::cout<<static_cast<int>(input_proto[i]) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;

  input_proto= (int8_t*)model._output[3].buf; 
     for (int i = 0; i <200; i++) {
  std::cout<<static_cast<int>(input_proto[i]) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;






    std::cout<<"80*80"<<std::endl;
input_proto= (int8_t*)model._output[0].buf; 
 zp=model._output_tensor[0].zp;
  scale=model._output_tensor[0].scale;
  for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<tem <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;

input_proto= (int8_t*)model._output[1].buf; 
 zp=model._output_tensor[1].zp;
  scale=model._output_tensor[1].scale;
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
       std::cout<<tem <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


input_proto= (int8_t*)model._output[2].buf; 
 zp=model._output_tensor[2].zp;
  scale=model._output_tensor[2].scale;
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<tem <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


   input_proto= (int8_t*)model._output[3].buf; 
    zp=model._output_tensor[3].zp;
  scale=model._output_tensor[3].scale;
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
       std::cout<<tem <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;






 std::cout<<"80*80"<<std::endl;
input_proto= (int8_t*)model._output[0].buf; 
 zp=model._output_tensor[0].zp;
  scale=model._output_tensor[0].scale;
  for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<static_cast<int>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;

input_proto= (int8_t*)model._output[1].buf; 
 zp=model._output_tensor[1].zp;
  scale=model._output_tensor[1].scale;
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<static_cast<int>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


input_proto= (int8_t*)model._output[2].buf; 
 zp=model._output_tensor[2].zp;
  scale=model._output_tensor[2].scale;
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<static_cast<int>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


   input_proto= (int8_t*)model._output[3].buf; 
    zp=model._output_tensor[3].zp;
  scale=model._output_tensor[3].scale;
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<static_cast<int>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;
   



 
model.release_output_data();
    return 0;
}