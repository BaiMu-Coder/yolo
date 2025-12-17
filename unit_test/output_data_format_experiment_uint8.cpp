#include "yolov8seg.hpp"
#include "image_process.hpp"
#include <opencv2/opencv.hpp>


#include <iostream>





inline static uint32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}


static float deqnt_affine_to_f32(uint8_t proto, uint32_t zp, float scale)   //这种处理方式是唯一解
{
  return ((float)proto-(float)zp)*scale ;
}

static uint8_t qnt_f32_to_affine(float f32 , uint32_t zp , float scale)
{
  float dst_val=(f32/scale)+zp;
  uint8_t res=(uint8_t)__clip(dst_val,0,255);
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


  uint8_t* input_proto = (uint8_t*)model._output[12].buf; 
  uint32_t zp=model._output_tensor[12].zp;
  float scale=model._output_tensor[12].scale;



  // std::cout<<"proto"<<std::endl;
  //  for (int i = 0; i <200; i++) {
  //        std::cout<<static_cast<int>(input_proto[i])<<" ";
  //   }


  //  std::cout<<std::endl;
  //  std::cout<<std::endl;


  // for (int i = 0; i <200; i++) {
  //     float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
  //       std::cout<<static_cast<int>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
  //   }
  // std::cout<<std::endl;
  //  std::cout<<std::endl;




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
input_proto= (uint8_t*)model._output[0].buf; 
  for (int i = 0; i <200; i++) {
        std::cout<<static_cast<uint>(input_proto[i]) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;

input_proto= (uint8_t*)model._output[1].buf; 
     for (int i = 0; i <200; i++) {
      std::cout<<static_cast<uint>(input_proto[i]) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


input_proto= (uint8_t*)model._output[2].buf; 
     for (int i = 0; i <200; i++) {
     std::cout<<static_cast<uint>(input_proto[i]) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;

  input_proto= (uint8_t*)model._output[3].buf; 
     for (int i = 0; i <200; i++) {
  std::cout<<static_cast<uint>(input_proto[i]) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;






    std::cout<<"80*80"<<std::endl;
input_proto= (uint8_t*)model._output[0].buf; 
  for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<tem <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;

input_proto= (uint8_t*)model._output[1].buf; 
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
       std::cout<<tem <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


input_proto= (uint8_t*)model._output[2].buf; 
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<tem <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


   input_proto= (uint8_t*)model._output[3].buf; 
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
       std::cout<<tem <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;






 std::cout<<"80*80"<<std::endl;
input_proto= (uint8_t*)model._output[0].buf; 
  for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<static_cast<uint>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;

input_proto= (uint8_t*)model._output[1].buf; 
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<static_cast<uint>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


input_proto= (uint8_t*)model._output[2].buf; 
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<static_cast<uint>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;


   input_proto= (uint8_t*)model._output[3].buf; 
     for (int i = 0; i <200; i++) {
      float tem=deqnt_affine_to_f32(input_proto[i],zp,scale);
        std::cout<<static_cast<uint>(qnt_f32_to_affine(tem,zp,scale)) <<" ";
    }
  std::cout<<std::endl;
   std::cout<<std::endl;
   



 
model.release_output_data();
    return 0;
}