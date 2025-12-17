cmake_minimum_required(VERSION 3.15)

project(rknn_model_zoo_3rdparty_libs)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set (TARGET_LIB_ARCH ${CMAKE_ANDROID_ARCH_ABI})    #CMAKE_ANDROID_ARCH_ABI 表示Android的ABI名称
else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)   #CMAKE_SIZEOF_VOID_P表示void*指针的大小  从而判断出是32位 系统还是64位系统 ;这个是自动变量（只读） 他会用工具链自己编一个程序来得到指针大小
        set (TARGET_LIB_ARCH aarch64)
    else()
        set (TARGET_LIB_ARCH armhf)
    endif()
    if (CMAKE_C_COMPILER MATCHES "uclibc")  #判断编译器是不是用uClibc  ;  MATCHES正则匹配字符串  
        set (TARGET_LIB_ARCH ${TARGET_LIB_ARCH}_uclibc)
    endif()
endif()



# stb_image.h(纯头文件库)  非常常用、非常轻量级的图像读取库
# 由著名的单头文件库作者 Sean Barrett 编写。  主要功能：从文件中读取图片并解码成原始像素数据（RGB、RGBA、灰度）
set(STB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/stb_image)
set(STB_INCLUDES ${STB_PATH} PARENT_SCOPE)


# jpeg turbo 是一个“超快的 JPEG 编解码库”，专门用来压缩/解压 JPG 图片的
#主要功能：
#把 JPEG 文件解码成原始像素（RGB / BGR / YUV / 灰度）
#把原始像素编码成 JPEG 文件
set(JPEG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_turbo)
set(LIBJPEG ${JPEG_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libturbojpeg.a PARENT_SCOPE)
set(LIBJPEG_INCLUDES ${JPEG_PATH}/include PARENT_SCOPE)



# rknn runtime     瑞芯微的NPU推理运行时库，就是哪些RKNN的api函数
# for rknpu2
if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk3576" OR TARGET_SOC STREQUAL "rk356x" OR TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103" OR TARGET_SOC STREQUAL "rv1126b")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/rknpu2)
    if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk356x" OR TARGET_SOC STREQUAL "rk3576")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknnrt.so)
    endif()
    if (TARGET_SOC STREQUAL "rv1126b")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknnrt.so)
    endif()
    if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/armhf-uclibc/librknnmrt.so)
    endif()
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include PARENT_SCOPE)
endif()


# for rknpu1
if(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/rknpu1)

    set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknn_api.so)

    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include PARENT_SCOPE)
endif()
install(PROGRAMS ${LIBRKNNRT} DESTINATION lib)
set(LIBRKNNRT ${LIBRKNNRT} PARENT_SCOPE)




# rga=Rockchip Graphics Accelerator 瑞芯微自己的 2D 图像硬件加速模块
#常用来：把摄像头 / 视频输出的 YUV 图按 NPU 需要的分辨率/格式做 快速缩放 & 格式转换； 比单纯用 CPU / OpenCV resize 要省不少时间。
set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librga)
set(LIBRGA ${RGA_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librga.a PARENT_SCOPE)
set(LIBRGA_INCLUDES ${RGA_PATH}/include PARENT_SCOPE)
install(PROGRAMS ${RGA_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librga.so DESTINATION lib)  
#PROGRAMS按“普通文件”的方式安装这些路径上的文件，保留权限，不做 RPATH 等特殊处理
#DESTINATION指的是在安装根目录下的相对路径。 安装根目录 = CMAKE_INSTALL_PREFIX



# timer 一个小工具   用来计时打点工具
set(TIMER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/timer)
set(LIBTIMER_INCLUDES ${TIMER_PATH} PARENT_SCOPE)


# libsndfile标准的音频文件读写库
set(LIBSNDFILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libsndfile)
set(LIBSNDFILE_INCLUDES ${LIBSNDFILE_PATH}/include PARENT_SCOPE)
set(LIBSNDFILE ${LIBSNDFILE_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libsndfile.a PARENT_SCOPE)

# fftw=Fastest Fourier Transform in the West 是一个高性能 FFT 库； libfftw3f.a是单精度版
#主要用途：
# 对音频做 FFT / STFT，把时域信号变到频域；
# 为计算谱图、mel 滤波器组、fbank 等提供底层 FFT 支持。
set(LIBFFTW_PATH ${CMAKE_CURRENT_SOURCE_DIR}/fftw)
set(LIBFFTW_INCLUDES ${LIBFFTW_PATH}/include PARENT_SCOPE)
set(LIBFFTW ${LIBFFTW_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libfftw3f.a PARENT_SCOPE)


# kaldi_native_fbank  这是从 Kaldi 派生出来的一个 fbank 特征提取库，通常叫 kaldi-native-fbank  fbank = filterbank，mel 滤波器组特征，是语音识别里常用的输入特征。
set(LIBKALDI_NATIVE_FBANK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/kaldi_native_fbank)
set(LIBKALDI_NATIVE_FBANK_INCLUDES ${LIBKALDI_NATIVE_FBANK_PATH}/include PARENT_SCOPE)
set(LIBKALDI_NATIVE_FBANK ${LIBKALDI_NATIVE_FBANK_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libkaldi-native-fbank-core.a PARENT_SCOPE)