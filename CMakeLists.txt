cmake_minimum_required(VERSION 3.10)

#环境配置区  只放 交叉编译 / 工具链 /平台相关设置
set(TOOLCHAIN_DIR /home/baimu/tools/rk3588_cross_compiler)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu-gcc)
set(TARGET_SOC rk3588)
set(CMAKE_SYSTEM_NAME Linux)
set(SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot/usr/include)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot)   #设置CMake的目标系统根目录





#正式内容
message(STATUS "cmake version ${CMAKE_VERSION}")
project(yolov8seg)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_INSTALL_PREFIX  ${CMAKE_CURRENT_SOURCE_DIR}/install)



#把这些字符串原样加到编译命令行（目录及其子目录的作用域）
add_definitions(-g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)
add_definitions(-Wno-write-strings -Wno-return-type)


find_package(OpenCV REQUIRED)      # 用了 OpenCV

add_subdirectory(3rdparty)


# message(STATUS "LIBRKNNRT_INCLUDES = ${LIBRKNNRT_INCLUDES}") 打印变量信息
  
include_directories(
    #3rdparty
    ${STB_INCLUDES}
    ${LIBJPEG_INCLUDES}
    ${LIBRKNNRT_INCLUDES}
    ${LIBRGA_INCLUDES}
    ${LIBTIMER_INCLUDES}
    ${LIBSNDFILE_INCLUDES}
    ${LIBFFTW_INCLUDES}
    ${LIBKALDI_NATIVE_FBANK_INCLUDES}
 

    #各模块
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/image_process
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_infer_pool
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/thread_pool
    ${CMAKE_CURRENT_SOURCE_DIR}/visual
    ${CMAKE_CURRENT_SOURCE_DIR}/business
    ${CMAKE_CURRENT_SOURCE_DIR}/block_queue

    #三方库
    ${OpenCV_INCLUDE_DIRS}   
)


file(GLOB CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image_process/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infer_pool/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thread_pool/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/*.cpp
)

add_library(yolov8seg_core STATIC ${CORE_SOURCES})

 
target_link_libraries(yolov8seg_core   #静态库本质就是一堆 .o文件   链接阶段只会把需要的.o文件拷贝进可执行程序
      PUBLIC      
        ${OpenCV_LIBS} 
        ${LIBRKNNRT}
        ${LIBJPEG}
        ${LIBRGA}
        ${LIBSNDFILE}
        ${LIBFFTW}
        ${LIBKALDI_NATIVE_FBANK}
)



# 2. 把 unit_test 子目录也加入构建
add_subdirectory(unit_test)


