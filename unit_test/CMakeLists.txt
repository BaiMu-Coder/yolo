# 这里只写本目录的测试目标





# add_executable(unit_test videofile_test.cpp)
# add_executable(output_data_format_experiment output_data_format_experiment_int8.cpp)
# add_executable(full_process full_process_test.cpp)
add_executable(model_output_test model_output_test.cpp)


set_target_properties(model_output_test PROPERTIES    #设置“安装后运行时找动态库的路径（RPATH）”
    INSTALL_RPATH "\$ORIGIN/lib"
)

target_link_libraries(model_output_test
      PRIVATE  yolov8seg_core          # 刚才在顶层定义的库
)

install(TARGETS model_output_test DESTINATION .)