# 这里只写本目录的测试目标





# add_executable(unit_test videofile_test.cpp)
# add_executable(output_data_format_experiment output_data_format_experiment_int8.cpp)
# add_executable(full_process full_process_test.cpp)
# add_executable(model_output_test1 model_output_test.cpp)
# add_executable(baogao_chutu baogao_chutu.cpp)
# add_executable(test_new test.cpp)
# add_executable(shipin_show final_version.cpp)
add_executable(performance_test performance_test.cpp)


set_target_properties(performance_test PROPERTIES    #设置“安装后运行时找动态库的路径（RPATH）”
    INSTALL_RPATH "\$ORIGIN/lib"
    INSTALL_RPATH "\$ORIGIN/runtime"
)

target_link_libraries(performance_test
      PRIVATE  yolov8seg_core          # 刚才在顶层定义的库
)

install(TARGETS performance_test DESTINATION .)