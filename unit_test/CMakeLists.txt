# 测试和可视化入口。所有涉及椭圆拟合的目标都链接统一的
# ellipse_fitter 模块（已包含在 yolov8seg_core 中）。

set(YOLO_TEST_TARGETS
    batch_image_video
    batch_mask_ring_visualization
    yolo_mask_bbox_error
    single_frame_pipeline
    shipin_show
)

add_executable(batch_image_video batch_image_video.cpp)
add_executable(batch_mask_ring_visualization batch_mask_ring_visualization.cpp)
add_executable(yolo_mask_bbox_error yolo_mask_bbox_error.cpp)
add_executable(single_frame_pipeline single_frame_pipeline.cpp)
add_executable(shipin_show final_version.cpp)

foreach(target_name IN LISTS YOLO_TEST_TARGETS)
    set_target_properties(${target_name} PROPERTIES
        INSTALL_RPATH "\$ORIGIN/lib;\$ORIGIN/runtime"
    )
    target_link_libraries(${target_name} PRIVATE yolov8seg_core)
endforeach()

install(TARGETS ${YOLO_TEST_TARGETS} DESTINATION .)
