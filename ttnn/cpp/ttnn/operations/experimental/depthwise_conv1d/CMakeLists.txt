add_library(ttnn_op_experimental_depthwise_conv1d ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DepthwiseConv1dV2 ALIAS ttnn_op_experimental_depthwise_conv1d)

tt_reuse_precompile_headers(ttnn_op_experimental_depthwise_conv1d TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_depthwise_conv1d)

target_sources(
    ttnn_op_experimental_depthwise_conv1d
    PRIVATE
        depthwise_conv1d.cpp
        device/depthwise_conv1d_device_operation.cpp
        device/depthwise_conv1d_forward_device_operation.cpp
        device/depthwise_conv1d_forward_program_factory.cpp
        device/depthwise_conv1d_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_depthwise_conv1d PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_depthwise_conv1d
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_depthwise_conv1d LIBRARY COMPONENT tar)
