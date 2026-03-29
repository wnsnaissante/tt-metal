add_library(ttnn_op_experimental_depthwiseconv1d ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DepthwiseConv1d ALIAS ttnn_op_experimental_depthwiseconv1d)

tt_reuse_precompile_headers(ttnn_op_experimental_depthwiseconv1d TTNN::PCH)

target_sources(
    ttnn_op_experimental_depthwiseconv1d
    PRIVATE
        depthwise_conv1d.cpp
        device/depthwise_conv1d_forward_device_operation.cpp
        device/depthwise_conv1d_forward_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_depthwiseconv1d PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_depthwiseconv1d
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_depthwiseconv1d LIBRARY COMPONENT tar)
