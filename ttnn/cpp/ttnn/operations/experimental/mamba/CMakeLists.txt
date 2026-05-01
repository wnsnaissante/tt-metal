add_library(ttnn_op_experimental_mamba ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Mamba ALIAS ttnn_op_experimental_mamba)

tt_reuse_precompile_headers(ttnn_op_experimental_mamba TTNN::PCH)

target_sources(
    ttnn_op_experimental_mamba
    PRIVATE
        ssd_output/mamba_ssd_output.cpp
        ssd_recurrence/mamba_ssd_recurrence.cpp
        ssd_chunk_scan/mamba_ssd_chunk_scan.cpp
        ssd_output/device/mamba_ssd_output_device_operation.cpp
        ssd_output/device/mamba_ssd_output_program_factory.cpp
        ssd_recurrence/device/mamba_ssd_recurrence_device_operation.cpp
        ssd_recurrence/device/mamba_ssd_recurrence_program_factory.cpp
        ssd_chunk_scan/device/mamba_ssd_chunk_scan_device_operation.cpp
        ssd_chunk_scan/device/mamba_ssd_chunk_scan_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_mamba PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_mamba
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_mamba LIBRARY COMPONENT tar)
