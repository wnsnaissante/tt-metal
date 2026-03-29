// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_recurrence_program_factory.hpp"
#include "mamba_ssd_recurrence_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

namespace {

struct HiddenTileWorkSplit {
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
    CoreRangeSet all_cores = CoreRangeSet{};
    CoreRangeSet core_group_1 = CoreRangeSet{};
    CoreRangeSet core_group_2 = CoreRangeSet{};
    uint32_t hidden_tiles_per_core_group_1 = 0;
    uint32_t hidden_tiles_per_core_group_2 = 0;
};

CoreCoord get_worker_grid_size(const MambaSSDRecurrenceParams& operation_attributes, tt::tt_metal::IDevice* device) {
    if (operation_attributes.has_core_grid) {
        return CoreCoord{operation_attributes.core_grid_x, operation_attributes.core_grid_y};
    }
    return device->compute_with_storage_grid_size();
}

HiddenTileWorkSplit split_hidden_tile_work(
    const MambaSSDRecurrenceParams& operation_attributes,
    const MambaSSDRecurrenceKernelConfig& kernel_config,
    tt::tt_metal::IDevice* device) {
    const auto worker_grid_size = get_worker_grid_size(operation_attributes, device);
    const uint32_t head_tiles = tt::div_up(kernel_config.head_dim, tt::constants::TILE_HEIGHT);
    const uint32_t state_tiles = tt::div_up(kernel_config.state_size, tt::constants::TILE_WIDTH);
    const uint32_t total_hidden_tiles = kernel_config.bh * head_tiles * state_tiles;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         hidden_tiles_per_core_group_1,
         hidden_tiles_per_core_group_2] = tt::tt_metal::split_work_to_cores(worker_grid_size, total_hidden_tiles);
    return HiddenTileWorkSplit{
        .num_cores = num_cores,
        .num_cores_y = worker_grid_size.y,
        .all_cores = all_cores,
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .hidden_tiles_per_core_group_1 = hidden_tiles_per_core_group_1,
        .hidden_tiles_per_core_group_2 = hidden_tiles_per_core_group_2,
    };
}

}  // namespace

MambaSSDRecurrenceProgramFactory::cached_program_t MambaSSDRecurrenceProgramFactory::create(
    const MambaSSDRecurrenceParams& operation_attributes,
    const MambaSSDRecurrenceInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const auto& states_bhcpn = tensor_args.states_bhcpn;
    const auto& initial_states = tensor_args.initial_states;
    const auto& a_end_bhc = tensor_args.a_end_bhc;
    auto& states_out = tensor_return_value.at(0);
    auto& final_state = tensor_return_value.at(1);

    auto* states_buffer = states_bhcpn.buffer();
    auto* initial_states_buffer = initial_states.buffer();
    auto* a_end_buffer = a_end_bhc.buffer();
    const auto kernel_config = build_mamba_ssd_recurrence_kernel_config(operation_attributes, tensor_args);
    auto* states_out_buffer = states_out.buffer();
    auto* final_state_buffer = final_state.buffer();
    tt::tt_metal::Program program{};
    const auto work_split = split_hidden_tile_work(operation_attributes, kernel_config, states_bhcpn.device());

    const auto states_data_format = tt::tt_metal::datatype_to_dataformat_converter(states_bhcpn.dtype());
    const auto initial_states_data_format = tt::tt_metal::datatype_to_dataformat_converter(initial_states.dtype());
    const auto a_end_data_format = tt::tt_metal::datatype_to_dataformat_converter(a_end_bhc.dtype());
    const auto states_out_data_format = tt::tt_metal::datatype_to_dataformat_converter(states_out.dtype());
    const auto final_state_data_format = tt::tt_metal::datatype_to_dataformat_converter(final_state.dtype());

    const auto states_tile_size = tt::tile_size(states_data_format);
    const auto initial_states_tile_size = tt::tile_size(initial_states_data_format);
    const auto a_end_tile_size = tt::tile_size(a_end_data_format);
    const auto states_out_tile_size = tt::tile_size(states_out_data_format);
    const auto final_state_tile_size = tt::tile_size(final_state_data_format);
    const auto intermediary_data_format = tt::DataFormat::Float16_b;
    const auto intermediary_tile_size = tt::tile_size(intermediary_data_format);

    constexpr uint32_t states_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t initial_states_cb_index = tt::CBIndex::c_1;
    constexpr uint32_t a_end_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t states_out_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t final_state_cb_index = tt::CBIndex::c_17;
    constexpr uint32_t a_end_scratch_cb_index = tt::CBIndex::c_25;
    constexpr uint32_t h_prev_cb_index = tt::CBIndex::c_26;
    constexpr uint32_t ah_cb_index = tt::CBIndex::c_27;
    constexpr uint32_t h_cb_index = tt::CBIndex::c_28;
    constexpr uint32_t exp_a_cb_index = tt::CBIndex::c_29;
    constexpr uint32_t h_acc_cb_index = tt::CBIndex::c_31;
    const uint32_t max_hidden_tiles_per_core =
        std::max(work_split.hidden_tiles_per_core_group_1, work_split.hidden_tiles_per_core_group_2);

    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(states_tile_size, {{states_cb_index, states_data_format}})
            .set_page_size(states_cb_index, states_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(
            initial_states_tile_size, {{initial_states_cb_index, initial_states_data_format}})
            .set_page_size(initial_states_cb_index, initial_states_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(a_end_tile_size, {{a_end_cb_index, a_end_data_format}})
            .set_page_size(a_end_cb_index, a_end_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(states_out_tile_size, {{states_out_cb_index, states_out_data_format}})
            .set_page_size(states_out_cb_index, states_out_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(final_state_tile_size, {{final_state_cb_index, final_state_data_format}})
            .set_page_size(final_state_cb_index, final_state_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(a_end_tile_size, {{a_end_scratch_cb_index, a_end_data_format}})
            .set_page_size(a_end_scratch_cb_index, a_end_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(2 * intermediary_tile_size, {{h_prev_cb_index, intermediary_data_format}})
            .set_page_size(h_prev_cb_index, intermediary_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(2 * intermediary_tile_size, {{ah_cb_index, intermediary_data_format}})
            .set_page_size(ah_cb_index, intermediary_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(2 * intermediary_tile_size, {{h_cb_index, intermediary_data_format}})
            .set_page_size(h_cb_index, intermediary_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(2 * intermediary_tile_size, {{exp_a_cb_index, intermediary_data_format}})
            .set_page_size(exp_a_cb_index, intermediary_tile_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(
            max_hidden_tiles_per_core * intermediary_tile_size, {{h_acc_cb_index, intermediary_data_format}})
            .set_page_size(h_acc_cb_index, intermediary_tile_size));

    auto reader_compile_args =
        std::vector<uint32_t>{states_cb_index, initial_states_cb_index, a_end_cb_index, a_end_scratch_cb_index};
    const auto compute_compile_args = std::vector<uint32_t>{
        states_cb_index,
        initial_states_cb_index,
        a_end_cb_index,
        states_out_cb_index,
        final_state_cb_index,
        h_prev_cb_index,
        ah_cb_index,
        h_cb_index,
        exp_a_cb_index,
        h_acc_cb_index,
        1};
    auto writer_compile_args = std::vector<uint32_t>{states_out_cb_index, final_state_cb_index};
    tt::tt_metal::TensorAccessorArgs(*states_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*initial_states_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*a_end_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*states_out_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*final_state_buffer).append_to(writer_compile_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kReaderKernelPath},
        work_split.all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kComputeKernelPath},
        work_split.all_cores,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_args});

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kWriterKernelPath},
        work_split.all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    for (uint32_t i = 0, hidden_tile_start = 0; i < work_split.num_cores; ++i) {
        CoreCoord core = {i / work_split.num_cores_y, i % work_split.num_cores_y};
        uint32_t hidden_tile_count = 0;
        if (work_split.core_group_1.contains(core)) {
            hidden_tile_count = work_split.hidden_tiles_per_core_group_1;
        } else if (work_split.core_group_2.contains(core)) {
            hidden_tile_count = work_split.hidden_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                states_buffer->address(),
                initial_states_buffer->address(),
                a_end_buffer->address(),
                kernel_config.batch_size,
                kernel_config.num_heads,
                kernel_config.num_chunks,
                kernel_config.head_dim,
                kernel_config.state_size,
                hidden_tile_start,
                hidden_tile_count,
            });
        tt::tt_metal::SetRuntimeArgs(
            program,
            compute_kernel_id,
            core,
            {
                kernel_config.num_chunks,
                kernel_config.head_dim,
                kernel_config.state_size,
                hidden_tile_count,
            });
        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                states_out_buffer->address(),
                final_state_buffer->address(),
                kernel_config.batch_size,
                kernel_config.num_heads,
                kernel_config.num_chunks,
                kernel_config.head_dim,
                kernel_config.state_size,
                hidden_tile_start,
                hidden_tile_count,
            });
        hidden_tile_start += hidden_tile_count;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = work_split.all_cores}};
}

void MambaSSDRecurrenceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MambaSSDRecurrenceParams& operation_attributes,
    const MambaSSDRecurrenceInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    const auto kernel_config = build_mamba_ssd_recurrence_kernel_config(operation_attributes, tensor_args);
    const auto work_split =
        split_hidden_tile_work(operation_attributes, kernel_config, tensor_args.states_bhcpn.device());

    for (uint32_t i = 0, hidden_tile_start = 0; i < work_split.num_cores; ++i) {
        CoreCoord core = {i / work_split.num_cores_y, i % work_split.num_cores_y};
        uint32_t hidden_tile_count = 0;
        if (work_split.core_group_1.contains(core)) {
            hidden_tile_count = work_split.hidden_tiles_per_core_group_1;
        } else if (work_split.core_group_2.contains(core)) {
            hidden_tile_count = work_split.hidden_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        auto& reader_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
        reader_runtime_args[0] = tensor_args.states_bhcpn.buffer()->address();
        reader_runtime_args[1] = tensor_args.initial_states.buffer()->address();
        reader_runtime_args[2] = tensor_args.a_end_bhc.buffer()->address();
        reader_runtime_args[3] = kernel_config.batch_size;
        reader_runtime_args[4] = kernel_config.num_heads;
        reader_runtime_args[5] = kernel_config.num_chunks;
        reader_runtime_args[6] = kernel_config.head_dim;
        reader_runtime_args[7] = kernel_config.state_size;
        reader_runtime_args[8] = hidden_tile_start;
        reader_runtime_args[9] = hidden_tile_count;

        auto& compute_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.compute_kernel_id, core);
        compute_runtime_args[0] = kernel_config.num_chunks;
        compute_runtime_args[1] = kernel_config.head_dim;
        compute_runtime_args[2] = kernel_config.state_size;
        compute_runtime_args[3] = hidden_tile_count;

        auto& writer_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        writer_runtime_args[0] = tensor_return_value.at(0).buffer()->address();
        writer_runtime_args[1] = tensor_return_value.at(1).buffer()->address();
        writer_runtime_args[2] = kernel_config.batch_size;
        writer_runtime_args[3] = kernel_config.num_heads;
        writer_runtime_args[4] = kernel_config.num_chunks;
        writer_runtime_args[5] = kernel_config.head_dim;
        writer_runtime_args[6] = kernel_config.state_size;
        writer_runtime_args[7] = hidden_tile_start;
        writer_runtime_args[8] = hidden_tile_count;
        hidden_tile_start += hidden_tile_count;
    }
}

}  // namespace ttnn::experimental::prim
