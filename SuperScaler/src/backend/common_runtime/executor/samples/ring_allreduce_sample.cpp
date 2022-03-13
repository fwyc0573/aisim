// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>

#include "../executor_pub.hpp"

template <class DataType, class ReduceKernel>
int ring_allreduce_worker(compute_dev_id_t self_compute_dev,
                          compute_dev_id_t prev_peer_compute_dev,
                          compute_dev_id_t next_peer_compute_dev,
                          uint64_t self_compute_dev_idx,
                          uint64_t num_compute_devs,
                          DataType* cpu_data_buf,
                          uint64_t num_elements,
                          uint64_t num_loop)
{
    PollExecutor exec(self_compute_dev);
    std::vector<rank_t> peer_devs =
        {static_cast<rank_t>(prev_peer_compute_dev),
         static_cast<rank_t>(next_peer_compute_dev)};
    auto cuda_channel = std::make_shared<CudaChannel>(
        self_compute_dev, peer_devs);
    int error_ret = -1;
    int ret = 0;
    DataType *cuda_data_buf = nullptr;
    DataType *cuda_recv_buf = nullptr;
    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t k = 0;
    task_id_t send_task_id = 0;
    task_id_t recv_task_id = 0;
    task_id_t reduce_task_id = 0;
    message_id_t send_msg_id = 0;
    message_id_t recv_msg_id = 0;
    uint64_t num_chunk_elements = 0;
    uint64_t chunk_id = 0;

    std::vector<uint64_t> last_write_on_chunk_id(num_compute_devs, false);
    std::vector<std::vector<uint64_t>> last_read_ids_on_chunk_id(num_compute_devs);
    std::vector<uint64_t> last_write_id_on_chunk_id(num_compute_devs);

    uint64_t last_write_on_recv_buf = false;
    std::vector<uint64_t> last_read_ids_on_recv_buf;
    uint64_t last_write_id_on_recv_buf;

    std::vector<uint64_t> task_ids_created;

    if (!(num_compute_devs > 1 && num_elements > 0 &&
        num_elements % num_compute_devs == 0)) {
        fprintf(
            stderr,
            "[Peer %lu] num_compute_devs should be > 1, "
            "num_elements should be > 0 and "
            "divisible by num_compute_devs.\n", self_compute_dev_idx);
        return error_ret;
    }
    num_chunk_elements = num_elements / num_compute_devs;

    // Allocate GPU buffer
    checkCudaErrors(cudaMalloc(
        &cuda_data_buf, num_elements * sizeof(DataType)));
    checkCudaErrors(cudaMalloc(
        &cuda_recv_buf, num_chunk_elements * sizeof(DataType)));

    // Copy data to GPU
    checkCudaErrors(cudaMemcpy(
        cuda_data_buf, cpu_data_buf, num_elements * sizeof(DataType),
        cudaMemcpyDefault));

    // Construct DAG
    for (i = 0; i < num_loop; i++) {
        // Scatter-reduce
        for (j = 0; j < num_compute_devs - 1; j++) {
            chunk_id = (self_compute_dev_idx + num_compute_devs - j) %
                num_compute_devs;
            send_task_id = exec.create_task<SendTask>(
                &exec, nullptr, cuda_channel, next_peer_compute_dev,
                send_msg_id, cuda_data_buf + chunk_id * num_chunk_elements,
                num_chunk_elements * sizeof(DataType));
            send_msg_id++;
            task_ids_created.push_back(send_task_id);

            // Send task reads chunk_id-th buffer
            if (last_write_on_chunk_id[chunk_id]) {
                exec.add_dependence(send_task_id, last_write_id_on_chunk_id[chunk_id]);
            }
            last_read_ids_on_chunk_id[chunk_id].push_back(send_task_id);

            recv_task_id = exec.create_task<RecvTask>(
                &exec, nullptr, cuda_channel, prev_peer_compute_dev,
                recv_msg_id, cuda_recv_buf,
                num_chunk_elements * sizeof(DataType));
            recv_msg_id++;
            task_ids_created.push_back(recv_task_id);

            // Receive task writes receive buffer
            for (k = 0; k < last_read_ids_on_recv_buf.size(); k++) {
                exec.add_dependence(recv_task_id, last_read_ids_on_recv_buf[k]);
            }
            if (last_write_on_recv_buf && last_read_ids_on_recv_buf.size() == 0) {
                exec.add_dependence(recv_task_id, last_write_id_on_recv_buf);
            }
            last_write_on_recv_buf = true;
            last_write_id_on_recv_buf = recv_task_id;
            last_read_ids_on_recv_buf.clear();

            chunk_id = (self_compute_dev_idx + num_compute_devs - j - 1) %
                num_compute_devs;
            reduce_task_id =
                exec.create_task<ReductionTask<DataType, ReduceKernel>>(
                    &exec,
                    [&](TaskState) { cudaStreamSynchronize(
                        exec.get_context()->compute_dev_stream); },
                    cuda_recv_buf,
                    cuda_data_buf + chunk_id * num_chunk_elements,
                    ReduceKernel(), num_chunk_elements);
            task_ids_created.push_back(reduce_task_id);

            // Reduce task reads receive buffer, reads chunk_id-th buffer and writes chunk_id-th buffer
            if (last_write_on_recv_buf) {
                exec.add_dependence(reduce_task_id, last_write_id_on_recv_buf);
            }
            last_read_ids_on_recv_buf.push_back(reduce_task_id);

            for (k = 0; k < last_read_ids_on_chunk_id[chunk_id].size(); k++) {
                exec.add_dependence(reduce_task_id, last_read_ids_on_chunk_id[chunk_id][k]);
            }
            if (last_write_on_chunk_id[chunk_id] && last_read_ids_on_chunk_id[chunk_id].size() == 0) {
                exec.add_dependence(reduce_task_id, last_write_id_on_chunk_id[chunk_id]);
            }
            last_write_on_chunk_id[chunk_id] = true;
            last_write_id_on_chunk_id[chunk_id] = reduce_task_id;
            last_read_ids_on_chunk_id[chunk_id].clear();
        }

        // All-gather
        for (j = 0; j < num_compute_devs - 1; j++) {
            chunk_id = (self_compute_dev_idx + num_compute_devs - j + 1) %
                num_compute_devs;
            send_task_id = exec.create_task<SendTask>(
                &exec, nullptr, cuda_channel, next_peer_compute_dev,
                send_msg_id, cuda_data_buf + chunk_id * num_chunk_elements,
                num_chunk_elements * sizeof(DataType));
            send_msg_id++;
            task_ids_created.push_back(send_task_id);

            // Send task reads chunk_id-th buffer
            if (last_write_on_chunk_id[chunk_id]) {
                exec.add_dependence(send_task_id, last_write_id_on_chunk_id[chunk_id]);
            }
            last_read_ids_on_chunk_id[chunk_id].push_back(send_task_id);

            chunk_id = (self_compute_dev_idx + num_compute_devs - j) %
                num_compute_devs;
            recv_task_id = exec.create_task<RecvTask>(
                &exec, nullptr, cuda_channel, prev_peer_compute_dev,
                recv_msg_id,
                cuda_data_buf + chunk_id * num_chunk_elements,
                num_chunk_elements * sizeof(DataType));
            recv_msg_id++;
            task_ids_created.push_back(recv_task_id);

            // Receive task writes chunk_id-th buffer
            for (k = 0; k < last_read_ids_on_chunk_id[chunk_id].size(); k++) {
                exec.add_dependence(recv_task_id, last_read_ids_on_chunk_id[chunk_id][k]);
            }
            if (last_write_on_chunk_id[chunk_id] && last_read_ids_on_chunk_id[chunk_id].size() == 0) {
                exec.add_dependence(recv_task_id, last_write_id_on_chunk_id[chunk_id]);
            }
            last_write_on_chunk_id[chunk_id] = true;
            last_write_id_on_chunk_id[chunk_id] = recv_task_id;
            last_read_ids_on_chunk_id[chunk_id].clear();
        }
    }

    // Start all tasks
    for (i = 0; i < task_ids_created.size(); i++) {
        exec.add_task(task_ids_created[i]);
    }

    // Wait all tasks
    for (i = 0; i < task_ids_created.size(); i++) {
        if (exec.wait().get_state() != ExecState::e_success) {
            fprintf(stderr, "[Peer %lu] wait error\n", self_compute_dev_idx);
            ret = error_ret;
            goto clean_up;
        }
    }

    // Copy data from GPU
    checkCudaErrors(cudaMemcpy(
        cpu_data_buf, cuda_data_buf, num_elements * sizeof(DataType),
        cudaMemcpyDefault));

clean_up:
    // Clean GPU buffer
    checkCudaErrors(cudaFree(cuda_data_buf));
    checkCudaErrors(cudaFree(cuda_recv_buf));
    return ret;
}

int main(int argc, char** argv)
{
    constexpr uint64_t num_cuda_devs = 8;
    compute_dev_id_t cuda_dev_ids[num_cuda_devs] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint64_t num_elements = 64 * 1024 * 1024;
    uint64_t num_loop = 10;
    std::vector<float> cpu_data_buf;
    std::vector<float> expected_results;
    size_t i = 0;
    pid_t pid = 0;
    uint64_t self_compute_dev_idx = 0;
    float expected_result = 1.;
    int ret = 0;

    for (i = 1; i < num_cuda_devs; i++) {
        pid = fork();
        if (pid == 0) {
            self_compute_dev_idx = i;
            break;
        }
    }

    cpu_data_buf.resize(num_elements);
    expected_results.resize(num_elements);
    for (i = 0; i < num_loop; i++) {
        expected_result *= num_cuda_devs;
    }
    for (i = 0; i < num_elements; i++) {
        cpu_data_buf[i] = 1;
        expected_results[i] = expected_result;
    }

    ret = ring_allreduce_worker<float, SumKernelGPUImpl>(
        cuda_dev_ids[self_compute_dev_idx],
        cuda_dev_ids[
            (self_compute_dev_idx + num_cuda_devs - 1) % num_cuda_devs],
        cuda_dev_ids[(self_compute_dev_idx + 1) % num_cuda_devs],
        self_compute_dev_idx,
        num_cuda_devs,
        cpu_data_buf.data(),
        num_elements,
        num_loop);

    if (ret) {
        fprintf(stderr, "[Peer %lu] ring_allreduce_worker error\n",
            self_compute_dev_idx);
        return ret;
    }

    // Should disable result checking if num_loop > 1
    for (i = 0; i < num_elements; i++) {
        if (cpu_data_buf[i] != expected_results[i]) {
            fprintf(stderr,
                "[Peer %lu] %lu-th data error, "
                "expected: %g, actual: %g\n",
                self_compute_dev_idx, i, expected_results[i], cpu_data_buf[i]);
            return -1;
        }
    }
    fprintf(stdout, "[Peer %lu] Success\n", self_compute_dev_idx);

    return 0;
}
