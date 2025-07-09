#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <xcl2.hpp>
#include <ap_int.h>

#define CH_NM 16
#define N_TASK (CH_NM * 100)
#define MAX_MSG_BYTES 64

std::string get_sha1_string(ap_uint<512> hash) {
    std::ostringstream oss;
    for (int i = 0; i < 20; ++i) {
        unsigned char byte = hash.range(i * 8 + 7, i * 8);
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
    }
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " sha.xclbin" << std::endl;
        return 1;
    }

    std::string xclbin_path = argv[1];
    std::vector<std::string> messages;
    const char* msgA = "abcde";
    const char* msgB = "fghij";

    for (int i = 0; i < N_TASK; ++i) {
        messages.push_back((i % 2 == 0) ? msgA : msgB);
    }

    std::vector<ap_uint<512>, aligned_allocator<ap_uint<512>>> inputData(1 + N_TASK);
    std::vector<ap_uint<512>, aligned_allocator<ap_uint<512>>> outputData(N_TASK);

    ap_uint<512> config = 0;
    config.range(511, 448) = strlen(msgA);
    config.range(447, 384) = N_TASK;
    inputData[0] = config;

    for (int i = 0; i < N_TASK; ++i) {
        ap_uint<512> msgBlock = 0;
        const std::string& msg = messages[i];
        for (size_t j = 0; j < msg.size(); ++j) {
            msgBlock.range(j * 8 + 7, j * 8) = msg[j];
        }
        inputData[1 + i] = msgBlock;
    }

    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    cl::Context context(device, nullptr, nullptr, nullptr, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_path);
    cl::Program program(context, {device}, bins);
    cl::Kernel kernel(program, "sha1Kernel");

    cl::Buffer inBuf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(ap_uint<512>) * inputData.size(), inputData.data(), &err);
    cl::Buffer outBuf(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                      sizeof(ap_uint<512>) * outputData.size(), outputData.data(), &err);

    kernel.setArg(0, inBuf);
    kernel.setArg(1, outBuf);

    // Start enqueue
    q.enqueueMigrateMemObjects({inBuf}, 0);

    // 커널 실행 시간 측정용 이벤트
    cl::Event kernel_event;
    q.enqueueTask(kernel, nullptr, &kernel_event);
    q.enqueueMigrateMemObjects({outBuf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    // 커널 실행 시간 추출
    cl_ulong start, end;
    kernel_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    kernel_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    double kernel_time_us = (end - start) / 1000.0;

    const std::string golden[2] = {
        "03de6c570bfe24bfc328ccd7ca46b76eadaf4334",
        "27c97e303a5d4a98562a9f76696ba1fec889fc02"
    };

    for (int i = 0; i < N_TASK; ++i) {
        std::string msg = messages[i];
        std::string result = get_sha1_string(outputData[i]);
        std::string expected = golden[i % 2];
        std::cout << "[CH" << std::setw(2) << (i % CH_NM)
                  << "] [msg: " << msg << "] SHA1: " << result
                  << ((result == expected) ? "  OK" : "  MISMATCH") << std::endl;
    }

    std::cout << "Kernel execution time: " << kernel_time_us << " us\n";
    return 0;
}
