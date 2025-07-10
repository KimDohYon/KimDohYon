

#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>

#include "event_timer.hpp"

#define DBG

#define CHECK_CL(status, msg) \
    if (status != CL_SUCCESS) { std::cerr << msg << " Error: " << status << std::endl; exit(1); }

void printDivider() {
    std::cout << "------------------------------------------------------------------------------" << std::endl;
}


void printPlatformInfo(const cl::Platform& platform) {
    std::string name, vendor, version;
    platform.getInfo(CL_PLATFORM_NAME, &name);
    platform.getInfo(CL_PLATFORM_VENDOR, &vendor);
    platform.getInfo(CL_PLATFORM_VERSION, &version);
    std::cout << "  Name            : " << name << std::endl;
    std::cout << "  Vendor          : " << vendor << std::endl;
    std::cout << "  Version         : " << version << std::endl;
}

void printDeviceInfo(const cl::Device& device) {
    std::string name;
    cl_device_type type;
    cl_uint compute_units;

    device.getInfo(CL_DEVICE_NAME, &name);
    device.getInfo(CL_DEVICE_TYPE, &type);
    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compute_units);

    std::string type_str = (type == CL_DEVICE_TYPE_CPU)         ? "CPU" :
                           (type == CL_DEVICE_TYPE_GPU)         ? "GPU" :
                           (type == CL_DEVICE_TYPE_ACCELERATOR) ? "ACCELERATOR" :
                           (type == CL_DEVICE_TYPE_CUSTOM)      ? "CUSTOM" : "UNKNOWN";

    std::cout << "    Device Name   : " << name << std::endl;
    std::cout << "    Device Type   : " << type_str << std::endl;
    std::cout << "    Compute Units : " << compute_units << std::endl;
}

void printContextInfo(const cl::Context& context) {
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "  Context has " << devices.size() << " device(s):" << std::endl;

    for (const auto& dev : devices) {
        printDeviceInfo(dev);
    }
}

int main() {

    event_timer et;
    const int size = 1024;
  
    // Prepare input
    et.add("OCL data preparation");
    std::vector<int>    A(size);
    std::vector<int>    B(size);
    for (int k = 0; k < size; k++) {
        A[k] = rand() % 1000;
        B[k] = rand() % 1000;
    }
    et.finish();
  
    std::vector<int>    C(size);
    std::vector<int>    C_SW(size);
  
    et.add("SW computation");
    for (int k = 0; k < size; k++) {
        C_SW[k] = A[k] + B[k];
    }
    et.finish();

    // 1. Enumerate platforms
    et.add("1. OCL platform enumeration");

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!!" << std::endl;
        return 1;
    }
    std::cout << "Found " << platforms.size() << " OpenCL platform(s):" << std::endl;
    et.finish();

    // 2. Enumerate devices in platforms
    et.add("2. OCL device enumeration");
    std::vector<std::vector<cl::Device>> all_devices;
    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        all_device  s.push_back(devices);
    }

    printDivider();
    for (size_t i = 0; i < platforms.size(); i++) {
        printPlatformInfo(platforms[i]);
        for (auto& device : all_devices[i]) {
            printDeviceInfo(device);
        }
        printDivider();
    }
    et.finish();

    // 3. Create Context - we assume we use platform 0, device 0
    et.add("3. OCL create context");
    cl::Device device = all_devices[0][0];
    cl::Context context({device});
    printContextInfo(context);
    et.finish();

    // 4. Create Command Queue
    et.add("4. OCL create command queue");
    cl::CommandQueue queue(context, device);
    et.finish();

    // 5. Load Program
    et.add("5. OCL load program");
    std::ifstream kernelFile("./hw_src/vadd.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Kernel file not found!" << std::endl;
        return 1;
    }

    std::string source((std::istreambuf_iterator<char>(kernelFile)), {});
    cl::Program::Sources sources(1, std::make_pair(source.c_str(), source.length()));
    et.finish();

    // 6. Build Program
    et.add("6. OCL build program");
    cl::Program program(context, sources);
    cl_int err = program.build({device});
    if (err != CL_SUCCESS) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Build log:\n" << build_log << std::endl;
        return 1;
    }
    et.finish();

    // 7. Create Kernel
    et.add("7. OCL create kernel");
    cl::Kernel kernel(program, "vadd");
    et.finish();

    // 8-9. Create Buffer & Transfer to Global Memory
    et.add("8-9. OCL buffer write");
    cl::Buffer bufA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, A.data());
    cl::Buffer bufB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, B.data());
    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(int) * size);
    et.finish();

    // 10. Setup Kernel Arguments
    et.add("10. OCL setup kernel");
    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);
    et.finish();

    // 11. Execute Kernel
    et.add("11. OCL execute kernel");
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size));
    queue.finish();
    et.finish();

    // 12. Read Buffer
    et.add("12. OCL buffer readback");
    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(int) * size, C.data());
    et.finish();

    // 13. Check Result
    et.add("compare");
    for (int k = 0; k < size; k++) {
        if (C[k] != C_SW[k]) {
            printf("FAIL!! : A[%4d] = %4d / B[%4d] = %4d / C[%4d] = %4d / C_SW[%4d] = %4d\n", k, A[k], k, B[k], k, C[k], k, C_SW[k]);
            break;
        } else {
            printf("PASS!! : A[%4d] = %4d / B[%4d] = %4d / C[%4d] = %4d / C_SW[%4d] = %4d\n", k, A[k], k, B[k], k, C[k], k, C_SW[k]);
        }
    }
    et.finish();

    et.print();
    return 0;
}
