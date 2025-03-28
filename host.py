from pynq import Overlay, allocate
import numpy as np
import time
import datetime

# 1. 오버레이 로드
ol = Overlay("rmsnorm.xclbin")
print("Overlay loaded.")

# 오버레이에 포함된 IP 코어 목록 출력
ip_info = ol.ip_dict
ip_info_str = "\n".join([f"{key}: {value}" for key, value in ip_info.items()])
print("Available IP cores in the overlay:")
print(ip_info_str)

# 2. 커널 IP 가져오기 (디자인에 따라 이름이 다를 수 있음)
kernel = ol.rmsnorm_hls_1
print("Kernel IP acquired:", kernel)

DIM = 512

# 3. DDR 버퍼 할당 (pynq.allocate 사용)
x_buf = allocate(shape=(DIM,), dtype=np.float32)
weight_buf = allocate(shape=(DIM,), dtype=np.float32)
out_buf = allocate(shape=(DIM,), dtype=np.float32)

# 4. 외부 파일에서 입력 데이터 로드
x_buf[:] = np.loadtxt("x.txt", dtype=np.float32)
weight_buf[:] = np.loadtxt("weight.txt", dtype=np.float32)
print("Input data loaded from files.")
print("x_buf[0:5] =", x_buf[:5])
print("weight_buf[0:5] =", weight_buf[:5])

# 5. 할당된 버퍼의 물리 주소 출력
x_phys = hex(x_buf.physical_address)
weight_phys = hex(weight_buf.physical_address)
out_phys = hex(out_buf.physical_address)
print("x_buf physical address:", x_phys)
print("weight_buf physical address:", weight_phys)
print("out_buf physical address:", out_phys)

# 6. 커널 인자 설정 (HLS 디자인에 맞는 레지스터 오프셋 사용)
kernel.write(0x10, x_buf.physical_address)
kernel.write(0x1C, out_buf.physical_address)
kernel.write(0x28, weight_buf.physical_address)
print("Kernel arguments set (physical addresses written).")

# 7. 커널 실행 및 성능 측정
start_time = time.time()
kernel.write(0x00, 0x01)  # CTRL 레지스터에 AP_START 신호 전송
print("Kernel started.")

# AP_DONE 비트(비트1)를 폴링하여 커널 완료 확인
while True:
    ctrl = kernel.read(0x00)
    if ctrl & (1 << 1):
        break
    time.sleep(0.001)
end_time = time.time()
kernel_exec_time = (end_time - start_time) * 1000  # ms 단위
print("Kernel execution finished.")
print("Kernel execution time: {:.3f} ms".format(kernel_exec_time))

# 8. 출력 데이터 확인 및 파일 저장
print("Output preview (first 8 values):", out_buf[:8])
np.savetxt("out.txt", out_buf, fmt="%.8f")
print("Results saved to out.txt.")

# 9. 모든 성능 및 디버그 정보를 full_info.txt에 저장
with open("full_info.txt", "w") as f:
    f.write("=== Full Performance and Debug Info ===\n")
    f.write("Timestamp: " + str(datetime.datetime.now()) + "\n\n")
    f.write("Overlay file: rmsnorm.xclbin\n\n")
    f.write("Available IP cores in the overlay:\n")
    f.write(ip_info_str + "\n\n")
    f.write("Buffer Physical Addresses:\n")
    f.write("x_buf: " + x_phys + "\n")
    f.write("weight_buf: " + weight_phys + "\n")
    f.write("out_buf: " + out_phys + "\n\n")
    f.write("Kernel Execution Time: {:.3f} ms\n".format(kernel_exec_time))
    f.write("Output Preview (first 8 values): " + str(out_buf[:8]) + "\n")
print("Full info saved to full_info.txt.")

# 10. 할당된 버퍼 해제
x_buf.close()
weight_buf.close()
out_buf.close()
from pynq import Overlay, allocate
import numpy as np
import time
import datetime

# 1. 오버레이 로드
ol = Overlay("rmsnorm.xclbin")
print("Overlay loaded.")

# 오버레이에 포함된 IP 코어 목록 출력
ip_info = ol.ip_dict
ip_info_str = "\n".join([f"{key}: {value}" for key, value in ip_info.items()])
print("Available IP cores in the overlay:")
print(ip_info_str)

# 2. 커널 IP 가져오기 (디자인에 따라 이름이 다를 수 있음)
kernel = ol.rmsnorm_hls_1
print("Kernel IP acquired:", kernel)

DIM = 512

# 3. DDR 버퍼 할당 (pynq.allocate 사용)
x_buf = allocate(shape=(DIM,), dtype=np.float32)
weight_buf = allocate(shape=(DIM,), dtype=np.float32)
out_buf = allocate(shape=(DIM,), dtype=np.float32)

# 4. 외부 파일에서 입력 데이터 로드
x_buf[:] = np.loadtxt("x.txt", dtype=np.float32)
weight_buf[:] = np.loadtxt("weight.txt", dtype=np.float32)
print("Input data loaded from files.")
print("x_buf[0:5] =", x_buf[:5])
print("weight_buf[0:5] =", weight_buf[:5])

# 5. 할당된 버퍼의 물리 주소 출력
x_phys = hex(x_buf.physical_address)
weight_phys = hex(weight_buf.physical_address)
out_phys = hex(out_buf.physical_address)
print("x_buf physical address:", x_phys)
print("weight_buf physical address:", weight_phys)
print("out_buf physical address:", out_phys)

# 6. 커널 인자 설정 (HLS 디자인에 맞는 레지스터 오프셋 사용)
kernel.write(0x10, x_buf.physical_address)
kernel.write(0x1C, out_buf.physical_address)
kernel.write(0x28, weight_buf.physical_address)
print("Kernel arguments set (physical addresses written).")

# 7. 커널 실행 및 성능 측정
start_time = time.time()
kernel.write(0x00, 0x01)  # CTRL 레지스터에 AP_START 신호 전송
print("Kernel started.")

# AP_DONE 비트(비트1)를 폴링하여 커널 완료 확인
while True:
    ctrl = kernel.read(0x00)
    if ctrl & (1 << 1):
        break
    time.sleep(0.001)
end_time = time.time()
kernel_exec_time = (end_time - start_time) * 1000  # ms 단위
print("Kernel execution finished.")
print("Kernel execution time: {:.3f} ms".format(kernel_exec_time))

# 8. 출력 데이터 확인 및 파일 저장
print("Output preview (first 8 values):", out_buf[:8])
np.savetxt("out.txt", out_buf, fmt="%.8f")
print("Results saved to out.txt.")

# 9. 모든 성능 및 디버그 정보를 full_info.txt에 저장
with open("full_info.txt", "w") as f:
    f.write("=== Full Performance and Debug Info ===\n")
    f.write("Timestamp: " + str(datetime.datetime.now()) + "\n\n")
    f.write("Overlay file: rmsnorm.xclbin\n\n")
    f.write("Available IP cores in the overlay:\n")
    f.write(ip_info_str + "\n\n")
    f.write("Buffer Physical Addresses:\n")
    f.write("x_buf: " + x_phys + "\n")
    f.write("weight_buf: " + weight_phys + "\n")
    f.write("out_buf: " + out_phys + "\n\n")
    f.write("Kernel Execution Time: {:.3f} ms\n".format(kernel_exec_time))
    f.write("Output Preview (first 8 values): " + str(out_buf[:8]) + "\n")
print("Full info saved to full_info.txt.")

# 10. 할당된 버퍼 해제
x_buf.close()
weight_buf.close()
out_buf.close()

