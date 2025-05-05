import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# 파라미터 설정
fs = 31250000  # 샘플링 주파수 (31.25 MHz)
snr = 0  # SNR (Signal-to-Noise Ratio)

# 이진 파일 읽기 (예: I/Q 데이터)
filename = 'radar_data.bin'
raw_data = np.fromfile(filename, dtype=np.float32)  # 파일 형식에 맞게 dtype 조정

# I/Q 데이터 분리 (I와 Q가 교대로 저장되었다고 가정)
I = raw_data[::2]  # 홀수 인덱스: I 데이터
Q = raw_data[1::2]  # 짝수 인덱스: Q 데이터

# 복소수 신호로 재구성
signal = I + 1j * Q  # I + jQ

# SNR 적용 (SNR = 0으로 가정)
# SNR이 0이므로 신호에 노이즈를 추가하지 않음 (원본 신호 그대로 사용)
# 만약 SNR을 조정하려면, 아래 코드를 사용하여 노이즈를 추가할 수 있습니다.
# signal_power = np.mean(np.abs(signal)**2)
# noise_power = signal_power / (10 ** (snr / 10))
# noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
# signal_with_noise = signal + noise

# STFT 계산
f, t, Zxx = stft(signal, fs, nperseg=256, noverlap=128)  # 윈도우 크기 및 중첩 설정

# 스펙토그램 시각화
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('Radar Signal Spectrogram (FS = 31.25 MHz, SNR = 0)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.show()
