import numpy as np

WINDOW_SIZE = 4096
path = "/home/andrew/DroneDetect_V2_SLICED/BOTH/INS_HO/INS_1101_03_seg0427.fc32"

def analyze_file(path):
    data = np.fromfile(path, dtype=np.complex64)
    max_energy = 0
    max_std = 0
    for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE):
        window = data[i:i+WINDOW_SIZE]
        abs_vals = np.abs(window)
        energy = np.mean(abs_vals ** 2)
        spread = np.std(abs_vals)
        if energy > max_energy:
            max_energy = energy
        if spread > max_std:
            max_std = spread
    return max_energy, max_std

energy, stddev = analyze_file(path)
print(f"Max energy  : {energy:.4f}")
print(f"Max stddev  : {stddev:.4f}")
print(f"\nPasses threshold? {'YES' if energy > 0.1861 and stddev > 0.2022 else 'NO'}")

