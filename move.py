import os
import numpy as np
import shutil

# Base path to processed IQ segments
BASE_DIR = "/home/andrew/DroneDetect_V2_SLICED/CLEAN".replace("\\", "/")

# Sliding window size and threshold for signal detection
WINDOW_SIZE = 4096
ENERGY_THRESHOLD = 0.2
MIN_SAMPLE_THRESHOLD = 60000  # Below this, assume corrupted or empty

def has_valid_packet(file_path):
    """
    Check if a file contains a valid RF signal burst
    by analyzing sliding window average energy.
    """
    try:
        data = np.fromfile(file_path, dtype=np.complex64)
        if data.size < MIN_SAMPLE_THRESHOLD:
            return False
        for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE):
            window = data[i:i+WINDOW_SIZE]
            energy = np.mean(np.abs(window)**2)
            if energy > ENERGY_THRESHOLD:
                return True
        return False
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return False

# Scan all subfolders under the base directory
for subfolder in os.listdir(BASE_DIR):
    full_path = os.path.join(BASE_DIR, subfolder).replace("\\", "/")
    if not os.path.isdir(full_path) or "_EMPTY" in subfolder:
        continue  # Skip non-folders and already-empty folders

    # Create corresponding _EMPTY directory
    empty_path = f"{full_path}_EMPTY"
    os.makedirs(empty_path, exist_ok=True)

    # Filter only .fc32 files
    file_list = sorted(f for f in os.listdir(full_path) if f.endswith('.fc32'))
    print(f"\n▶ Scanning folder: {subfolder} ({len(file_list)} files)")

    for fname in file_list:
        fpath = os.path.join(full_path, fname).replace("\\", "/")
        if not has_valid_packet(fpath):
            print(f"   [EMPTY] {fname} → moving...")
            shutil.move(fpath, os.path.join(empty_path, fname).replace("\\", "/"))
        else:
            print(f"   [VALID] {fname}")
