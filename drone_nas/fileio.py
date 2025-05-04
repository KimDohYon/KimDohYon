import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WaveformDataset(Dataset):
    """
    드론 IQ 데이터를 불러오는 Dataset 클래스.
    지정한 폴더 내 .bin 파일들을 읽어 (1, 2, 32, 64) 모양의 텐서와 레이블을 생성.
    """
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        # 각 클래스별 하위 디렉토리를 순회하며 .bin 파일 목록 구성
        for class_name in sorted(os.listdir(root_dir)):  # '00', '01', ..., '11'
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = int(class_name)  # 클래스 디렉토리명을 정수 레이블로 변환
            file_list = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.bin')]
            file_list.sort()
            self.files.extend(file_list)
            self.labels.extend([label] * len(file_list))
        # 파일 목록과 레이블 목록 길이가 같은지 확인
        assert len(self.files) == len(self.labels), "파일 목록과 레이블 목록의 길이가 일치하지 않습니다."

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # .bin 파일을 읽어 IQ 신호 텐서를 생성
        file_path = self.files[idx]
        label = self.labels[idx]
        # 이진 파일에서 int16 값들을 읽어 numpy 배열로 변환
        data = np.fromfile(file_path, dtype=np.int16)

        try:
            iq_pairs = data.reshape(-1, 2)  # (2048, 2)
        except:
            raise ValueError(f"파일 {file_path}의 크기가 예상과 다릅니다.")

        iq_pairs = iq_pairs.astype(np.float32)
        iq_tensor = torch.from_numpy(iq_pairs).T.view(2, 32, 64)  # 최종 shape: (2, 32, 64)
        return iq_tensor, label

