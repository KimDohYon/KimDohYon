import torch

class Architect:
    def __init__(self, model, criterion, arch_lr, arch_weight_decay):
        """
        아키텍처 파라미터(알파)를 업데이트하는 관리 클래스.
        """
        self.model = model
        self.criterion = criterion
        # 아키텍처 파라미터만 Adam optimizer로 관리
        self.optimizer = torch.optim.Adam(model.arch_parameters(), lr=arch_lr, betas=(0.5, 0.999), weight_decay=arch_weight_decay)

    def step(self, input_train, target_train, input_valid, target_valid, unrolled=False):
        """
        아키텍처 파라미터 한 스텝 업데이트 수행.
        unrolled=True 인 경우 2차 미분(unrolled optimization)을 사용할 수 있으나,
        여기서는 기본 1차 미분(First-order) 방식으로 구현.
        """
        # 아키텍처 파라미터 그라디언트 초기화
        self.optimizer.zero_grad()
        # 모델의 기존 기울기도 초기화하여 학습 파라미터 기울기 누적 방지
        self.model.zero_grad()
        # 검증 세트로 forward pass 및 loss 계산
        was_training = self.model.training
        self.model.eval()
        logits_valid = self.model(input_valid)
        loss_valid = self.criterion(logits_valid, target_valid)
        # 아키텍처 파라미터에 대한 그라디언트 계산 (weights에 대한 그라디언트도 함께 계산되지만 사용하지 않음)
        loss_valid.backward()
        # 아키텍처 파라미터 업데이트
        self.optimizer.step()
        # 모델 모드 원복
        if was_training:
            self.model.train()
