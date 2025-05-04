import torch

class AverageMeter:
    """평균 값 및 현재 값 저장하는 헬퍼 클래스"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """output과 target을 받아 top-k 정확도 계산 (기본 top-1 정확도)"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # 예측 확률이 가장 높은 top-k 인덱스 추출
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size * 100.0).item())
        return res
