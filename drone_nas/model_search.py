import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS, PRIMITIVES, ReLUConvBN, FactorizedReduce
from genotypes import Genotype

# 혼합 연산 클래스: 한 edge에서 여러 후보 연산을 가중합으로 수행
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        # 모든 primitive 연산에 대한 모듈 생성
        self.ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)  # 탐색 단계에서는 BN affine을 False로 설정
            self.ops.append(op)
    def forward(self, x, weights):
        # weights 텐서가 각 연산에 대한 가중치(softmax 결과)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

# Neural Architecture Search용 Cell 클래스 (normal 또는 reduction)
class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier
        # 이전 두 개 출력에 대한 사전 처리
        if reduction_prev:
            # 이전 셀이 reduction이었다면 그 이전 출력은 해상도/채널 맞춤 필요
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        # 직전 출력은 항상 1x1 conv로 채널수 맞춤
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        # MixedOp 모듈 리스트 생성 (각 edge마다 하나씩)
        self._ops = nn.ModuleList()
        # 각 intermediate node에 대해 가능한 모든 입력(edge) 처리
        # 노드 인덱스: 0,1은 셀의 입력(prev_prev, prev), 이후 2~(1+steps)가 내부 노드
        edge_index = 0
        for i in range(self.steps):
            # 노드 i+2에 연결되는 모든 이전 노드로부터 edge 생성
            for j in range(i + 2):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights):
        # s0: 두 단계 이전 출력, s1: 바로 이전 출력
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        # 각 intermediate node에 대해 weighted sum 계산
        for i in range(self.steps):
            res = 0
            # node i+2의 출력 = 모든 입력 노드들의 op 결과 가중합
            for j in range(i + 2):
                res = res + self._ops[offset + j](states[j], weights[offset + j])
            states.append(res)
            offset += i + 2
        # cell의 결과: 마지막 multiplier 개수의 intermediate 노드 출력들을 concat
        return torch.cat(states[-self.multiplier:], dim=1)

class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion=None, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C  # 초기 채널 수
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        # 초기 stem 정의: 입력 채널(1) -> C*stem_multiplier 채널로 conv
        input_channels = 2
        C_cur = C * stem_multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C_cur, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # 모델의 cell 생성
        self.cells = nn.ModuleList()
        C_prev_prev = C_cur  # 두 단계 전 출력 채널
        C_prev = C_cur       # 직전 출력 채널
        reduction_prev = False
        reduction_layers = [layers // 3, 2 * layers // 3]  # reduction 적용할 layer 인덱스
        for i in range(layers):
            if i in reduction_layers:
                C_curr = C * 2
                reduction = True
            else:
                C_curr = C
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            C_prev_prev = C_prev
            C_prev = C_curr * multiplier
            if reduction:
                C = C * 2

        # 클래스 분류를 위한 global pooling과 linear classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # 아키텍처 파라미터(alpha) 초기화
        k = 0
        for i in range(self._steps):
            k += i + 2
        num_ops = len(PRIMITIVES)
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def weights(self):
        # 모든 파라미터 중 alphas를 제외한 가중치들 반환
        return [param for name, param in self.named_parameters() if "alpha" not in name]

    def forward(self, x):
        # stem 처리
        s0 = self.stem(x)
        s1 = s0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def genotype(self):
        """현재 학습된 알파 값으로부터 discrete genotype (건축)을 추출"""
        # normal과 reduce 세트에 대해 각각 최상의 2개 연결 추출
        gene_normal = []
        gene_reduce = []
        for alphas, gene in [(self.alphas_normal, gene_normal), (self.alphas_reduce, gene_reduce)]:
            weights = alphas.data.cpu().numpy()
            n = 0
            for i in range(self._steps):
                offset = i + 2
                edge_candidates = []
                for j in range(offset):
                    # 'none' 연산의 alpha는 매우 작은 값으로 취급하여 제외
                    ops_weights = weights[n + j]
                    # PRIMITIVES[0] == 'none'
                    idx_best = np.argmax(ops_weights[1:]) + 1  # 'none' 제외한 최댓값 index (1부터 시작)
                    op_name = PRIMITIVES[idx_best]
                    edge_candidates.append((op_name, j, ops_weights[idx_best]))
                # 가중치 기준 상위 2개의 edge 선택
                edge_candidates.sort(key=lambda x: x[2], reverse=True)
                top2 = edge_candidates[:2]
                gene.extend([(op, idx) for op, idx, _ in top2])
                n += offset
        normal_concat = list(range(2, 2 + self._steps))
        reduce_concat = list(range(2, 2 + self._steps))
        return Genotype(normal=gene_normal, normal_concat=normal_concat,
                        reduce=gene_reduce, reduce_concat=reduce_concat)
