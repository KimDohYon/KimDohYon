import os
import torch
import numpy as np
import argparse
from fileio import WaveformDataset
from genotypes import NASGenotype
from operations import OPS, ReLUConvBN, FactorizedReduce, PRIMITIVES
import torch.nn as nn

# 최종 탐색된 architecture로 모델 구성
class NetworkFixed(nn.Module):
    def __init__(self, genotype, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
        super(NetworkFixed, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        input_channels = 1  # 입력 데이터는 (1, 2048, 2) -> 채널 1개
        C_cur = C * stem_multiplier
        # stem: 입력 채널 -> C_cur 채널
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C_cur, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        # 셀 생성
        self.cells = nn.ModuleList()
        C_prev_prev = C_cur
        C_prev = C_cur
        reduction_prev = False
        reduction_layers = [layers // 3, 2 * layers // 3]
        for i in range(layers):
            if i in reduction_layers:
                C_curr = C * 2
                reduction = True
            else:
                C_curr = C
                reduction = False
            cell_genotype = genotype.reduce if reduction else genotype.normal
            cell = Cell(genotype=cell_genotype, steps=self._steps, multiplier=self._multiplier,
                        C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr,
                        reduction=reduction, reduction_prev=reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            C_prev_prev = C_prev
            C_prev = C_curr * multiplier
            if reduction:
                C = C * 2

        # Classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = s0
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

class Cell(nn.Module):
    def __init__(self, genotype, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        # 두 입력에 대한 전처리
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
        # genotype 리스트에서 노드별 두 개의 연결 정보 추출
        self._steps = steps
        self._multiplier = multiplier
        self.ops_by_node = nn.ModuleList()
        self.indices_by_node = []
        for i in range(steps):
            op_name1, inp1 = genotype[2 * i]
            op_name2, inp2 = genotype[2 * i + 1]
            stride1 = 2 if reduction and inp1 < 2 else 1
            stride2 = 2 if reduction and inp2 < 2 else 1
            op1 = OPS[op_name1](C, stride1, affine=True)
            op2 = OPS[op_name2](C, stride2, affine=True)
            self.ops_by_node.append(nn.ModuleList([op1, op2]))
            self.indices_by_node.append((inp1, inp2))

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for ops, (inp1, inp2) in zip(self.ops_by_node, self.indices_by_node):
            h1 = ops[0](states[inp1])
            h2 = ops[1](states[inp2])
            s = h1 + h2
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

def main():
    parser = argparse.ArgumentParser("DRONE-IQ DARTS Inference")
    parser.add_argument('--train_dir', type=str, default='data/train', help='훈련 데이터 디렉토리')
    parser.add_argument('--test_dir', type=str, default='data/test', help='테스트 데이터 디렉토리')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=50, help='학습 epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='초기 채널 수')
    parser.add_argument('--layers', type=int, default=8, help='총 셀(레이어) 수')
    parser.add_argument('--classes', type=int, default=12, help='클래스 수')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='학습률')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD 모멘텀')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='가중치 감쇠 계수')
    parser.add_argument('--threads', type=int, default=4, help='CPU 사용 스레드 수')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    # 데이터 로드
    train_data = WaveformDataset(args.train_dir)
    test_data = WaveformDataset(args.test_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 모델 및 손실함수
    genotype = NASGenotype  # 탐색된 genotype
    model = NetworkFixed(genotype, C=args.init_channels, num_classes=args.classes, layers=args.layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 학습 loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = train_loss / total
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%")

    # 테스트 세트 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # ONNX 모델 저장
    dummy_input = torch.randn(1, 1, 2048, 2, device=device)
    model_path = "model.onnx"
    torch.onnx.export(model, dummy_input, model_path, opset_version=11)
    print(f"ONNX 모델이 '{model_path}' 로 저장되었습니다.")

if __name__ == "__main__":
    main()
