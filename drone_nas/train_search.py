import os
import torch
import argparse
from torch.utils.data import random_split
from fileio import WaveformDataset
from model_search import Network
from architect import Architect
from utils import AverageMeter, accuracy

def main():
    parser = argparse.ArgumentParser("DRONE-IQ DARTS Search")
    parser.add_argument('--train_dir', type=str, default='data/train', help='학습용 데이터 디렉토리')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='weights SGD 학습률')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD 모멘텀')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weights에 대한 가중치 감쇠')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='arch 파라미터 학습률')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='arch 파라미터 가중치 감쇠')
    parser.add_argument('--epochs', type=int, default=50, help='아키텍처 탐색 epoch 수')
    parser.add_argument('--init_channels', type=int, default=16, help='네트워크 초기 채널 수')
    parser.add_argument('--layers', type=int, default=8, help='탐색 시 네트워크의 총 layer(cell) 수')
    parser.add_argument('--classes', type=int, default=12, help='클래스 개수')
    parser.add_argument('--train_portion', type=float, default=0.8, help='train 데이터 중 가중치 학습에 사용할 비율 (나머지는 아키텍처 검증용)')
    parser.add_argument('--unrolled', action='store_true', help='(고급) unrolled bilevel 최적화 사용 여부')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping 값')
    parser.add_argument('--threads', type=int, default=4, help='CPU 사용 스레드 수')
    args = parser.parse_args()

    # CPU 세팅
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')

    # 데이터셋 로드 및 분할 (train/validation)
    dataset = WaveformDataset(args.train_dir)
    train_size = int(len(dataset) * args.train_portion)
    valid_size = len(dataset) - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 모델, 손실함수, 아키텍처관리자 설정
    model = Network(args.init_channels, args.classes, args.layers)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    architect = Architect(model, criterion, args.arch_learning_rate, args.arch_weight_decay)
    optimizer = torch.optim.SGD(model.weights(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        valid_loss_meter = AverageMeter()
        valid_acc_meter = AverageMeter()

        # validation 데이터 반복자 (아키텍처 업데이트용)
        valid_iter = iter(valid_loader)
        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 1. weight 파라미터 한 스텝 학습
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.weights(), args.grad_clip)
            optimizer.step()
            # 학습 loss/정확도 기록
            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            train_loss_meter.update(loss.item(), targets.size(0))
            train_acc_meter.update(prec1, targets.size(0))
            # 2. arch 파라미터 한 스텝 학습
            try:
                input_val, target_val = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_loader)
                input_val, target_val = next(valid_iter)
            input_val = input_val.to(device)
            target_val = target_val.to(device)
            architect.step(inputs, targets, input_val, target_val, args.unrolled)

        # epoch 종료 후 validation 세트로 아키텍처 성능 평가
        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                prec1 = accuracy(outputs, targets, topk=(1,))[0]
                valid_loss_meter.update(val_loss.item(), targets.size(0))
                valid_acc_meter.update(prec1, targets.size(0))
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss_meter.avg:.3f}, Train Acc: {train_acc_meter.avg:.2f}% | "
              f"Val Loss: {valid_loss_meter.avg:.3f}, Val Acc: {valid_acc_meter.avg:.2f}%")
        # 현재 epoch에서의 genotype 출력
        genotype = model.genotype()
        print(f"Genotype: {genotype}")

    # 탐색 완료: 최종 genotype 출력
    final_genotype = model.genotype()
    print(f"최종 검색된 Genotype: {final_genotype}")
    print("genotypes.py 파일의 NASGenotype 변수에 위 genotype 값을 복사하여 저장하세요.")

if __name__ == "__main__":
    main()
