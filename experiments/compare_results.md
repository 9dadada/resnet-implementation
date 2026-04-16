# PyTorch 공식 ResNet18 vs 내 구현 비교 실험

## 실험 목적

PyTorch 공식 ResNet18은 ImageNet(224x224)용 설계다. CIFAR-10(32x32)에서 내 구현과 어떤 성능 차이가 나는지 확인한다.

## 실험 조건

| 항목 | 설정 |
|------|------|
| 데이터셋 | CIFAR-10 |
| Batch Size | 128 |
| Optimizer | SGD (momentum=0.9, weight_decay=1e-4) |
| Learning Rate | 0.1, CosineAnnealingLR |
| Epochs | 30 |

## 핵심 차이점

| | 내 구현 | PyTorch 공식 |
|---|---|---|
| conv1 | 3x3, stride 1 | 7x7, stride 2 |
| maxpool | 없음 | 3x3, stride 2 |
| conv1 후 특징맵 크기 | 32x32 | 8x8 |

PyTorch 공식 모델은 32x32 이미지에 stride 2 conv + stride 2 maxpool을 적용하면 특징맵이 8x8까지 줄어든다. 이후 layer2~4에서 추가 다운샘플링이 일어나면 1x1까지 축소되어 정보가 크게 손실된다.

## 실험 결과

(Colab 실험 후 결과 추가 예정)

## 실행 방법

```bash
python experiments/compare_resnet.py
```
