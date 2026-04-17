# ResNet Implementation

ResNet 논문을 읽고 PyTorch로 직접 구현한 프로젝트입니다.
CIFAR-10 데이터셋으로 학습하여 이미지 분류를 수행합니다.

## 프로젝트 구조

```
├── model/resnet.py              # ResNet18 모델 구현
├── train.py                     # 학습 스크립트
├── inference.py                 # 추론 스크립트
├── experiments/
│   ├── compare_resnet.py        # PyTorch 공식 ResNet18과 비교 실험
│   └── compare_results.md       # 비교 실험 결과
├── notebooks/
│   └── resnet_train.ipynb       # Kaggle 학습 노트북
├── results/                     # 학습 결과 (그래프, 모델 등)
└── requirements.txt
```

## 논문 리뷰

논문 정리는 [여기](https://github.com/9dadada/AI-study/blob/main/papers/Resnet/resnet_review.md)에서 확인할 수 있습니다.

## 실행 방법

### Kaggle Notebook (GPU)
`notebooks/resnet_train.ipynb`을 Kaggle Notebook에서 열어 실행

### 로컬
```bash
pip install -r requirements.txt
python train.py
```

## 학습 결과

(학습 결과 추가 예정)

## Reference

- [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
