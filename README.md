# 졸음운전 감지

졸음 운전을 감지하는 모델 제작하고자 한다. 총 2개의 class로 구성되어있는 데이터셋을 이용해 모델을 완성하고, 분류한 결과에 따라 졸음 운전 여부에 대한 정보를 출력한다.


# 1. 개발 동기

졸음운전은 운전자의 반응속도를 급격히 저하시키며 한순간의 실수가 큰 사고로 이어지는 매우 위험한 요소이다. 특히 장거리 운전이나 야간 운행에서는 졸음 징후가 미세하게 나타나기 때문에 운전자 스스로 상태를 인지하기 어렵다. 이러한 상황은 교통사고로 직결될 가능성이 높아, 이를 사전에 감지하고 예방하는 기술의 필요성이 더욱 강조되고 있다.

이 문제를 해결하기 위해서는 컴퓨터 비전과 딥러닝 기반의 움직임·표정 분석 기술이 필수적이다. 운전자의 눈 깜빡임 패턴, 얼굴 표정 변화, 고개 움직임 등은 졸음 상태를 나타내는 핵심적인 신호이며, 딥러닝 모델은 이러한 미세한 변화를 높은 정확도로 감지하는 데 강점을 가진다. 특히 CNN 기반의 얼굴 특징 추출 모델이나 시간적 변화를 분석하는 모델은 졸음 상태 분류에 적합하다.

본 프로젝트에서는 이러한 신호를 분석하여 졸음 징후를 감지하는 딥러닝 기반 모델을 개발하였다. 이를 통해 운전자에게 위험을 조기에 알릴 수 있으며, 결과적으로 졸음운전으로 인한 사고 발생을 효과적으로 줄일 수 있을 것으로 기대된다.


---

# 2. 필요한 라이브러리

gpu = L4 GPU

colab 이용

torch = 2.0.1+cu117 torchvision = 0.15.2+cu117

[코랩에서 제공하는 라이브러리 이용]

## Tech Stack

<div align="left">

<!-- TensorFlow -->
<img src="https://img.shields.io/badge/TensorFlow-FED7AA?style=for-the-badge&logo=tensorflow&logoColor=white"/>

<!-- NumPy -->
<img src="https://img.shields.io/badge/NumPy-A7F3D0?style=for-the-badge&logo=numpy&logoColor=white"/>

<!-- OpenCV -->
<img src="https://img.shields.io/badge/OpenCV-A5B4FC?style=for-the-badge&logo=opencv&logoColor=white"/>

<!-- Matplotlib -->
<img src="https://img.shields.io/badge/Matplotlib-FBCFE8?style=for-the-badge&logo=plotly&logoColor=white"/>

</div>

### PyTorch 관련 라이브러리

from torchvision import models, transforms  
from torch.utils.data import DataLoader, Dataset  
import torch  
import torch.nn as nn  
import torchaudio  
import torch.nn.functional as F  
from torch.optim.lr_scheduler import (MultiStepLR, StepLR, CyclicLR, CosineAnnealingLR, ExponentialLR)  
from torchsummary import summary  

### 일반 라이브러리

import argparse  
import numpy as np  
import random  
import os  
from PIL import Image  
import matplotlib.pyplot as plt  
import time  
from tqdm import tqdm  
from sklearn.metrics import f1_score  
from sklearn.model_selection import train_test_split  

### 추가 설치가 필요한 라이브러리

pip install torchsummary  
pip install tqdm  
pip install scikit-learn  

# 3. 데이터셋

### 데이터셋 사용 안내

본 프로젝트는 Kaggle에서 제공하는 [**졸음운전 예측 데이터셋**](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset)을 사용하여 학습과 테스트를 진행한다.  

### 데이터 구성
- **drowsy**: 졸음 상태의 운전자 이미지
- **normal**: 정상 상태의 운전자 이미지


| 디렉토리 | 설명           | 예제 이미지 |
|----------|----------------|-------------|
| `drowsy/` | 졸음 상태 이미지 | <img src="https://github.com/user-attachments/assets/6e98a220-e440-409b-8329-d5f69d19c788" width="200"> |
| `normal/` | 정상 상태 이미지 | <img src="https://github.com/user-attachments/assets/6e2db094-f730-4113-9396-a40e28e49f40" width="200"> |


###  Data Augmentation


<img src="https://github.com/user-attachments/assets/7aae4596-44bf-4f2f-81e0-29099661755b" width="500">

---

# 4. 모델 설명 (아키텍쳐)  

<img width="332" height="230" alt="CNN_졸음운전_아키텍쳐" src="https://github.com/user-attachments/assets/24ce8b4f-f5f4-4da7-a1b7-3fa1c6c37018" />  

본 프로젝트는 운전자 얼굴 이미지를 입력으로 받아 졸음(drowsy) / 정상(normal) 을 분류하는 2D CNN 기반 이진 분류 모델을 구현하였다.
모델은 3개의 Convolution Block으로 특징을 추출한 뒤, Flatten–Fully Connected Layer를 통해 최종 확률을 출력한다.

- Input: 145 × 145 × 3 (RGB)

- Conv Blocks: 32 → 64 → 128 filters (공간 해상도는 단계적으로 축소)

- Classifier: Flatten → Dense(128, ReLU) → Dense(1, Sigmoid)

- Output: P(drowsy) ∈ [0, 1] (임계값 기준으로 클래스 결정)
---
# 5. 실험 결과 및 한계

### 1) 실험 결과  
본 모델은 운전자 얼굴 이미지를 기반으로 한 졸음 상태 이진 분류 실험에서 높은 분류 정확도와 안정적인 학습 성능을 보였다.

- **Accuracy: 95.7%**  
  → 졸음(drowsy)과 정상(normal) 상태를 높은 신뢰도로 구분함을 확인하였다.

- **Loss: 0.25**  
  → 학습 과정에서 손실 값이 안정적으로 감소하며 수렴하였고, 과적합 없이 일반화된 분류 성능을 유지하였다.

---

### 2) 한계  
본 모델의 평균 추론 시간은 이미지 1장당 약 0.15초로, 모든 프레임을 처리하는 실시간 영상 분석 시스템에 적용하기에는 제한이 있다.
다만 프레임 샘플링 기반의 졸음 경고 시스템과 같은 환경에서는 실제 적용 가능성을 확인할 수 있었다.

---
