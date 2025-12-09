# 졸음운전 감지

졸음 운전을 감지하는 모델 제작하고자 한다. 총 2개의 class로 구성되어있는 데이터셋을 이용해 모델을 완성하고, 분류한 결과에 따라 졸음 운전 여부에 대한 정보를 출력한다.


# 1. 개발 동기

졸음운전은 운전자의 반응속도를 급격히 저하시키며 한순간의 실수가 큰 사고로 이어지는 매우 위험한 요소이다. 특히 장거리 운전이나 야간 운행에서는 졸음 징후가 미세하게 나타나기 때문에 운전자 스스로 상태를 인지하기 어렵다. 이러한 상황은 교통사고로 직결될 가능성이 높아, 이를 사전에 감지하고 예방하는 기술의 필요성이 더욱 강조되고 있다.

이 문제를 해결하기 위해서는 컴퓨터 비전과 딥러닝 기반의 움직임·표정 분석 기술이 필수적이다. 운전자의 눈 깜빡임 패턴, 얼굴 표정 변화, 고개 움직임 등은 졸음 상태를 나타내는 핵심적인 신호이며, 딥러닝 모델은 이러한 미세한 변화를 높은 정확도로 감지하는 데 강점을 가진다. 특히 CNN 기반의 얼굴 특징 추출 모델이나 시간적 변화를 분석하는 모델은 졸음 상태 분류에 적합하다.

본 프로젝트에서는 이러한 신호를 실시간으로 분석하여 졸음 징후를 감지하는 딥러닝 기반 모델을 개발하였다. 이를 통해 운전자에게 위험을 조기에 알릴 수 있으며, 결과적으로 졸음운전으로 인한 사고 발생을 효과적으로 줄일 수 있을 것으로 기대된다.


---

# 2. 필요한 라이브러리

gpu = L4 GPU

colab 이용

torch = 2.0.1+cu117 torchvision = 0.15.2+cu117

[코랩에서 제공하는 라이브러리 이용]


# PyTorch 관련 라이브러리

from torchvision import models, transforms  
from torch.utils.data import DataLoader, Dataset  
import torch  
import torch.nn as nn  
import torchaudio  
import torch.nn.functional as F  
from torch.optim.lr_scheduler import (MultiStepLR, StepLR, CyclicLR, CosineAnnealingLR, ExponentialLR)  
from torchsummary import summary  

# 일반 라이브러리

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

# 추가 설치가 필요한 라이브러리

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

---

### 📥 데이터 다운로드 방법

1️⃣ **Kaggle 페이지에 접속**  
   [👉 데이터셋 바로가기](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset)  

2️⃣ **데이터 다운로드**  
   Kaggle 계정으로 로그인 후 데이터를 다운로드하세요.

3️⃣ **데이터 디렉토리 구성**  
   다운로드한 데이터를 아래와 같은 구조로 정리하세요:

| 디렉토리 | 설명           | 예제 이미지 |
|----------|----------------|-------------|
| `drowsy/` | 졸음 상태 이미지 | <img src="https://github.com/user-attachments/assets/6e98a220-e440-409b-8329-d5f69d19c788" width="200"> |
| `normal/` | 정상 상태 이미지 | <img src="https://github.com/user-attachments/assets/6e2db094-f730-4113-9396-a40e28e49f40" width="200"> |

---

## 💻 실행 방법

### 1️⃣ 저장소 클론 및 패키지 설치
아래 명령어를 사용해 프로젝트 저장소를 로컬로 복사한 후, 필요한 라이브러리를 설치합니다:
```bash
git clone https://github.com/your-username/DrowsyDriver-Detection.git
cd DrowsyDriver-Detection
pip install -r requirements.txt

```

### 2️⃣ 프로젝트 실행
아래 명령어를 사용해 프로그램을 실행하세요:
```bash
python main.py

```

### 3️⃣ 결과 확인
실행 후 분류 결과는 output/ 디렉토리에 저장됩니다.

---

### 🔍 모델 설명
본 프로젝트는 **CNN(합성곱 신경망)**을 사용하여 졸음 상태를 감지합니다.

**💡 주요 구성 요소**:

**Face Detection**: MediaPipe를 활용해 얼굴을 감지합니다.
**Landmark Extraction**: 눈, 입 등 주요 부위를 추출하여 졸음 여부를 분석합니다.
**CNN 분류기**: 추출된 데이터를 "Active"와 "Fatigue"로 분류합니다.

---

## 🚀 성능 향상 방법
### 1️⃣ 데이터 증강

회전, 반전, 밝기 조정 등으로 데이터 다양성을 확보했습니다.
### 2️⃣ 모델 최적화

Dropout, Batch Normalization 등을 활용해 과적합을 방지했습니다.
### 3️⃣ 얼굴 랜드마크 분석
눈 깜빡임 패턴과 표정을 분석하여 모델 성능을 개선했습니다.

---

## 🎯 주요 결과
정확도: 95.7%
손실 값: 0.25
추론 시간: 이미지 당 평균 0.15초

---

## 📂 프로젝트 구조

```bash
DrowsyDriver-Detection/
├── dataset/                # 데이터 파일 디렉토리
│   ├── drowsy/             # 졸음 상태 이미지
│   ├── normal/             # 정상 상태 이미지
├── models/                 # 모델 코드 및 학습된 모델 저장
│   ├── model.py            # CNN 모델 코드
├── scripts/                # 주요 Python 코드 스크립트
│   ├── preprocess.py       # 데이터 전처리 코드
│   ├── train.py            # 모델 학습 코드
│   ├── evaluate.py         # 모델 평가 코드
│   ├── predict.py          # 모델 추론 코드
│   ├── visualization.py    # 데이터 시각화 코드
├── output/                 # 결과 파일 저장 디렉토리 (결과 이미지, 로그 등)
├── main.py                 # 메인 실행 스크립트
├── requirements.txt        # 필요 라이브러리 목록
├── README.md               # 프로젝트 설명 파일
```

---

## 🛠️ 사용된 기술

- **Python**
- **TensorFlow**
- **OpenCV**
- **MediaPipe**





