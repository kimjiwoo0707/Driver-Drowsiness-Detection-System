# 🚗 졸음운전 탐지 시스템 🌙

운전 중 졸음으로 인한 사고를 예방하기 위한 실시간 졸음운전 탐지 AI!  
이 프로젝트는 **교통사고 예방**, **운전자 안전 강화**, **졸음 인식 기술 개발** 등 다양한 분야에서 활용될 수 있는 Vision AI 기술을 목표로 합니다.

---

# 📑 목차
1. [프로젝트 소개](#-졸음운전-탐지-시스템-)
2. [데이터셋 사용 안내](#-데이터셋-사용-안내)
3. [실행 방법](#-실행-방법)
4. [모델 설명](#-모델-설명)
5. [성능 향상 방법](#-성능-향상-방법)
6. [주요 결과](#-주요-결과)
7. [프로젝트 구조](#-프로젝트-구조)
8. [사용된 기술](#-사용된-기술)

---

## 🛠️ 프로젝트 소개

졸음운전으로 인한 교통사고와 사망률이 증가하고 있는 상황에서, 이를 효과적으로 예방할 수 있는 기술적 솔루션이 필요합니다.  
본 프로젝트는 **운전자의 얼굴 표정을 실시간으로 분석하여 졸음 상태를 감지**하고, 졸음이 감지된 경우 즉각 경고를 제공하는 AI 시스템을 개발하는 것을 목표로 합니다.

---

## 📂 데이터셋 사용 안내

본 프로젝트는 Kaggle에서 제공하는 [**졸음운전 예측 데이터셋**](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset)을 사용하여 학습과 테스트를 진행합니다.  

### 📥 데이터 구성
- **drowsy**: 졸음 상태의 운전자 이미지
- **normal**: 정상 상태의 운전자 이미지

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

DrowsyDriver-Detection/
├── dataset/
│   ├── drowsy/          # 졸음 상태 이미지
│   ├── normal/          # 정상 상태 이미지
├── models/
│   ├── model.py         # CNN 모델 코드
├── scripts/
│   ├── preprocess.py    # 데이터 전처리 코드
│   ├── train.py         # 모델 학습 코드
│   └── evaluate.py      # 모델 평가 코드
├── output/              # 결과 파일 저장 디렉토리
├── main.py              # 메인 실행 스크립트
├── requirements.txt     # 라이브러리 목록
└── README.md            # 리드미 파일

---

## 🛠️ 사용된 기술
Python
TensorFlow
OpenCV
MediaPipe




