🌈 졸음운전 탐지 시스템 🚗
운전 중 졸음으로 인한 사고를 예방하기 위한 실시간 졸음운전 탐지 시스템!
이 프로젝트는 교통사고 예방, 운전자 안전 강화, 졸음 인식 기술 개발 등 다양한 분야에서 활용될 수 있는 AI 기반 Vision 기술을 목표로 합니다.

📑 목차
프로젝트 소개
데이터셋 사용 안내
실행 방법
모델 설명
성능 향상 방법
주요 결과
프로젝트 구조
사용된 기술
🛠️ 프로젝트 소개
최근 졸음운전으로 인한 사고와 사망률이 증가하는 문제를 해결하기 위해, 운전자의 얼굴 표정을 실시간으로 분석하여 졸음 상태를 감지하는 AI 시스템을 개발했습니다.
이 시스템은 MediaPipe와 CNN을 활용하여 운전자의 상태를 "활동"과 "피로"로 분류하며, 졸음이 감지되면 즉각 경고를 제공합니다.

📂 데이터셋 사용 안내
본 프로젝트는 운전자 얼굴 이미지를 기반으로 학습과 테스트를 진행합니다.
데이터는 다음과 같이 구성되어 있습니다:

Active Subjects (활동 상태): 운전 중 깨어 있는 상태의 이미지
Fatigue Subjects (피로 상태): 졸음이 감지된 상태의 이미지
📥 데이터 준비 방법
1️⃣ 데이터 수집 및 전처리

운전자의 얼굴 이미지를 캡처하고 전처리합니다.
이미지 크기를 145x145 픽셀로 조정하고 정규화합니다.
2️⃣ 데이터 디렉토리 구성

Active Subjects와 Fatigue Subjects로 데이터를 분류하여 저장합니다.
디렉토리 구조는 다음과 같습니다:
bash
코드 복사
/content/drive/MyDrive/0_FaceImages/
    ├── Active Subjects/
    └── Fatigue Subjects/
💻 실행 방법
1️⃣ 저장소 클론

bash
코드 복사
git clone https://github.com/your-username/DrowsyDriver-Detection.git
2️⃣ 필수 패키지 설치

bash
코드 복사
pip install -r requirements.txt
3️⃣ 프로젝트 실행

bash
코드 복사
python main.py
🔍 모델 설명
본 프로젝트에서는 CNN (합성곱 신경망)을 활용하여 졸음 상태를 분류합니다.
모델의 주요 구성은 다음과 같습니다:

4개의 합성곱층: 랜드마크 특징 추출
드롭아웃: 과적합 방지를 위한 정규화
완전 연결층: 졸음 여부를 분류
최종 출력: 활성화 함수 Sigmoid를 사용해 이진 분류 수행
🧠 성능 향상 방법
데이터 증강: 회전, 확대, 반전 등 변환 기법으로 데이터 다양성 확보
하이퍼파라미터 튜닝: 학습률, 배치 크기 최적화
얼굴 랜드마크 분석: MediaPipe를 이용해 정확도 개선
🎯 주요 결과
정확도: 95.7%
손실 값: 0.25
📂 프로젝트 구조
css
코드 복사
DrowsyDriver-Detection/
├── data/
│   ├── Active Subjects/
│   └── Fatigue Subjects/
├── models/
├── scripts/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── main.py
├── README.md
└── requirements.txt
🛠️ 사용된 기술
Python
TensorFlow
OpenCV
MediaPipe
Matplotlib
