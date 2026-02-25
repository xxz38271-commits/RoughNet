# RoughNet 🔬
**CNN 기반 밀링 가공면 표면거칠기 실시간 예측 자동화 시스템**

> 한국정밀공학회 창의경진대회 **최우수상** (2024.11.15)  
> 14개 참가팀 중 1위

---

## 프로젝트 소개

기존 접촉식 표면거칠기 측정 장비는 측정 속도가 느리고, 센서 마모와 접촉 오류 문제가 있어 현장 적용에 한계가 있었습니다.  
이를 해결하기 위해 밀링 가공면의 이미지만으로 표면거칠기(Ra)를 실시간으로 예측하는 **비접촉식 CNN 회귀 모델**을 개발했습니다.

| 구분 | 기존 방식 | 본 시스템 |
|------|-----------|-----------|
| 측정 방식 | 접촉식 | 비접촉식 (이미지) |
| 측정 속도 | 느림 | 실시간 |
| 센서 마모 | 발생 | 없음 |
| 자동화 | 어려움 | 가능 |

---

## 시스템 구조

```
금속 표면 촬영 (자동 이동 스테이지)
        ↓
이미지 전처리
  - 그레이스케일 변환
  - 0° / 120° / 240° 회전
  - 224×224 center crop
  - Laplacian 필터링 (경계 강조)
  - CLAHE 적용 (명암 대비 개선)
        ↓
이미지 증강 (수평 / 수직 / 양방향 flip → 총 12장)
        ↓
CNN 회귀 모델 추론 (ONNX → C++ 실시간 처리)
        ↓
소수점 첫째 자리 반올림 후 최빈값 평균 → 최종 Ra 출력
```

---

## 핵심 결과

| 손실 함수 | 예측 정확도 (ARa) | R² |
|-----------|-------------------|----|
| MSE | 89.75% | 0.940 |
| MAE | 90.53% | 0.955 |
| **Huber** | **91.72%** | **0.957** |
| Log-Cosh | 88.74% | 0.943 |

→ **Huber 손실 함수 적용 모델이 최고 성능** (ARa 91.72%, R² 0.957)

---

## 기술 스택

![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

- **추론 엔진**: ONNX Runtime (C++17) — PyTorch 학습 모델을 ONNX로 변환하여 C++ 환경에 통합
- **이미지 처리**: OpenCV — Laplacian Filter, CLAHE, 회전/반전 증강
- **딥러닝**: PyTorch, CNN Regression (6 Conv + 2 FC layers)
- **빌드**: CMake 3.10+, C++17
- **하드웨어**: 머신비전 카메라(DFK23GM021), 자동 이송 스테이지(PMAC)
- **가공 장비**: 밀링머신, CNC (AL6061, AL7075 시편 직접 가공)

---

## 데이터셋

- 알루미늄 합금(AL6061, AL7075)을 다양한 가공 조건으로 밀링
  - 이송 속도: 50~550 mm/min
  - 회전 속도: 5,000~20,000 RPM
  - 절삭 깊이: 0.1~0.3 mm
- 전처리 및 증강 후 총 **40,000장** 이미지 확보
  - Train: 32,000 / Validation: 4,000 / Test: 4,000

---

## 모델 구조

```
Input (224×224×3)  ← ImageNet mean/std 정규화 적용
→ Conv2d(32)  → AvgPool → ReLU
→ Conv2d(64)  → AvgPool → ReLU
→ Conv2d(128) → AvgPool → ReLU
→ Conv2d(256) → AvgPool → ReLU
→ Conv2d(512) → AvgPool → ReLU
→ Conv2d(1024)→ AvgPool → ReLU
→ Flatten
→ FC(1024) → Dropout(0.5)
→ FC(512)
→ Output: Ra 값 (회귀)
```

**학습 설정**: Adam (lr=1e-4), Batch 32, Epoch 60  
**배포**: PyTorch 학습 → ONNX 변환 → C++ ONNX Runtime으로 실시간 추론

---

## 예측 방식

이미지 1장을 입력받아 아래 과정으로 최종 Ra 값을 산출합니다.

1. 전처리 3방향 (0°/120°/240°) × 증강 4방향 (원본/수평flip/수직flip/양방향flip) → 총 **12장** 생성
2. 12장 각각 ONNX 모델로 추론 → 12개의 Ra 예측값
3. 소수점 첫째 자리로 반올림 후 **최빈값** 선택
4. 최빈값에 해당하는 예측값들의 **평균** → 최종 Ra 출력

---

## 빌드 및 실행

**의존성**
- OpenCV
- ONNX Runtime
- CMake 3.10+, C++17

```bash
# 빌드
mkdir build && cd build
cmake ..
cmake --build . --config Release

# 실행 (예측 이미지를 saved_image.jpg로 저장 후 실행)
./ONNX_Image_Prediction
```

---

## 파일 구조

```
RoughNet/
├── module.cpp        # C++ 추론 코드 (전처리 + 증강 + ONNX 추론)
├── CMakeLists.txt    # CMake 빌드 설정
├── model.onnx        # 학습된 CNN 모델 (Huber loss, ARa 91.72%)
└── README.md
```

---

## 팀 구성 및 역할

| 이름 | 역할 |
|------|------|
| 유서진 | CNN 모델 설계 및 학습 (PyTorch) |
| **전형준** | C++ 추론 시스템 구축, 현미경 자동 이동 촬영 시스템 개발, 금속 시편 직접 가공 |
| 박원찬 | 표면 표본 수집 및 Ra 측정 |

---

## 논문

> 유서진, 박원찬, 전형준, *"인공지능(CNN)을 활용하여 밀링 가공된 재료의 표면 거칠기 실시간 예측 자동화 시스템 개발"*, 한국정밀공학회 (2024)
