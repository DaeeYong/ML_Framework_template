# ML_Framework_template
딥러닝 모델을 연구 및 개발할 때, 체계적이지 않은 방식을 사용하게 되면
프로젝트의 규모가 커졌을 때 각 기능들과 실행 부분들이 제대로 분리가 되지 않은 경우에 문제가 발생합니다.
나중에 코드를 볼 때, 
필요한 부분을 찾기까지 너무 많은 시간이 걸리게 되는 것을 경험하였습니다.  
이러한 불편 사항을 방지하기 위해서 전체적인 개발 프레임워크 템블릿을 만들었습니다.  
해당 템플릿을 이용하면 개발의 `로직 부분(/src)`과 `실행 부분(/scripts)`이 분리가 되어 있기 때문에 효율적인 코드 관리가 가능합니다.

## hierarchy structure
```
ML_Framework_template/
├── checkpoints/      # 최종적으로 사용할 모델의 가중치 저장 폴더
│   └── final_model.pt
│   └── 공통_체크_포인트.pt
├── data/
│   ├── raw/          # 원본 데이터
│   ├── processed/    # 전처리된 데이터
│   └── ...
├── src/
│   ├── data/         # 데이터 전처리 및 로더 관련 코드(전처리)
│   ├── models/       # 모델 정의 및 모델 관련 유틸(모델)
│   ├── training/     # 학습 루프, 손실함수, 옵티마이저 설정, 훈련 스크립트 등(학습)
│   ├── evaluation/   # 평가 및 지표 계산 코드(평가)
│   └── utils/        # 공통 유틸 함수(유틸)
├── configs/
│   └── config.yaml   # # (하이퍼파라미터, 경로 설정 등)(하이퍼파라미터, 경로 등)
├── scripts/
│   ├── train.py      # 전체 학습 실행 스크립트
│   ├── eval.py       # 평가 실행 스크립트
│   └── inference.py  # 추론 실행 스크립트
├── tests/
│   ├── test_data.py  # 데이터 전처리/로더 단위 테스트
│   ├── test_models.py
│   └── ...
├── experiments/
│   ├── experiment_1/
│   │   ├── config.yaml
│   │   ├── logs/
│   │   │   └── training.log
│   │   ├── figures/
│   │   │   └── loss_curve.png
│   │   └── checkpoints/
│   │       ├── epoch_10.pt
│   │       ├── epoch_20.pt
│   │       └── best_model.pt
│   ├── experiment_2/
│   │   ├── config.yaml
│   │   ├── logs/
│   │   │   └── training.log
│   │   ├── figures/
│   │   │   └── loss_curve.png
│   │   └── checkpoints/
│   │       ├── epoch_5.pt
│   │       ├── epoch_15.pt
│   │       └── best_model.pt
│   └── ...
├── requirements.txt
├── README.md
└── .gitignore
```

### checkpoints/
최종적인 모델의 가중치를 저장하는 폴더 입니다. 모든 실험에서 공통적으로 사용하는 모델의 가중치를 저장하는 역할도 합니다.
### data/
모델 학습에 필요한 데이터를 위치하는 폴더 입니다. `raw/` 에 전처리 전 데이터를, `processed/` 에 전처리 후 최종 데이터가 들어가게 됩니다.
### src/
- `data/` 
  - 데이터 전처리 및 로더 관련 코드가 들어가는 폴더 입니다.
  - `dataset.py` 에서 데이터의 형태를 정의하고, `dataloader.py` 를 통해서 정의한 데이터를 사용하게 됩니다.
- `models/`
  - 모델 클래스를 정의하는 공간입니다.
  - `{your_model_name}.py` 와 같은 형태로 파일을 만드시면 됩니다.
  - 모델 클래스 1개당 `.py` 파일 하나 입니다.
  
나머지 작성중...

### configs/

### scripts/

### tests/

### experiments/



## Flow
### 구현 로직
전처리(data) -> 모델(models) -> 학습(training) -> 평가(evaluation)

### 실행 로직
script에서 하면 됨.

### summary
1. 데이터 전처리/로딩
2. 모델 정의
   1. 여러개의 모델을 만들 경우 src/models 안에 여러 파일로 나누어 관리하는 것을 추천
3. 학습 로직
      1. train_loop.py에서 epoch마다 학습하고, loss를 추적
4. 평가 로직
   1. evalute.py에서 검증/테스트 단계의 손실, 정확도를 계산
   2. 다른 지표를 계산하고 싶다면 metrics.py에 함수를 추가해서 쓰면 됨.
5. 실행 스크립트
   1. train.py에서 모델 학습 + 검증
   2. eval.py에서 모델 학습이 잘 되어있는지 테스트

### 추가 팁
- 전처리 로직만 바꿀 때는 dataset.py만 고치면 됨.
- 다른 모델 아키텍처로 실험하고 싶으면 train.py의 import만 수정하면 됨.
- config.yaml 파일 쓰면 편함.
  - 하이퍼 파라미터 같은 게 하드코딩 되어 있으면 유지 보수 어려움.
- tests/ 폴더에서 작성한 코드 테스트
  - e.g.
  - test_dataset.py
  - test_model.py
  
### 실제 예시
1. 데이터 준비(data/processed/fake_data.npy)
2. configs/config.yaml           # 설정
3. src/data/dataset.py           # 데이터 전처리 및 로더
4. src/models/simple_net.py      # 모델 정의
5. src/training/train_loop.py    # 학습 루프 정의
6. src/evaluation/evaluate.py    # 평가 로직
7. scripts/train.py              # 학습 스크립트
8. scripts/eval.py               # 평가 스크립트
9. scripts/inference.py          # 추론 스크립트
10. scripts/select_best_model.py # 최종 모델 옮기기

### 정리
위 코드를 통해 전체 폴더 구조를 실제로 어떻게 사용하는지를 작은 예시로 확인할 수 있다:
1.	데이터(data/processed/fake_data.npy) 준비
2.	config.yaml 에서 하이퍼파라미터와 경로 등을 관리
3.	src/ 내부에서 데이터 로더, 모델, 학습 루프, 평가 로직 등 모듈화
4.	scripts/ 내부에서 학습(train.py), 평가(eval.py), 추론(inference.py) 스크립트를 작성
5.	학습 시에는 experiments/experiment_1/checkpoints 에 에폭별 모델 가중치를 저장하고, 최종적으로 쓸 모델은 checkpoints/final_model.pth 로 옮겨 사용

이 예시를 바탕으로, 실제 프로젝트에서는 더 복잡한 데이터 전처리, 다양한 모델 및 학습 전략, 평가 메트릭, 실험 자동화(하이퍼파라미터 스윕) 등을 확장해 나가면 됨.
