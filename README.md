📌 ## 프로젝트 개요

<br>

## 💻 프로젝트 소개
### 서울시 관광지 기온 예측 서비스 (SeoulTourTempForecast)
- 서울시 주요 관광지의 기온을 예측하여 시민과 관광객에게 유용한 정보를 제공하는 기상 기반 관광 지원 서비스입니다. <br>
  기상청 API로 수집한 날씨 데이터를 전처리하고, 특정 관광지의 과거 기온 데이터를 기반으로 머신러닝 모델을 훈련시켜 예측 서비스를 제공합니다.
  이 프로젝트는 MLOps 관점에서 데이터 수집 → 모델 훈련 → 예측 서비스 제공까지의 전 과정을 자동화하는 것이 목적입니다.

### <작품 소개>
- 주요 관광지 몇개의 내일 온도를 예측해서 지도에 보여줍니다.

<br>

## 👨‍👩‍👦‍👦 팀 구성원

| ![김주형](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이재용](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최지희](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김재덕](https://avatars.githubusercontent.com/u/156163982?v=4) | ![강연경](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김주형](https://github.com/UpstageAILab)             |            [이재용](https://github.com/UpstageAILab)             |            [최지희](https://github.com/UpstageAILab)             |            [김재덕](https://github.com/UpstageAILab)             |            [강연경](https://github.com/UpstageAILab)             |
|                            팀장, 데이터 파이프라인                             |                            모델 학습                             |                            화면 구성                             |                            모니터링 및 배포                             |                            전체 worksapce 구성                             |

<br>

## 🔨 개발 환경 및 기술 스택
- 주 언어 : Python 3.11
- 🧰 프레임워크 : Apache Airflow (workflow orchestration), FastAPI (API 서버), MLFlow (모델 모니터링 및 배포)
- 버전 및 이슈관리 : github
- 협업 툴 : github, notion
- 📦 패키지 관리 : pip + requirements.txt
- 🧪 머신러닝 : scikit-learn, LightGBM
- 🧪 실험/모델 모니터링 : MLflow
- ☁️ 배포 환경 : AWS EC2 (Ubuntu), Docker
- 🔧 이슈관리 : GitHub Issues
- 🧑‍🤝‍🧑 협업 툴 : Notion, Slack
- 📄 환경 설정 : .env + dotenv

<br>

## 📁 프로젝트 구조
```
├── src
│   ├── dataset
│   │   └── CrossValidation.py
│   │   └── data_collector.py
│   │   └── EDA.ipynb
│   │   └── inferance.ipynb
│   │   └── preprocess.py
│   │   └── run_pipeline.py
│   │   └── requirements.txt
│   ├── evaluation
│   │   └── evaluation_def.py
│   ├── model
│   │   └── BaseTrainer.py
│   │   └── CatBoostTrainer.py
│   │   └── LightGBMTrainer.py
│   │   └── RandomForestTrainer.py
│   │   └── XGBoostTrainer.py
│   ├── test
│   │   └── catBoostTest.py
│   │   └── lightGBMTest.py
│   │   └── randomForestTest.py
│   │   └── xgboostTest.py
│   ├── util
│   │   └── korean_matplot_setting.py
│   │   └── preprocessor.py
│   │   └── s3_handler.py
│   │   └── util_function.py
├── docs
│   ├── pdf
│   │   └── [패스트캠퍼스] Upstage AI Lab 7기_ML-프로젝트_8조.pptx
│   └── Data_Collection.md
└── dataset
│   ├── weather_data_202528.csv
│   ├── preprocessed_weather_data_20250605.csv
│   ├── target_scaler.joblib
└── datapipeline
│   ├── Dockerfile
│   ├── README.md
└── airflow
│   ├── Airflow_README.md
│   ├── docker_run.cmd
│   ├── Dockerfile
│   ├── entrypoint.sh
└── mlflow
│   ├── 
├── .dockerignore
├── main.py
├── requirements.txt
```

<br>

## 💻​ 구현 기능
### 🔄 데이터 수집
- 기상청 API를 활용한 기온/강수/습도 등 자동 수집
### 🧹 데이터 전처리
- 이상치 제거, 결측치 처리, Feature Engineering
### 📊 EDA
- 시각화를 통한 데이터 탐색 분석
### 🧠 모델링
- LightGBM을 통한 기온 예측 회귀 분석 모델 구축
### 📈 모델 성능 평가
- RMSE 기준 성능 측정
### ⚙️ 워크플로우 자동화
- Airflow DAG을 통한 주기적 수집/학습/예측 파이프라인
### 🚀 API 서비스
- FastAPI를 통해 특정 관광지의 예측 기온을 제공하는 API
### 📊 모니터링
- MLflow를 통한 실험 및 성능 추적
### 🐳 컨테이너화
- Dockerfile을 통한 이식성 높은 실행 환경 제공

<br>

<br>

## 🚨​ 트러블 슈팅

### 1. Airflow DockerOperator ML 파이프라인 FileNotFoundError 트러블슈팅

#### 설명
- Airflow DockerOperator를 사용한 ML 파이프라인에서 FileNotFoundError가 발생했습니다. ML 코드가 실행되는 도커 이미지 내부에 파일이 존재함에도 불구하고, 태스크 실행 시 파일을 찾지 못했습니다. 이는 DockerOperator가 매번 새로운 컨테이너를 생성하는 방식과 ML 코드 내 데이터셋 파일 경로 참조 방식(상대 경로 사용)의 불일치 때문이었습니다. 새로 생성된 컨테이너가 의도한 위치에서 파일을 찾지 못하면서 오류가 발생했습니다.

#### 해결
- 도커 이미지 및 버전 재확인: Dockerfile에서 데이터셋 파일이 /opt/mlops/dataset/ 경로에 정확히 COPY되는지 확인하고, Airflow가 최신 버전의 이미지를 사용하도록 보장했습니다.
- ML 코드 내 마운트 되는 경로 절대 경로로 수정

## 📰​ 참고자료
- [ppt](https://docs.google.com/presentation/d/1BX9PCfKckJTmf3du9hvXv3o8dujUtgkN/edit?slide=id.p1#slide=id.p1)
