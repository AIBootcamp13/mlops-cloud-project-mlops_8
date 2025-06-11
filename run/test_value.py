# 표준 라이브러리
import sys
import os
from datetime import datetime
# 현재 파일 기준 상위 경로 (mlops-project 디렉토리)를 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import requests
import pandas as pd


from tqdm import tqdm
from src.dataset.preprocess import get_datasets
from src.evaluation.evaluation_def import regression_metrics  # 사용자 정의 성능 함수

# ✅ API 서버 주소
API_URL = "http://127.0.0.1:8000/predict"

# ✅ 데이터 로드
_, test_df = get_datasets()
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# ✅ 결과 저장
y_true_list = []
y_pred_list = []

# ✅ 반복 테스트 (100개)
for idx in tqdm(range(100)):
    input_data = X_test.iloc[idx].to_dict()
    y_true = y_test.iloc[idx]

    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()

        result = response.json()
        y_pred = result["prediction"][0]

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    except Exception as e:
        print(f"[❌ Error at index {idx}] {e}")
        continue

print(f"\n✅ 총 예측 성공 건수: {len(y_pred_list)}개")

# ✅ 성능 평가
regression_metrics(y_true_list, y_pred_list)
for idx in range(100):  # ← 반드시 루프가 있어야 함
    input_data = X_test.iloc[idx].to_dict()
    y_true = y_test.iloc[idx]

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code != 200:
            print(f"[⚠️ HTTP {response.status_code}] {response.text}")
            continue  # ← 이제 문제 없음

        result = response.json()
        y_pred = result["prediction"][0]

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    except Exception as e:
        print(f"[❌ Error at index {idx}] {e}")
        continue  # ← 여기도 루프 안이므로 OK
