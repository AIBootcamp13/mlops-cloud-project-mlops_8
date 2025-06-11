import mlflow
from mlflow.tracking import MlflowClient
import os 

# 💡 MLflow 환경변수 URI 설정 (기본값 포함)
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_uri)



# 1. 클라이언트 생성
client = MlflowClient()

# 2. 모델 이름 지정 (Registry에 등록된 이름과 일치해야 함)
model_name="TourTempLightGBMModel"

# 3. 모델 버전 목록을 모두 가져온 후 생성일 기준 내림차순 정렬
versions = client.search_model_versions(f"name='{model_name}'")
latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

print(f"📦 현재 가장 최신 모델 버전: {latest_version.version}")
# ✅ 버전 숫자만 추출
version_number = latest_version.version  # 예: '1'

# 4. 스테이지 할당 staging  (예: Staging → Production)
client.transition_model_version_stage(
    name=model_name,
    version=version_number,
    stage="Production",  # 또는 "Staging"
    archive_existing_versions=True  # 이전 스테이지에 올라온 버전 자동 Archive
)
print(f"🎉 모델 {model_name} version {version_number}이(가) 'Production'으로 전환되었습니다.")
# 5. 모델 설명 추가
client.update_model_version(
name=model_name,
version=version_number,
description="결정된 회귀모델 version - LGBM 방식"
)

# 6. 알고리즘, 데이터셋 등 메타 태그 추가
client.set_model_version_tag(name=model_name, version=version_number, key="framework", value="LGBM") #모델 명
client.set_model_version_tag(name=model_name, version=version_number, key="dataset", value="Tourtemppredict") # 데이터 셋 명 
client.set_model_version_tag(name=model_name, version=version_number, key="owner", value="team_8") #작업자, 소유자 명 



