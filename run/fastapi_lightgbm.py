from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow.pyfunc
import mlflow
from mlflow.tracking import MlflowClient
import time
import logging
import os
import numpy as np
from mlflow.exceptions import MlflowException

# 환경변수에서 URI 설정 (기본값 포함)
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", "http://127.0.0.1:5000"))

print("📡 현재 MLflow Tracking URI:", mlflow.get_tracking_uri())


app = FastAPI(title="MLflow Model API", description="ML model serving via FastAPI", version="1.0")
# 사용자가 직접 정의한 title, description, version 정의한 fast api 인스턴스 생성 

# ✅ 모델 이름과 스테이지
model_name = "TourTempLightGBMModel"
model_stage = "Production"


# 모델 로드
try:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
    print("✅ 모델 로딩 성공")
except MlflowException as e:
    print(f"❌ 모델 로딩 실패: {e}")
    model = None  # 추후 예측 시 에러 처리
#mlflow의 모델 레지스트리에서 staging 스테이지에 등록된 모델 불러옴
#/your_model_name/Production: 모델이 저장된 경로 이름

if model:
    # ✅ 모델 입력 컬럼 확인
    print("🔍 모델 입력 컬럼:", model.metadata.signature.inputs.input_names())

    # ✅ 모델 version 및 하이퍼파라미터 정보 출력(제대로 모델 가져왔는지 검증용)
    client = MlflowClient()
    # 모든 버전을 검색한 후, 원하는 스테이지만 필터링
    all_versions = client.search_model_versions(f"name='{model_name}'")

    # 원하는 스테이지(ex. Production)인 모델 버전만 추출
    latest_versions = [v for v in all_versions if v.current_stage == model_stage]

    if latest_versions:
        model_version_info = latest_versions[0]  # 첫 번째 production 모델
        run_id = model_version_info.run_id
        version = model_version_info.version
        print(f"📌 모델 버전: {version}, Run ID: {run_id}")

        # 하이퍼파라미터 출력
        run_data = client.get_run(run_id).data
        print("📊 모델 하이퍼파라미터:")
        for k, v in run_data.params.items():
            print(f"  {k}: {v}")
    else:
        print(f"❌ '{model_name}' 모델의 '{model_stage}' 스테이지 버전을 찾을 수 없습니다.")
else:
    print(f"⚠️ 모델이 None입니다. 이후 정보 출력은 생략합니다.")

    
class InputData(BaseModel):#post 요청에서 받을 json 데이터를 정의하는 클래스
    Spot_id: int = Field(..., description="관광지 고유 식별자")
    YMD: int = Field(..., description="연월일 (예: '20250605')")
    STN_ID: int = Field(..., description="기상청 지점 번호")
    LAT: float = Field(..., description="위도")
    LON: float = Field(..., description="경도")
    Sum_rainfall: float = Field(..., description="합계 강수량 (mm)")
    Max_rainfall_1H: float = Field(..., description="1시간 최다 강수량 (mm)")
    Max_rainfall_1H_occur_time: float = Field(..., description="1시간 최다 강수량 발생 시각 (시)")
    Average_humidity: float = Field(..., description="평균 상대 습도 (%)")
    Min_humidity: float = Field(..., description="최소 상대 습도 (%)")
    year: int = Field(..., description="연도 (예: 2025)")
    month: int = Field(..., description="월 (예: 6)")
    day: int = Field(..., description="일 (예: 5)")

# -> 이 클래스를 기반으로 입력 검증+자동 문서화(swagger)까지 처리 

# ✅ 응답 모델 정의 (Response Body): 
# 반환되는 json 형식이 swagger UI에서 더 명확하게 표시되도록 설정
class PredictionResponse(BaseModel):
    spot_id: int
    lat: float
    lon: float
    prediction: list[float]



@app.post("/predict") #/predict 경로 등록+ /predict 경로로 들어오는 post 요청 처리+자동으로 swagger ui에 등록됨 
#이 후 아래의 predict 함수를 받아들여서 post 함수에 predict 함수가 실행됨
def predict(data: InputData,response_model=PredictionResponse, tags=['Model Prediction']): 
#fast api는 클라이언트의 json body 형식을 pydantic으로 inputdata로 자동 변환함
    try:
        df = pd.DataFrame([data.dict()]) #pydantic 모델 -> python dict -> pandas dataframe으로 변환 
        prediction = model.predict(df) #mlflow 모델에 실제 입력 데이터를 넘겨 예측 수행 
        return {
            "spot_id": data.Spot_id,
            "lat": data.LAT,
            "lon": data.LON,
            "prediction": prediction.tolist()
        } #numpy 결과 or pandas 객체를 json 응답으로 반환 -> .tolist() 방향으로 필요
          #나머지는 input 데이터에서 값을 가져와서 응답 json에 포함 
    #fastapi는 이 json을 클라이언트에 HTTP 응답으로 보내줌 uvicorn fastapi_lightgbm:app --reload
    except Exception as e:
        print("❌ 예측 중 오류 발생:", e)
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}") #예측 실패 예외 처리


#응답 속도 로깅 

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@app.middleware("http")
async def log_response_time(request, call_next):
    start_time = time.time()
    response = await call_next(request)  # 실제 요청 처리
    process_time = time.time() - start_time
    logging.info(f"⏱️ {request.method} {request.url.path} 처리시간: {process_time:.4f}초")
    response.headers["X-Process-Time"] = f"{process_time:.4f}"  # 응답 헤더에 처리시간 추가 (선택)
    return response