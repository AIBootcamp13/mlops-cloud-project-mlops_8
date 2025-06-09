# 표준 라이브러리
import sys
import os
from datetime import datetime
# 현재 파일 기준 상위 경로 (mlops-project 디렉토리)를 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# 서드파티
import pandas as pd
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

# 로컬 모듈
from src.model.LightGBMTrainer import LightGBMTrainer
from src.dataset.preprocess import get_datasets
from src.dataset.CrossValidation import CrossValidator
from src.evaluation.evaluation_def import regression_metrics




# 💡 MLflow 환경변수 URI 설정 (기본값 포함)
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_uri)



#실험 이름 등록 
mlflow.set_experiment("TourTempForecast")
def main():
    """모델 학습 및 MLflow 등록을 위한 메인 함수"""

    # ✅ 데이터 로드 및 분리
    train_df, test_df = get_datasets()
    # target 값 중 5개 출력
    print("📊 train target 샘플:", train_df["target"].head().tolist())
    
    X = train_df.drop("target", axis=1)
    y = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]


 
    # ✅ 학습/검증 분리
    cv = CrossValidator()
    X_train, X_val, y_train, y_val = cv.split(X, y)

    # ✅ 모델 생성
    trainer = LightGBMTrainer(num_boost_round=100, log_transformed_target=False,log_period=30,verbose=True)

    
    # run name 설정 
    params = trainer.get_model_hyperparams()
    run_name = f"lgbm_experiment_{params['learning_rate']}_{datetime.now().strftime('%H%M%S')}"

    # ✅ 단일 실험 시작
    with mlflow.start_run(run_name=run_name):
        print("단일 실험 시작")
		#현재 설정된 파라미터가 수동인지/자동인지 출력
        params = trainer.get_model_hyperparams()
	
        # 수동 설정 여부 확인 예시
        if params == trainer.default_params():
            print("✅ 기본(default) 하이퍼파라미터 사용 중입니다.")
        else:
            print("✅ 수동으로 설정된 하이퍼파라미터가 사용 중입니다.")
		# 하이퍼파라미터 및 설정 기록

        print('하이퍼파라미터 및 설정을 기록합니다 ')
        mlflow.log_params(trainer.get_model_hyperparams())
        mlflow.log_params(trainer.get_trainer_configs())

        print('학습을 시작합니다.')
        # 학습 + 평가
        trainer.fit(X_train, y_train, X_val, y_val)
        #rmse = trainer.evaluate(X_test, y_test)

 
        print('메트릭을 기록하기 시작합니다')
        # 메트릭 추가 기록 (custom metrics)
        y_pred = trainer.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)

        for k, v in metrics.items():
            mlflow.log_metric(k.lower(), v)
        
        # signature 및 input example 정의
        signature = infer_signature(X_train, y_pred)
        input_example = X_train.iloc[:1]
        print('모델 저장을 시작합니다.')
        # 모델 저장
        mlflow.lightgbm.log_model(
            trainer.model,
            "model",
            input_example=input_example,
            signature=signature
        )

        # 모델 Registry 등록
        run_id = mlflow.active_run().info.run_id
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="TourTempLightGBMModel"
        )

if __name__ == "__main__":
    main()
