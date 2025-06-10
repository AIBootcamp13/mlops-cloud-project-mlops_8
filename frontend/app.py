import pandas as pd
from flask import Flask, render_template
from datetime import datetime
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# FastAPI 서버 설정
FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')  # FastAPI 서버 주소
PREDICTIONS_ENDPOINT = os.getenv('PREDICTIONS_ENDPOINT', '/predictions/temperature')  # 예측 API 엔드포인트

TOURIST_SPOTS = [
    {"Spot_id": "7705", "lat": 37.55093, "lon": 126.989146},
    {"Spot_id": "6103", "lat": 37.56584, "lon": 126.972368},
    {"Spot_id": "5502", "lat": 37.562962, "lon": 126.984657}
]
  
def get_predictions_from_api():
    """FastAPI 서버에서 예측된 기온 데이터를 가져옵니다."""
    try:
        response = requests.get(f"{FASTAPI_URL}{PREDICTIONS_ENDPOINT}")
        response.raise_for_status()  # HTTP 에러 체크
        predictions = response.json()
        return predictions
    except requests.exceptions.RequestException as e:
        print(f"Error fetching predictions from API: {e}")
        # 에러 발생 시 빈 딕셔너리 반환
        return {}

@app.route('/')
def index():
    current_time = datetime.now().strftime("%H:%M")
    weather_data = []
    
    # FastAPI 서버에서 예측 데이터 가져오기
    predictions = get_predictions_from_api()
    print(predictions)
    
    # 관광지 정보 데이터프레임 로드
    df = pd.read_csv("dataset/seoul_tour_hotspot15.csv")    
    
    # 데이터프레임의 각 관광지에 대해 처리
    for _, row in df.iterrows():
        spot_id = str(row['관광지 아이디'])  # 문자열로 변환

        # 예측 데이터 가져오기 (없으면 None 사용)
        prediction = predictions.get(spot_id, {})
        temp = prediction.get("Average_temperature", None)

        weather_info = {
            "name": row['관광지명_분리'],
            "temp": round(float(temp), 1) if temp is not None else None,
            "lat": row['위도(도)'],
            "lon": row['경도(도)']
        }
        weather_data.append(weather_info)
    
    return render_template('index.html', 
                         weather_data=weather_data,
                         current_time=current_time)

if __name__ == '__main__':
    app.run(debug=True) 