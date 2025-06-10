# 서울시 관광지 날씨 정보 서비스

서울시의 주요 관광지 15곳의 실시간 날씨 정보를 제공하는 웹 서비스입니다.

## 주요 기능

- 실시간 날씨 정보 표시
- 관광지별 온도 표시
- 지도 기반 위치 시각화

## 기술 스택

- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- 지도: Leaflet.js + OpenStreetMap
- UI 프레임워크: Bootstrap 5

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd [project-directory]
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
- `.env` 파일을 프로젝트 루트 디렉토리에 생성
- 다음 내용을 추가:
```
WEATHER_API_KEY=your_openweathermap_api_key_here
```

## 실행 방법

1. Flask 서버 실행
```bash
python app.py
```

2. 웹 브라우저에서 접속
```
http://localhost:5000
```
