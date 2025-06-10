FROM python:3.10-slim

WORKDIR /app

# LightGBM과 ML 라이브러리들에 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# 필요한 Python 패키지 설치를 위한 requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 구조 복사
COPY . .

# Python 경로 설정
ENV PYTHONPATH=/app

# 컨테이너가 종료되지 않도록 tail 명령어 추가
CMD ["sh", "-c", "python main.py && tail -f /dev/null"]