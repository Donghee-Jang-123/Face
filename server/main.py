import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core import models
from app.api import auth_routes, acting_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 이벤트 핸들러"""
    # Startup: 모델 로드
    print("🚀 서버 시작 중...")
    
    # 1. 얼굴 인식 모델 (InsightFace)
    models.load_models()
    
    # 2. 연기 분석 모델 (lazy loading - 첫 요청 시 로드)
    # audio_service와 video_service는 필요할 때 로드됩니다.
    
    # 3. 필요한 폴더 생성
    os.makedirs("temp", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    
    print("✅ 서버 준비 완료!")
    
    yield
    
    # Shutdown
    print("👋 서버 종료 중...")


app = FastAPI(
    title="Face Recognition & Acting Analysis API",
    description="얼굴 인식 회원가입/로그인 + 연기 분석 서비스",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정: 프론트엔드(Next.js)와의 연결을 허용합니다.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(auth_routes.router)      # /register, /login
app.include_router(acting_routes.router)    # /analyze/acting

# 정적 파일 서빙 (assets 폴더)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/")
async def root():
    """API 상태 확인용 엔드포인트"""
    return {
        "status": "running",
        "message": "Face Recognition & Acting Analysis API",
        "endpoints": {
            "auth": ["/register", "/login"],
            "acting": ["/analyze/acting"]
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
