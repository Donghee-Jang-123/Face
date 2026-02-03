import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core import models
from app.api import auth_routes, acting_routes
from app.api.recommend_routes import router as recommend_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 이벤트 핸들러"""
    # Startup: 모델 로드
    print("🚀 서버 시작 중...")
    
    # 1. 얼굴 인식 모델 (InsightFace)
    models.load_models()
    
    # 2. 필요한 폴더 생성
    os.makedirs("temp", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    os.makedirs("assets/videos", exist_ok=True)
    os.makedirs("data/references", exist_ok=True)
    
    # 3. Assets 폴더 자동 동기화 (새 MP4만 분석)
    print("\n📂 Assets 폴더 동기화 중...")
    try:
        from app.services.acting_analysis_pipeline import get_acting_pipeline
        pipeline = get_acting_pipeline()
        pipeline.sync_assets()
    except Exception as e:
        print(f"⚠️  Assets 동기화 중 오류 발생: {e}")
        print("   서버는 계속 실행되지만, 일부 레퍼런스가 누락될 수 있습니다.")
    
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
app.include_router(recommend_router, prefix="/api", tags=["recommend"])

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
            "acting": [
                "POST /analyze/acting - 연기 평가",
                "POST /analyze/acting/quick - 빠른 평가 (오디오만)",
            ],
            "reference": [
                "POST /analyze/reference/prepare - 레퍼런스 등록",
                "GET /analyze/reference/list - 분석 완료된 레퍼런스 목록",
                "GET /analyze/reference/{actor_id} - 레퍼런스 상세",
            ],
            "assets": [
                "GET /analyze/assets/list - 모든 비디오 및 분석 상태",
                "GET /analyze/assets/pending - 미분석 비디오 목록",
                "POST /analyze/assets/sync - 새 비디오 자동 분석",
            ]
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
