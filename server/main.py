import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core import models
from app.api import routes

app = FastAPI()

# ⚠️ CORS 설정: 프론트엔드(Next.js)와의 연결을 허용합니다.
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 모델 로드 (Step 1 실행)
@app.on_event("startup")
def startup_event():
    models.load_models()

# API 경로 연결 (Step 3 연결: /register, /login 등)
app.include_router(routes.router)

if __name__ == "__main__":
    # "main:app"에서 main은 파일 이름(main.py)을 의미합니다.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)