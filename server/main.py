# backend/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# ⚠️ CORS 설정 (이게 없으면 프론트에서 에러 남!)
origins = [
    "http://localhost:3000",  # Next.js 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 헬스 체크용 (서버 살아있나 확인)
@app.get("/")
def read_root():
    return {"message": "FastAPI 서버가 정상 작동 중입니다! 🚀"}

# 2. 이미지 업로드 테스트용
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    return {"filename": file.filename, "status": "이미지 받기 성공!"}

if __name__ == "__main__":
    # 0.0.0.0은 외부 접속 허용, 로컬에서는 127.0.0.1과 같음
    uvicorn.run(app, host="0.0.0.0", port=8000)