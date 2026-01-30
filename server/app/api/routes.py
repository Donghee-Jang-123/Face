from fastapi import APIRouter, UploadFile, File, Form
from app.services import face_service
# 방금 만든 search_service를 불러옵니다
from app.services.search_service import vector_store 

router = APIRouter()

@router.post("/register")
async def register(
    nickname: str = Form(...),
    file: UploadFile = File(...)
):
    contents = await file.read()
    embedding = face_service.extract_embedding(contents)
    
    if embedding is None:
        return {"success": False, "message": "얼굴을 인식하지 못했습니다."}
    
    # [수정됨] 직접 저장 대신 vector_store에 추가 요청
    vector_store.add_user(embedding, nickname)
    
    return {"success": True, "message": f"{nickname}님, FAISS에 등록 완료!"}

@router.post("/login")
async def login(file: UploadFile = File(...)):
    contents = await file.read()
    embedding = face_service.extract_embedding(contents)
    
    if embedding is None:
        return {"success": False, "message": "얼굴을 인식하지 못했습니다."}
    
    # [수정됨] 반복문 대신 vector_store에 검색 요청 (0.01초 소요)
    result = vector_store.search_user(embedding)
    
    if result["found"]:
        return {
            "success": True, 
            "nickname": result["nickname"], 
            "score": result["score"],
            "message": f"{result['nickname']}님 환영합니다!"
        }
    else:
        return {"success": False, "message": "누구신지 잘 모르겠어요."}