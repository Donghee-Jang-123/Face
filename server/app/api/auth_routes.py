from fastapi import APIRouter, UploadFile, File, Form
from app.services import face_service
from app.services.search_service import vector_store 

router = APIRouter(tags=["Authentication"])

@router.post("/register")
async def register(
    nickname: str = Form(...),
    file: UploadFile = File(...)
):
    """
    얼굴 이미지로 회원가입합니다.
    - 얼굴에서 512차원 임베딩 추출
    - FAISS 벡터 DB에 저장
    """
    contents = await file.read()
    embedding = face_service.extract_embedding(contents)
    
    if embedding is None:
        return {"success": False, "message": "얼굴을 인식하지 못했습니다."}
    
    # FAISS에 추가
    vector_store.add_user(embedding, nickname)
    
    return {"success": True, "message": f"{nickname}님, FAISS에 등록 완료!"}


@router.post("/login")
async def login(file: UploadFile = File(...)):
    """
    얼굴 이미지로 로그인합니다.
    - 얼굴 임베딩 추출 후 FAISS에서 유사한 사용자 검색
    """
    contents = await file.read()
    embedding = face_service.extract_embedding(contents)
    
    if embedding is None:
        return {"success": False, "message": "얼굴을 인식하지 못했습니다."}
    
    # FAISS에서 검색 (0.01초 소요)
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
