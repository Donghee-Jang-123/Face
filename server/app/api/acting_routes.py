"""
연기 분석 API (DTW 기반)

Audio-only DTW로 동기화 후, 피치/볼륨/표정을 종합 평가합니다.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import shutil
import os

from app.services.acting_analysis_pipeline import (
    get_acting_pipeline,
    scoring_result_to_dict,
)

router = APIRouter(prefix="/analyze", tags=["Acting Analysis"])

# 임시 폴더 경로
TEMP_DIR = "temp"


# =============================================================================
# 연기 분석 API
# =============================================================================

@router.post("/acting")
async def analyze_acting(
    file: UploadFile = File(...),
    actor_id: str = Form(...),
    user_id: str = Form(default="user"),
):
    """
    DTW 기반 연기 분석 API.
    
    Audio-only DTW로 동기화 후, 피치/볼륨/표정을 종합 평가합니다.
    
    Args:
        file: 사용자 영상 파일 (.mp4, .webm)
        actor_id: 비교할 레퍼런스 배우 ID (사전 등록 필요)
        user_id: 사용자 ID (선택)
        
    Returns:
        종합 점수 및 세부 피드백
    """
    pipeline = get_acting_pipeline()
    
    # 레퍼런스 확인
    available_refs = pipeline.list_references()
    if actor_id not in available_refs:
        raise HTTPException(
            status_code=404,
            detail=f"레퍼런스를 찾을 수 없습니다: {actor_id}. "
                   f"사용 가능: {available_refs}"
        )
    
    # 임시 파일 저장
    os.makedirs(TEMP_DIR, exist_ok=True)
    user_video_path = f"{TEMP_DIR}/{file.filename}"
    
    try:
        with open(user_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🎬 분석 시작: {user_id} vs {actor_id}")
        
        # 평가 실행
        result = pipeline.evaluate_user(
            user_video=user_video_path,
            actor_id=actor_id,
            user_id=user_id,
        )
        
        # 응답 변환
        response = scoring_result_to_dict(result)
        response["actor_id"] = actor_id
        response["user_id"] = user_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 임시 파일 삭제
        if os.path.exists(user_video_path):
            os.remove(user_video_path)


@router.post("/acting/quick")
async def analyze_acting_quick(
    file: UploadFile = File(...),
    actor_id: str = Form(...),
):
    """
    [빠른 평가] 오디오만 분석하여 빠르게 동기화 점수를 반환합니다.
    
    비디오 처리를 생략하여 응답 속도가 빠릅니다.
    전체 평가 전 사전 체크용으로 사용할 수 있습니다.
    """
    pipeline = get_acting_pipeline()
    
    # 레퍼런스 확인
    available_refs = pipeline.list_references()
    if actor_id not in available_refs:
        raise HTTPException(
            status_code=404,
            detail=f"레퍼런스를 찾을 수 없습니다: {actor_id}"
        )
    
    # 임시 파일 저장
    os.makedirs(TEMP_DIR, exist_ok=True)
    audio_path = f"{TEMP_DIR}/{file.filename}"
    
    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 빠른 평가 실행
        result = pipeline.evaluate_audio_only(
            user_audio=audio_path,
            actor_id=actor_id,
        )
        
        result["actor_id"] = actor_id
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# =============================================================================
# 레퍼런스 관리 API
# =============================================================================

@router.post("/reference/prepare")
async def prepare_reference(
    file: UploadFile = File(...),
    actor_id: str = Form(...),
    force: bool = Form(default=False),
):
    """
    레퍼런스 영상을 사전 분석하여 등록합니다.
    
    Args:
        file: 레퍼런스 영상 파일 (.mp4)
        actor_id: 배우/영상 고유 ID
        force: True면 기존 데이터 덮어쓰기
        
    Returns:
        등록된 레퍼런스 정보
    """
    pipeline = get_acting_pipeline()
    
    # 임시 파일 저장
    os.makedirs(TEMP_DIR, exist_ok=True)
    video_path = f"{TEMP_DIR}/{file.filename}"
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📦 레퍼런스 준비: {actor_id}")
        
        # 분석 실행
        result = pipeline.prepare_reference(
            video_path=video_path,
            actor_id=actor_id,
            force=force,
        )
        
        return {
            "status": "success",
            "actor_id": actor_id,
            "duration_sec": result.duration_sec,
            "fps": result.fps,
            "frame_count": result.frame_count,
            "has_audio": result.has_audio,
            "has_video": result.has_video,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 임시 파일 삭제
        if os.path.exists(video_path):
            os.remove(video_path)


@router.get("/reference/list")
async def list_references():
    """
    분석 완료된 레퍼런스 목록을 반환합니다.
    """
    pipeline = get_acting_pipeline()
    refs = pipeline.list_references()
    
    return {
        "count": len(refs),
        "references": refs,
    }


@router.get("/reference/{actor_id}")
async def get_reference_info(actor_id: str):
    """
    특정 레퍼런스의 상세 정보를 반환합니다.
    """
    pipeline = get_acting_pipeline()
    
    try:
        info = pipeline.get_reference_info(actor_id)
        return info
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"레퍼런스를 찾을 수 없습니다: {actor_id}"
        )


# =============================================================================
# Assets 관리 API
# =============================================================================

@router.get("/assets/list")
async def list_assets():
    """
    assets 폴더의 모든 비디오와 분석 상태를 반환합니다.
    """
    pipeline = get_acting_pipeline()
    assets = pipeline.list_assets()
    
    analyzed = sum(1 for a in assets if a["is_analyzed"])
    pending = len(assets) - analyzed
    
    return {
        "total": len(assets),
        "analyzed": analyzed,
        "pending": pending,
        "assets": assets,
    }


@router.post("/assets/sync")
async def sync_assets(force: bool = False):
    """
    assets 폴더를 스캔하고 새로운 비디오를 분석합니다.
    
    Args:
        force: True면 모든 비디오를 재분석
        
    Returns:
        동기화 결과
    """
    pipeline = get_acting_pipeline()
    
    try:
        results = pipeline.sync_assets(force=force)
        
        analyzed = sum(1 for s in results.values() if s == "analyzed")
        skipped = sum(1 for s in results.values() if s == "skipped")
        errors = sum(1 for s in results.values() if s.startswith("error"))
        
        return {
            "status": "success",
            "analyzed": analyzed,
            "skipped": skipped,
            "errors": errors,
            "details": results,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets/pending")
async def get_pending_assets():
    """
    아직 분석되지 않은 비디오 목록을 반환합니다.
    """
    pipeline = get_acting_pipeline()
    pending = pipeline.get_pending_analyses()
    
    return {
        "count": len(pending),
        "pending": [
            {"actor_id": actor_id, "filename": str(path.name)}
            for actor_id, path in pending
        ],
    }
