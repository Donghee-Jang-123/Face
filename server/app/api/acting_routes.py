from fastapi import APIRouter, UploadFile, File, Form
import shutil
import os

from app.services.audio_service import get_audio_service
from app.services.video_service import get_video_service

router = APIRouter(prefix="/analyze", tags=["Acting Analysis"])

# 임시 폴더 경로
TEMP_DIR = "temp"
ASSETS_DIR = "assets"

@router.post("/acting")
async def analyze_acting(
    file: UploadFile = File(...),
    target_filename: str = Form(...)  # 프론트에서 보낸 타겟 영상 파일명 (assets 폴더 내)
):
    """
    사용자의 연기 영상을 타겟 영상과 비교 분석합니다.
    - 표정 싱크로율 (MediaPipe)
    - 감정 분석 (Wav2Vec2)
    """
    # 서비스 인스턴스 가져오기
    audio_service = get_audio_service()
    video_service = get_video_service()
    
    # 폴더 생성
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # 1. 유저 파일 저장
    user_video_path = f"{TEMP_DIR}/{file.filename}"
    with open(user_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. 타겟 파일 경로 확인
    target_video_path = f"{ASSETS_DIR}/{target_filename}"
    if not os.path.exists(target_video_path):
        # 테스트용: 타겟 영상이 없으면 유저 영상을 복사해서 테스트
        shutil.copy(user_video_path, target_video_path)

    print(f"🚀 분석 시작: User({user_video_path}) vs Target({target_video_path})")

    # 3. [Audio Service] 감정 분석
    user_audio_result = audio_service.analyze_emotion(user_video_path)
    target_audio_result = audio_service.analyze_emotion(target_video_path)

    # 4. [Video Service] 표정 싱크로율 분석
    user_shapes = video_service.process_video_shapes(user_video_path)
    target_shapes = video_service.process_video_shapes(target_video_path)
    
    # 두 데이터 비교
    sync_score = video_service.calculate_sync_rate(user_shapes, target_shapes)

    # 5. 결과 정리
    emotion_match = user_audio_result['emotion'] == target_audio_result['emotion']
    
    # 최종 점수 계산 (표정 70% + 감정일치 30%)
    final_score = (sync_score * 0.7) + (30 if emotion_match else 0)

    # 6. 임시 파일 삭제
    os.remove(user_video_path)

    return {
        "score": round(final_score, 1),
        "sync_rate": sync_score,
        "emotion": {
            "user": user_audio_result['emotion'],
            "target": target_audio_result['emotion'],
            "is_match": emotion_match
        },
        "feedback": _generate_feedback(sync_score, emotion_match)
    }


def _generate_feedback(score: float, emotion_match: bool) -> str:
    """분석 결과에 따른 피드백 메시지 생성"""
    if score > 80 and emotion_match:
        return "완벽해요! 표정과 감정 모두 훌륭합니다."
    elif not emotion_match:
        return "표정은 좋지만, 목소리의 감정을 다시 잡아보세요."
    else:
        return "입 모양과 눈의 움직임을 더 과감하게 표현해보세요!"
