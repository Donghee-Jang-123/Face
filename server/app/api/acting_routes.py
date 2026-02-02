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
    
    # 두 데이터 비교 (항목별 점수 포함)
    sync_result = video_service.calculate_sync_rate(user_shapes, target_shapes)
    sync_score = sync_result["total"]
    sync_details = sync_result["details"]

    # 5. 결과 정리
    emotion_match = user_audio_result['emotion'] == target_audio_result['emotion']
    
    # 최종 점수 계산 (표정 70% + 감정일치 30%)
    final_score = (sync_score * 0.7) + (30 if emotion_match else 0)

    # 6. 항목별 피드백 생성
    detailed_feedback = _generate_detailed_feedback(sync_details)

    # 7. 임시 파일 삭제
    os.remove(user_video_path)

    return {
        "score": round(final_score, 1),
        "sync_rate": sync_score,
        "sync_details": sync_details,
        "emotion": {
            "user": user_audio_result['emotion'],
            "target": target_audio_result['emotion'],
            "is_match": emotion_match
        },
        "feedback": _generate_feedback(sync_score, emotion_match),
        "detailed_feedback": detailed_feedback
    }


def _generate_feedback(score: float, emotion_match: bool) -> str:
    """분석 결과에 따른 피드백 메시지 생성"""
    if score > 80 and emotion_match:
        return "완벽해요! 표정과 감정 모두 훌륭합니다."
    elif not emotion_match:
        return "표정은 좋지만, 목소리의 감정을 다시 잡아보세요."
    else:
        return "얼굴 표정 연습을 더 해보세요."


def _generate_detailed_feedback(sync_details: dict) -> dict:
    """항목별 상세 피드백 생성"""
    feedback_messages = {
        "jawOpen": {
            "low": "입을 더 크게 벌려서 대사를 말해보세요.",
            "mid": "입 벌림이 적절해요. 조금만 더 과감하게!",
            "high": "입 벌림이 원본과 잘 맞아요!"
        },
        "mouthSmile": {
            "low": "입꼬리의 움직임을 더 신경 써보세요.",
            "mid": "미소 표현이 나쁘지 않아요. 조금 더 자연스럽게!",
            "high": "입꼬리 움직임이 훌륭해요!"
        },
        "browInnerUp": {
            "low": "눈썹을 더 적극적으로 사용해보세요. 놀람/의심 표현에 중요해요.",
            "mid": "눈썹 움직임이 괜찮아요. 감정에 따라 더 강조해보세요.",
            "high": "눈썹 표현이 원본과 잘 맞아요!"
        },
        "eyeWide": {
            "low": "눈을 더 크게 떠서 감정을 표현해보세요.",
            "mid": "눈 표현이 적당해요. 감정의 강도에 맞게 조절해보세요.",
            "high": "눈 크기 변화가 원본과 일치해요!"
        },
        "mouthFrown": {
            "low": "입꼬리의 상하 움직임을 더 신경 써보세요.",
            "mid": "입꼬리 높낮이가 괜찮아요. 감정에 따라 더 표현해보세요.",
            "high": "입꼬리 높낮이가 잘 맞아요!"
        },
        "pupil": {
            "low": "시선 처리를 더 신경 써보세요.",
            "mid": "시선이 괜찮아요. 원본의 눈 움직임을 더 관찰해보세요.",
            "high": "시선 처리가 원본과 잘 맞아요!"
        },
        "philtrum": {
            "low": "코와 입술 사이의 움직임을 더 표현해보세요.",
            "mid": "인중 표현이 괜찮아요.",
            "high": "인중 움직임이 원본과 일치해요!"
        }
    }
    
    result = {}
    weak_points = []
    strong_points = []
    
    for key, detail in sync_details.items():
        score = detail["score"]
        name = detail["name"]
        messages = feedback_messages.get(key, {})
        
        if score < 60:
            level = "low"
            weak_points.append(name)
        elif score < 80:
            level = "mid"
        else:
            level = "high"
            strong_points.append(name)
        
        result[key] = {
            "score": score,
            "name": name,
            "level": level,
            "message": messages.get(level, "")
        }
    
    # 요약 정보 추가
    result["summary"] = {
        "weak_points": weak_points,
        "strong_points": strong_points,
        "focus_message": f"특히 {', '.join(weak_points[:2])}에 집중해보세요!" if weak_points else "전체적으로 훌륭해요!"
    }
    
    return result
