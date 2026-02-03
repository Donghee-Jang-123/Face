"""
영상별 단어 타임스탬프 생성 스크립트 (Forced Alignment 방식)
============================================================
whisper-timestamped로 타임스탬프를 추출하되,
**텍스트는 사용자가 입력한 것을 그대로 사용**합니다.

핵심 원리:
    - whisper가 인식한 텍스트는 무시
    - whisper가 추출한 타임스탬프(start, end)만 사용
    - 사용자가 입력한 sentences[].text의 단어에 타임스탬프를 순서대로 매핑

사용자 입력 (actor_videos.json):
    {
      "sentences": [
        {"text": "제시카 외동딸"},
        {"text": "일리노이 시카고"}
      ]
    }

스크립트 실행 후 결과:
    {
      "sentences": [
        {
          "text": "제시카 외동딸",        ← 사용자 입력 그대로
          "start": 0.52,
          "end": 1.24,
          "words": [
            {"text": "제시카", ...},      ← 사용자 입력 그대로 (whisper 인식 결과 아님!)
            {"text": "외동딸", ...}
          ]
        }
      ]
    }

사용법:
    python scripts/generate_word_timestamps.py
    python scripts/generate_word_timestamps.py --video-id v_004
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.script_alignment_service import ScriptAlignmentService

# 경로 설정
DATABASE_FILE = PROJECT_ROOT / "app" / "database" / "actor_videos.json"


def load_videos() -> list[dict]:
    """actor_videos.json 로드"""
    with open(DATABASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_videos(videos: list[dict]) -> None:
    """actor_videos.json 저장"""
    with open(DATABASE_FILE, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    print(f"✅ 저장 완료: {DATABASE_FILE}")


def get_video_path(video_url: str) -> Path:
    """video_url에서 실제 파일 경로 추출"""
    relative_path = video_url.lstrip("/")
    return PROJECT_ROOT / relative_path


def extract_user_words(sentences: list[dict]) -> list[str]:
    """
    사용자가 입력한 문장들에서 단어 리스트 추출.
    공백으로 분리.
    """
    words = []
    for sentence in sentences:
        text = sentence.get("text", "")
        words.extend(text.split())
    return words


def map_timestamps_to_user_words(
    whisper_timestamps: list[dict],
    user_words: list[str]
) -> list[dict]:
    """
    whisper 타임스탬프를 사용자 단어에 매핑.
    
    핵심: whisper의 텍스트는 무시하고, 타임스탬프만 사용.
    사용자 단어와 whisper 단어를 1:1로 순서대로 매핑.
    
    Args:
        whisper_timestamps: whisper가 추출한 [{text, start, end}, ...]
        user_words: 사용자가 입력한 단어 리스트
        
    Returns:
        [{text: 사용자단어, start: whisper시간, end: whisper시간}, ...]
    """
    result = []
    
    whisper_count = len(whisper_timestamps)
    user_count = len(user_words)
    
    print(f"      📊 매핑: whisper {whisper_count}개 → 사용자 {user_count}개")
    
    if whisper_count == 0:
        print("      ⚠️  whisper 결과가 없습니다.")
        return result
    
    # 단어 수가 같으면 1:1 매핑
    if whisper_count == user_count:
        print("      ✅ 단어 수 일치 - 1:1 매핑")
        for i, user_word in enumerate(user_words):
            result.append({
                "text": user_word,  # 사용자 텍스트 그대로!
                "start": whisper_timestamps[i]["start"],
                "end": whisper_timestamps[i]["end"]
            })
    
    # whisper가 더 많으면: 사용자 단어 수만큼만 사용
    elif whisper_count > user_count:
        print(f"      ⚠️  whisper가 더 많음 - 앞에서 {user_count}개만 사용")
        for i, user_word in enumerate(user_words):
            result.append({
                "text": user_word,
                "start": whisper_timestamps[i]["start"],
                "end": whisper_timestamps[i]["end"]
            })
    
    # 사용자가 더 많으면: 타임스탬프 보간 (interpolation)
    else:
        print(f"      ⚠️  사용자 단어가 더 많음 - 타임스탬프 보간")
        
        # 전체 시간 범위
        total_start = whisper_timestamps[0]["start"]
        total_end = whisper_timestamps[-1]["end"]
        total_duration = total_end - total_start
        
        # 균등 분배
        word_duration = total_duration / user_count
        
        for i, user_word in enumerate(user_words):
            result.append({
                "text": user_word,
                "start": round(total_start + i * word_duration, 3),
                "end": round(total_start + (i + 1) * word_duration, 3)
            })
    
    return result


def assign_words_to_sentences(
    word_timestamps: list[dict],
    sentences: list[dict]
) -> list[dict]:
    """
    단어별 타임스탬프를 각 문장에 배분.
    """
    result_sentences = []
    word_index = 0
    
    for sentence in sentences:
        sentence_text = sentence.get("text", "")
        sentence_word_count = len(sentence_text.split())
        
        # 이 문장에 해당하는 단어들 가져오기
        sentence_words = []
        for _ in range(sentence_word_count):
            if word_index < len(word_timestamps):
                sentence_words.append(word_timestamps[word_index])
                word_index += 1
        
        # 문장 결과 생성
        if sentence_words:
            result_sentences.append({
                "text": sentence_text,  # 사용자 텍스트 그대로!
                "start": sentence_words[0]["start"],
                "end": sentence_words[-1]["end"],
                "words": sentence_words
            })
        else:
            result_sentences.append({
                "text": sentence_text,
                "start": 0,
                "end": 0,
                "words": []
            })
    
    return result_sentences


def _has_existing_timestamps(sentences: list[dict]) -> bool:
    """이미 타임스탬프가 존재하는지 확인합니다."""
    for sentence in sentences:
        if sentence.get("start") and sentence.get("end"):
            return True
        words = sentence.get("words", [])
        if isinstance(words, list) and len(words) > 0:
            return True
    return False


def generate_timestamps_for_video(
    service: ScriptAlignmentService,
    video: dict,
    force: bool,
) -> bool:
    """
    Forced Alignment 방식으로 타임스탬프 생성.
    
    1. 사용자 문장에서 단어 추출
    2. whisper로 오디오 분석 (타임스탬프 추출)
    3. whisper 타임스탬프를 사용자 단어에 매핑 (텍스트는 사용자 것 사용)
    """
    video_id = video.get("video_id", "unknown")
    video_url = video.get("video_url", "")
    sentences = video.get("sentences", [])
    
    if not sentences:
        print(f"  ⚠️  {video_id}: sentences가 없습니다. 건너뜁니다.")
        return False

    if not force and _has_existing_timestamps(sentences):
        print(f"  ⏭️  {video_id}: 기존 타임스탬프가 있어 스킵합니다. (--force로 재분석)")
        return False
    
    video_path = get_video_path(video_url)
    
    if not video_path.exists():
        print(f"  ❌ {video_id}: 영상 파일을 찾을 수 없습니다: {video_path}")
        return False
    
    # 1. 사용자 단어 추출
    user_words = extract_user_words(sentences)
    full_script = " ".join(user_words)
    
    print(f"  🎬 {video_id}: Forced Alignment 시작")
    print(f"      사용자 스크립트: \"{full_script}\"")
    print(f"      사용자 단어 수: {len(user_words)}개")
    
    try:
        # 2. whisper로 타임스탬프 추출 (힌트로 사용자 스크립트 전달)
        whisper_result = service.align(
            audio_path=str(video_path),
            script_text=full_script,
            language="ko"
        )
        
        print(f"\n      🔊 whisper 인식 결과 ({len(whisper_result)}개):")
        for w in whisper_result:
            print(f"          [{w['start']:.2f}s - {w['end']:.2f}s] \"{w['text']}\"")
        
        # 3. 타임스탬프를 사용자 단어에 매핑 (핵심!)
        print(f"\n      🔄 사용자 텍스트에 타임스탬프 매핑:")
        word_timestamps = map_timestamps_to_user_words(whisper_result, user_words)
        
        for w in word_timestamps:
            print(f"          [{w['start']:.2f}s - {w['end']:.2f}s] \"{w['text']}\"")
        
        # 4. 문장에 배분
        result_sentences = assign_words_to_sentences(word_timestamps, sentences)
        
        # 결과 업데이트
        video["sentences"] = result_sentences
        
        print(f"\n  📋 최종 결과:")
        for sent in result_sentences:
            print(f"      [{sent['start']:.2f}s - {sent['end']:.2f}s] \"{sent['text']}\"")
            for w in sent.get("words", []):
                print(f"          └ [{w['start']:.2f}s - {w['end']:.2f}s] \"{w['text']}\"")
        
        return True
        
    except Exception as e:
        print(f"  ❌ {video_id}: 분석 실패 - {e}")
        import traceback
        traceback.print_exc()
        return False


def main(target_video_id: str | None = None, force: bool = False):
    """메인 실행 함수"""
    print("=" * 60)
    print("🎤 Forced Alignment 타임스탬프 생성")
    print("=" * 60)
    print("📌 텍스트: 사용자 입력 그대로 사용")
    print("📌 타임스탬프: whisper에서 추출")
    print("=" * 60)
    
    # 서비스 초기화
    print("\n📦 Whisper 모델 로딩 중...")
    service = ScriptAlignmentService(model_name="base", device="cpu")
    
    # 영상 데이터 로드
    videos = load_videos()
    print(f"\n📁 {len(videos)}개 영상 발견")
    
    # 처리
    updated_count = 0
    
    for video in videos:
        video_id = video.get("video_id", "")
        
        if target_video_id and video_id != target_video_id:
            continue
        
        print(f"\n{'─' * 50}")
        
        success = generate_timestamps_for_video(service, video, force=force)
        
        if success:
            updated_count += 1
    
    # 저장
    print(f"\n{'=' * 60}")
    if updated_count > 0:
        print(f"💾 {updated_count}개 영상 업데이트 중...")
        save_videos(videos)
    else:
        print("⚠️  업데이트할 영상이 없습니다.")
    
    print("\n✨ 완료!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Forced Alignment 타임스탬프 생성")
    parser.add_argument(
        "--video-id", 
        type=str, 
        default=None,
        help="특정 영상 ID만 처리 (예: v_004)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 타임스탬프가 있어도 모두 재분석"
    )
    
    args = parser.parse_args()
    main(target_video_id=args.video_id, force=args.force)
