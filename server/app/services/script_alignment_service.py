"""
Script Alignment Service
========================
whisper-timestamped를 사용하여 스크립트 텍스트를 오디오와 정렬하고
단어 수준의 타임스탬프를 추출합니다.

Requirements:
    pip install whisper-timestamped
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 스크립트 데이터 경로
BASE_DIR = Path(__file__).parent.parent
VIDEOS_FILE = BASE_DIR / "database" / "actor_videos.json"


def load_video_scripts() -> dict[str, str]:
    """
    actor_videos.json에서 모든 영상의 스크립트를 로드합니다.
    
    Returns:
        {video_url: script_text} 매핑 딕셔너리
    """
    if not VIDEOS_FILE.exists():
        logger.warning(f"영상 데이터 파일을 찾을 수 없습니다: {VIDEOS_FILE}")
        return {}
    
    with open(VIDEOS_FILE, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    return {
        video.get("video_url", ""): video.get("script", "")
        for video in videos
        if video.get("script")
    }


def get_script_by_video_url(video_url: str) -> Optional[str]:
    """
    영상 URL로 해당 영상의 스크립트를 조회합니다.
    
    Args:
        video_url: 영상 URL (예: /assets/videos/어이가없네.mp4)
        
    Returns:
        스크립트 텍스트 또는 None
    """
    scripts = load_video_scripts()
    return scripts.get(video_url)


def get_script_by_filename(filename: str) -> Optional[str]:
    """
    영상 파일명으로 해당 영상의 스크립트를 조회합니다.
    
    Args:
        filename: 영상 파일명 (예: 어이가없네.mp4)
        
    Returns:
        스크립트 텍스트 또는 None
    """
    scripts = load_video_scripts()
    
    # 파일명으로 매칭
    for video_url, script in scripts.items():
        if video_url.endswith(filename):
            return script
    
    return None


class ScriptAlignmentService:
    """
    whisper-timestamped를 사용한 스크립트-오디오 정렬 서비스.
    
    주어진 스크립트 텍스트를 initial_prompt로 사용하여
    Whisper 모델이 스크립트에 맞는 단어를 인식하도록 유도합니다.
    """
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        ScriptAlignmentService 초기화.
        
        Args:
            model_name: Whisper 모델 이름 (tiny, base, small, medium, large)
            device: 실행 디바이스 ("cpu" 또는 "cuda")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        
    def _load_model(self):
        """Whisper 모델을 로드합니다 (지연 로딩)."""
        if self._model is None:
            try:
                import whisper_timestamped as whisper
                
                logger.info(f"Whisper 모델 로딩 중: {self.model_name} (device={self.device})")
                self._model = whisper.load_model(self.model_name, device=self.device)
                logger.info("Whisper 모델 로딩 완료")
                
            except ImportError:
                raise ImportError(
                    "whisper-timestamped가 설치되지 않았습니다. "
                    "다음 명령어로 설치하세요: pip install whisper-timestamped"
                )
        return self._model
    
    def align(
        self, 
        audio_path: str, 
        script_text: str,
        language: str = "ko",
        use_vad: bool = True,
        transcribe_options: Optional[dict] = None
    ) -> list[dict]:
        """
        오디오 파일과 스크립트 텍스트를 정렬하여 단어별 타임스탬프를 추출합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            script_text: 정렬할 스크립트 텍스트 (ground truth)
            language: 언어 코드 (기본값: "ko" - 한국어)
            use_vad: VAD(Voice Activity Detection) 사용 여부
            transcribe_options: whisper.transcribe 추가 옵션 (지원되는 키만 적용)
            
        Returns:
            단어별 타임스탬프 리스트:
            [
                {"text": "어이가", "start": 1.2, "end": 1.5},
                {"text": "없네", "start": 1.6, "end": 1.9},
                ...
            ]
        """
        # 스크립트 텍스트 전처리 (앞뒤 공백 제거)
        script_text = script_text.strip()
        
        logger.info(f"오디오 파일 처리 중: {audio_path}")
        logger.info(f"스크립트 힌트: {script_text[:50]}..." if len(script_text) > 50 else f"스크립트 힌트: {script_text}")
        
        try:
            # whisper-timestamped로 전사 수행
            result = self._transcribe(
                audio_path=audio_path,
                script_text=script_text,
                language=language,
                use_vad=use_vad,
                transcribe_options=transcribe_options,
            )
            
        except Exception as e:
            logger.error(f"전사 중 오류 발생: {e}")
            raise RuntimeError(f"오디오 전사 실패: {e}")
        
        # 결과 파싱
        word_timestamps = self._parse_result(result)
        
        # 스크립트와의 일치도 검증
        self._validate_alignment(script_text, word_timestamps)
        
        return word_timestamps
    
    def _parse_result(self, result: dict) -> list[dict]:
        """
        whisper-timestamped 결과에서 단어별 타임스탬프를 추출합니다.
        
        Args:
            result: whisper.transcribe() 반환값
            
        Returns:
            단어별 타임스탬프 리스트
        """
        word_timestamps = []
        
        segments = result.get("segments", [])
        
        for segment in segments:
            words = segment.get("words", [])
            
            for word_info in words:
                word_data = {
                    "text": word_info.get("text", "").strip(),
                    "start": round(word_info.get("start", 0.0), 3),
                    "end": round(word_info.get("end", 0.0), 3),
                }
                
                # 빈 텍스트는 건너뛰기
                if word_data["text"]:
                    word_timestamps.append(word_data)
        
        logger.info(f"추출된 단어 수: {len(word_timestamps)}")
        
        return word_timestamps

    def _transcribe(
        self,
        audio_path: str,
        script_text: str,
        language: str,
        use_vad: bool,
        transcribe_options: Optional[dict] = None,
    ) -> dict:
        """
        whisper.transcribe 호출을 공통 처리합니다.
        지원되지 않는 옵션은 자동으로 제거합니다.
        """
        import whisper_timestamped as whisper

        # 모델 로드
        model = self._load_model()

        # 품질 향상 기본 옵션
        options = {
            "language": language,
            "initial_prompt": script_text,
            "vad": use_vad,
            "compute_word_confidence": False,
            "verbose": False,
            # 안정적인 인식을 위한 기본값
            "temperature": 0.0,
            "beam_size": 5,
            "best_of": 5,
            "condition_on_previous_text": False,
        }

        if transcribe_options:
            options.update(transcribe_options)

        # whisper_timestamped 버전에 따라 미지원 옵션 필터링
        sig = inspect.signature(whisper.transcribe)
        filtered = {k: v for k, v in options.items() if k in sig.parameters}

        return whisper.transcribe(
            model,
            audio_path,
            **filtered,
        )
    
    def _validate_alignment(
        self, 
        script_text: str, 
        word_timestamps: list[dict],
        tolerance: float = 0.3
    ) -> None:
        """
        추출된 결과가 원본 스크립트와 일치하는지 검증합니다.
        
        Args:
            script_text: 원본 스크립트 텍스트
            word_timestamps: 추출된 단어 타임스탬프 리스트
            tolerance: 허용 오차 비율 (0.3 = 30%)
        """
        # 스크립트에서 단어 추출 (공백 기준)
        script_words = script_text.split()
        extracted_words = [w["text"] for w in word_timestamps]
        
        script_word_count = len(script_words)
        extracted_word_count = len(extracted_words)
        
        # 단어 수 비교
        if script_word_count == 0:
            logger.warning("스크립트가 비어있습니다.")
            return
            
        diff_ratio = abs(script_word_count - extracted_word_count) / script_word_count
        
        if diff_ratio > tolerance:
            logger.warning(
                f"단어 수 불일치 감지: "
                f"스크립트={script_word_count}, 추출={extracted_word_count} "
                f"(차이: {diff_ratio:.1%})"
            )
            logger.warning(f"스크립트 단어: {script_words[:10]}...")
            logger.warning(f"추출된 단어: {extracted_words[:10]}...")
        else:
            logger.info(f"정렬 검증 완료: 단어 수 일치율 양호 (차이: {diff_ratio:.1%})")
    
    def align_with_segments(
        self,
        audio_path: str,
        script_text: str,
        language: str = "ko",
        use_vad: bool = True,
        transcribe_options: Optional[dict] = None
    ) -> dict:
        """
        세그먼트 정보를 포함한 상세 정렬 결과를 반환합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            script_text: 정렬할 스크립트 텍스트
            language: 언어 코드
            use_vad: VAD(Voice Activity Detection) 사용 여부
            transcribe_options: whisper.transcribe 추가 옵션 (지원되는 키만 적용)
            
        Returns:
            {
                "words": [{"text": "...", "start": 0.0, "end": 0.0}, ...],
                "segments": [{"text": "...", "start": 0.0, "end": 0.0, "words": [...]}, ...],
                "full_text": "전체 텍스트"
            }
        """
        script_text = script_text.strip()

        result = self._transcribe(
            audio_path=audio_path,
            script_text=script_text,
            language=language,
            use_vad=use_vad,
            transcribe_options=transcribe_options,
        )
        
        # 단어 추출
        words = self._parse_result(result)
        
        # 세그먼트 추출
        segments = []
        for seg in result.get("segments", []):
            segment_words = []
            for w in seg.get("words", []):
                if w.get("text", "").strip():
                    segment_words.append({
                        "text": w.get("text", "").strip(),
                        "start": round(w.get("start", 0.0), 3),
                        "end": round(w.get("end", 0.0), 3),
                    })
            
            segments.append({
                "text": seg.get("text", "").strip(),
                "start": round(seg.get("start", 0.0), 3),
                "end": round(seg.get("end", 0.0), 3),
                "words": segment_words,
            })
        
        return {
            "words": words,
            "segments": segments,
            "full_text": result.get("text", "").strip(),
        }


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 서비스 인스턴스 생성
    service = ScriptAlignmentService(model_name="base", device="cpu")
    
    # 예시 사용법
    audio_path = "path/to/audio.wav"
    script_text = "어이가 없네 진짜 대체 왜 이러는 거야"
    
    # 단어별 타임스탬프 추출
    # word_timestamps = service.align(audio_path, script_text)
    # 
    # for word in word_timestamps:
    #     print(f"{word['text']}: {word['start']:.2f}s - {word['end']:.2f}s")
    
    print("ScriptAlignmentService가 준비되었습니다.")
    print("사용법: service.align(audio_path, script_text)")
