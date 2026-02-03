"""
Acting Analysis Pipeline (통합 파이프라인)

Stage 1~3을 통합하여 한 번의 호출로 연기 분석을 완료합니다.

사용 흐름:
1. 서버 시작 시 assets 폴더 스캔 → 새 MP4 자동 분석
2. 사용자 영상 업로드 시 실시간 분석 + 스코어링

Example:
    pipeline = get_acting_pipeline()
    
    # 서버 시작 시 자동 동기화 (새 MP4만 분석)
    pipeline.sync_assets()
    
    # 사용자 평가 (매 요청마다)
    result = pipeline.evaluate_user("user_video.webm", "어이가없네", "user_001")
    print(result.total_score)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from app.core.schemas import AnalysisResult, DTWResult, ScoringResult
from app.services.dtw_service import DTWService, get_dtw_service
from app.services.reference_analysis_service import (
    ReferenceAnalysisService,
    get_reference_analysis_service,
)
from app.services.scoring_service import ScoringService, get_scoring_service


def sanitize_actor_id(filename: str) -> str:
    """
    파일명을 actor_id로 변환.
    
    - 확장자 제거
    - 공백을 언더스코어로 변환
    - 특수문자 제거 (한글, 영문, 숫자, 언더스코어만 허용)
    """
    # 확장자 제거
    name = Path(filename).stem
    # 공백 → 언더스코어
    name = name.replace(" ", "_")
    # 허용된 문자만 남기기 (한글, 영문, 숫자, 언더스코어)
    name = re.sub(r'[^\w가-힣]', '_', name)
    # 연속 언더스코어 정리
    name = re.sub(r'_+', '_', name)
    # 앞뒤 언더스코어 제거
    name = name.strip('_')
    return name or "unknown"


class ActingAnalysisPipeline:
    """
    연기 분석 통합 파이프라인.
    
    레퍼런스 준비부터 사용자 평가까지 전체 프로세스를 관리합니다.
    assets 폴더의 MP4 파일을 자동으로 스캔하고 분석합니다.
    """

    # 지원하는 비디오 확장자
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}

    def __init__(
        self,
        # NOTE: 레퍼런스 비디오는 assets/videos 아래에 둡니다.
        assets_dir: str | Path = "assets/videos",
        reference_dir: str | Path = "data/references",
        analysis_service: Optional[ReferenceAnalysisService] = None,
        dtw_service: Optional[DTWService] = None,
        scoring_service: Optional[ScoringService] = None,
    ):
        """
        Args:
            assets_dir: 레퍼런스 비디오 파일이 있는 디렉토리 (기본: assets/videos)
            reference_dir: 분석 결과 파일 저장 디렉토리
            analysis_service: 분석 서비스 (None이면 기본 인스턴스)
            dtw_service: DTW 서비스 (None이면 기본 인스턴스)
            scoring_service: 스코어링 서비스 (None이면 기본 인스턴스)
        """
        self.assets_dir = Path(assets_dir)
        self.reference_dir = Path(reference_dir)
        
        # 디렉토리 생성
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)

        self.analysis_service = analysis_service or get_reference_analysis_service()
        self.dtw_service = dtw_service or get_dtw_service()
        self.scoring_service = scoring_service or get_scoring_service()

        # 캐시: 로드된 레퍼런스 데이터
        self._reference_cache: dict[str, AnalysisResult] = {}
        
        # actor_id → 원본 파일 경로 매핑
        self._actor_to_file: dict[str, Path] = {}

        print(f"🎬 ActingAnalysisPipeline: 초기화 완료")
        print(f"   Assets 디렉토리: {self.assets_dir}")
        print(f"   분석결과 디렉토리: {self.reference_dir}")

    # =========================================================================
    # Assets 폴더 자동 동기화
    # =========================================================================

    def scan_assets(self) -> dict[str, Path]:
        """
        assets 폴더의 비디오 파일을 스캔합니다.
        
        Returns:
            {actor_id: video_path} 매핑
        """
        videos = {}
        
        for ext in self.VIDEO_EXTENSIONS:
            for video_path in self.assets_dir.glob(f"*{ext}"):
                if video_path.is_file():
                    actor_id = sanitize_actor_id(video_path.name)
                    videos[actor_id] = video_path
                    
        self._actor_to_file = videos
        return videos

    def get_pending_analyses(self) -> list[tuple[str, Path]]:
        """
        분석이 필요한 (아직 분석되지 않은) 비디오 목록 반환.
        
        Returns:
            [(actor_id, video_path), ...] 리스트
        """
        videos = self.scan_assets()
        pending = []
        
        for actor_id, video_path in videos.items():
            ref_path = self._get_reference_path(actor_id)
            
            # 분석 결과가 없거나, 원본 비디오가 더 최신이면 재분석 필요
            if not ref_path.exists():
                pending.append((actor_id, video_path))
            elif video_path.stat().st_mtime > ref_path.stat().st_mtime:
                # 비디오가 수정되었으면 재분석
                pending.append((actor_id, video_path))
                
        return pending

    def sync_assets(self, force: bool = False) -> dict[str, str]:
        """
        assets 폴더를 스캔하고 새로운/변경된 비디오를 분석합니다.
        
        Args:
            force: True면 모든 비디오를 재분석
            
        Returns:
            {actor_id: status} 결과 딕셔너리
            status: "analyzed" | "skipped" | "error: ..."
        """
        results = {}
        videos = self.scan_assets()
        
        if not videos:
            print("📂 assets 폴더에 비디오 파일이 없습니다.")
            return results
        
        print(f"\n📂 Assets 동기화 시작 ({len(videos)}개 비디오 발견)")
        print("=" * 50)
        
        for actor_id, video_path in videos.items():
            ref_path = self._get_reference_path(actor_id)
            
            # 이미 분석된 경우 스킵 (force가 아닐 때)
            if ref_path.exists() and not force:
                # 비디오가 수정되었는지 확인
                if video_path.stat().st_mtime <= ref_path.stat().st_mtime:
                    print(f"  ⏭️  {actor_id}: 이미 분석됨 (스킵)")
                    results[actor_id] = "skipped"
                    continue
                else:
                    print(f"  🔄 {actor_id}: 비디오 변경됨 (재분석)")
            
            # 분석 실행
            try:
                print(f"  🔬 {actor_id}: 분석 중...")
                self.prepare_reference(video_path, actor_id, force=True)
                results[actor_id] = "analyzed"
                print(f"  ✅ {actor_id}: 분석 완료")
            except Exception as e:
                results[actor_id] = f"error: {str(e)}"
                print(f"  ❌ {actor_id}: 분석 실패 - {e}")
        
        # 요약
        analyzed = sum(1 for s in results.values() if s == "analyzed")
        skipped = sum(1 for s in results.values() if s == "skipped")
        errors = sum(1 for s in results.values() if s.startswith("error"))
        
        print("=" * 50)
        print(f"📊 동기화 완료: 분석 {analyzed}개, 스킵 {skipped}개, 오류 {errors}개\n")
        
        return results

    def get_reference_info(self, actor_id: str) -> dict:
        """
        레퍼런스 상세 정보 반환 (원본 파일 정보 포함).
        """
        ref = self.load_reference(actor_id)
        video_path = self._actor_to_file.get(actor_id)
        
        return {
            "actor_id": ref.actor_id,
            "source_file": ref.source_file,
            "video_path": str(video_path) if video_path else None,
            "duration_sec": ref.duration_sec,
            "fps": ref.fps,
            "sampling_rate": ref.sampling_rate,
            "frame_count": ref.frame_count,
            "has_audio": ref.has_audio,
            "has_video": ref.has_video,
        }

    def list_assets(self) -> list[dict]:
        """
        assets 폴더의 모든 비디오와 분석 상태 반환.
        """
        videos = self.scan_assets()
        result = []
        
        for actor_id, video_path in videos.items():
            ref_path = self._get_reference_path(actor_id)
            is_analyzed = ref_path.exists()
            
            info = {
                "actor_id": actor_id,
                "filename": video_path.name,
                "is_analyzed": is_analyzed,
            }
            
            if is_analyzed:
                try:
                    ref = self.load_reference(actor_id)
                    info["duration_sec"] = ref.duration_sec
                    info["frame_count"] = ref.frame_count
                except Exception:
                    info["error"] = "분석 파일 로드 실패"
            
            result.append(info)
        
        return result

    # =========================================================================
    # 레퍼런스 관리
    # =========================================================================

    def prepare_reference(
        self,
        video_path: str | Path,
        actor_id: str,
        force: bool = False,
    ) -> AnalysisResult:
        """
        레퍼런스 영상을 분석하고 저장합니다.
        
        Args:
            video_path: 레퍼런스 MP4 파일 경로
            actor_id: 배우/영상 고유 ID
            force: True면 기존 파일을 덮어씁니다
            
        Returns:
            AnalysisResult: 분석 결과
        """
        output_path = self._get_reference_path(actor_id)

        # 이미 존재하면 로드
        if output_path.exists() and not force:
            print(f"📂 기존 레퍼런스 사용: {actor_id}")
            return self.load_reference(actor_id)

        # 분석 실행
        print(f"🔬 레퍼런스 분석 시작: {actor_id}")
        result = self.analysis_service.analyze(
            video_path=video_path,
            actor_id=actor_id,
            output_path=output_path,
        )

        # 캐시에 저장
        self._reference_cache[actor_id] = result

        return result

    def load_reference(self, actor_id: str) -> AnalysisResult:
        """
        저장된 레퍼런스 데이터를 로드합니다.
        
        Args:
            actor_id: 배우/영상 고유 ID
            
        Returns:
            AnalysisResult: 분석 결과
            
        Raises:
            FileNotFoundError: 레퍼런스가 없는 경우
        """
        # 캐시 확인
        if actor_id in self._reference_cache:
            return self._reference_cache[actor_id]

        # 파일에서 로드
        ref_path = self._get_reference_path(actor_id)
        if not ref_path.exists():
            raise FileNotFoundError(f"레퍼런스를 찾을 수 없습니다: {actor_id}")

        result = AnalysisResult.load(ref_path)
        self._reference_cache[actor_id] = result

        return result

    def list_references(self) -> list[str]:
        """저장된 레퍼런스 ID 목록 반환."""
        return [
            p.stem for p in self.reference_dir.glob("*.msgpack")
        ]

    def _get_reference_path(self, actor_id: str) -> Path:
        """레퍼런스 파일 경로 반환."""
        return self.reference_dir / f"{actor_id}.msgpack"

    # =========================================================================
    # 사용자 평가
    # =========================================================================

    def evaluate_user(
        self,
        user_video: str | Path,
        actor_id: str,
        user_id: str = "user",
    ) -> ScoringResult:
        """
        사용자 영상을 평가합니다.
        
        Args:
            user_video: 사용자 영상 파일 경로
            actor_id: 비교할 레퍼런스 배우 ID
            user_id: 사용자 ID
            
        Returns:
            ScoringResult: 평가 결과
        """
        # 1. 레퍼런스 로드
        reference = self.load_reference(actor_id)

        # 2. 사용자 영상 분석
        print(f"🔬 사용자 영상 분석 중...")
        user_analysis = self.analysis_service.analyze(
            video_path=user_video,
            actor_id=user_id,
            output_path=None,  # 저장 안 함
        )

        # 3. DTW 동기화
        print(f"🔗 DTW 동기화 중...")
        dtw_result = self.dtw_service.synchronize(
            user_audio=user_video,  # 비디오에서 오디오 추출
            reference=reference,
            user_id=user_id,
        )

        # 4. 스코어링
        print(f"📊 스코어링 중...")
        result = self.scoring_service.score(
            user_analysis=user_analysis,
            reference=reference,
            dtw_result=dtw_result,
        )

        return result

    def evaluate_user_with_details(
        self,
        user_video: str | Path,
        actor_id: str,
        user_id: str = "user",
    ) -> tuple[ScoringResult, AnalysisResult, DTWResult]:
        """
        사용자 영상을 평가하고 상세 데이터도 반환합니다.
        
        Returns:
            (ScoringResult, user_analysis, dtw_result)
        """
        reference = self.load_reference(actor_id)

        user_analysis = self.analysis_service.analyze(
            video_path=user_video,
            actor_id=user_id,
            output_path=None,
        )

        dtw_result = self.dtw_service.synchronize(
            user_audio=user_video,
            reference=reference,
            user_id=user_id,
        )

        result = self.scoring_service.score(
            user_analysis=user_analysis,
            reference=reference,
            dtw_result=dtw_result,
        )

        return result, user_analysis, dtw_result

    # =========================================================================
    # 빠른 평가 (오디오만)
    # =========================================================================

    def evaluate_audio_only(
        self,
        user_audio: str | Path,
        actor_id: str,
        user_id: str = "user",
    ) -> dict:
        """
        오디오만으로 빠르게 평가합니다 (비디오 처리 생략).
        
        Args:
            user_audio: 사용자 오디오 파일 경로
            actor_id: 비교할 레퍼런스 배우 ID
            user_id: 사용자 ID
            
        Returns:
            {
                "sync_score": float,  # DTW 동기화 점수
                "confidence": float,  # 신뢰도
            }
        """
        from app.services.dtw_service import AdvancedDTWService

        reference = self.load_reference(actor_id)

        advanced_dtw = AdvancedDTWService()
        dtw_result, confidence = advanced_dtw.synchronize_with_confidence(
            user_audio=user_audio,
            reference=reference,
            user_id=user_id,
        )

        # 동기화 점수 (normalized_distance의 역수)
        sync_score = max(0, 100 * (1 - dtw_result.normalized_distance * 2))

        return {
            "sync_score": round(sync_score, 1),
            "confidence": round(confidence, 1),
            "dtw_distance": round(dtw_result.distance, 4),
        }


# =============================================================================
# 전역 인스턴스 (Lazy Loading)
# =============================================================================

_acting_pipeline: Optional[ActingAnalysisPipeline] = None


def get_acting_pipeline() -> ActingAnalysisPipeline:
    """ActingAnalysisPipeline 싱글톤 인스턴스 반환."""
    global _acting_pipeline
    if _acting_pipeline is None:
        _acting_pipeline = ActingAnalysisPipeline()
    return _acting_pipeline


# =============================================================================
# API 응답용 헬퍼 함수
# =============================================================================

def _score_detail_to_dict(detail) -> dict:
    """ScoreDetail을 딕셔너리로 변환 (서브메트릭 포함)."""
    if not detail:
        return {
            "score": 0,
            "feedback": "",
            "weight": 0,
            "sub_metrics": [],
        }
    
    sub_metrics = []
    for sm in detail.sub_metrics:
        sub_metrics.append({
            "name": sm.name,
            "score": sm.score,
            "weight": sm.weight,
            "feedback": sm.feedback,
            "details": sm.details,
        })
    
    return {
        "score": detail.score,
        "feedback": detail.feedback,
        "weight": detail.weight,
        "sub_metrics": sub_metrics,
    }


def scoring_result_to_dict(result: ScoringResult) -> dict:
    """ScoringResult를 API 응답용 딕셔너리로 변환 (Ultra-Precision 서브메트릭 포함)."""
    return {
        "total_score": result.total_score,
        "grade": _score_to_grade(result.total_score),
        "details": {
            "pitch": _score_detail_to_dict(result.audio_pitch_score),
            "energy": _score_detail_to_dict(result.audio_energy_score),
            "expression": _score_detail_to_dict(result.video_expression_score),
        },
        "overall_feedback": result.overall_feedback,
    }


def _score_to_grade(score: float) -> str:
    """점수를 등급 문자열로 변환."""
    if score >= 90:
        return "S"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


# =============================================================================
# CLI 테스트용
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🎬 Acting Analysis Pipeline - CLI")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\n사용법:")
        print("  1. Assets 동기화 (새 MP4 자동 분석):")
        print("     python -m app.services.acting_analysis_pipeline sync")
        print("     python -m app.services.acting_analysis_pipeline sync --force")
        print("")
        print("  2. Assets 상태 확인:")
        print("     python -m app.services.acting_analysis_pipeline status")
        print("")
        print("  3. 개별 레퍼런스 준비:")
        print("     python -m app.services.acting_analysis_pipeline prepare <video.mp4> <actor_id>")
        print("")
        print("  4. 사용자 평가:")
        print("     python -m app.services.acting_analysis_pipeline evaluate <user_video.webm> <actor_id>")
        print("")
        print("  5. 레퍼런스 목록:")
        print("     python -m app.services.acting_analysis_pipeline list")
        sys.exit(1)

    command = sys.argv[1]
    pipeline = get_acting_pipeline()

    if command == "sync":
        # Assets 폴더 동기화
        force = "--force" in sys.argv or "-f" in sys.argv
        if force:
            print("⚠️  강제 모드: 모든 비디오를 재분석합니다.")
        results = pipeline.sync_assets(force=force)

    elif command == "status":
        # Assets 상태 확인
        assets = pipeline.list_assets()
        print(f"\n📂 Assets 상태 ({len(assets)}개 비디오)")
        print("-" * 50)
        for item in assets:
            status = "✅ 분석됨" if item["is_analyzed"] else "⏳ 대기중"
            duration = f" ({item.get('duration_sec', 0):.1f}초)" if item.get("duration_sec") else ""
            print(f"  {status} {item['actor_id']}: {item['filename']}{duration}")
        
        # 요약
        analyzed = sum(1 for a in assets if a["is_analyzed"])
        pending = len(assets) - analyzed
        print("-" * 50)
        print(f"  분석됨: {analyzed}개, 대기중: {pending}개")

    elif command == "prepare":
        if len(sys.argv) < 4:
            print("Usage: prepare <video.mp4> <actor_id>")
            sys.exit(1)
        
        video_path = sys.argv[2]
        actor_id = sys.argv[3]
        
        result = pipeline.prepare_reference(video_path, actor_id, force=True)
        print(f"\n✅ 레퍼런스 준비 완료: {actor_id}")
        print(f"   프레임 수: {result.frame_count}")
        print(f"   길이: {result.duration_sec:.2f}초")

    elif command == "evaluate":
        if len(sys.argv) < 4:
            print("Usage: evaluate <user_video.webm> <actor_id>")
            sys.exit(1)
        
        user_video = sys.argv[2]
        actor_id = sys.argv[3]
        
        result = pipeline.evaluate_user(user_video, actor_id)
        
        print("\n" + "=" * 60)
        print("🎭 연기 평가 결과")
        print("=" * 60)
        print(f"\n  📊 종합 점수: {result.total_score:.1f}/100 ({_score_to_grade(result.total_score)})")
        
        if result.audio_pitch_score:
            print(f"\n  🎤 억양: {result.audio_pitch_score.score:.1f}/100")
            print(f"     → {result.audio_pitch_score.feedback}")
        
        if result.audio_energy_score:
            print(f"\n  🔊 볼륨: {result.audio_energy_score.score:.1f}/100")
            print(f"     → {result.audio_energy_score.feedback}")
        
        if result.video_expression_score:
            print(f"\n  😀 표정: {result.video_expression_score.score:.1f}/100")
            print(f"     → {result.video_expression_score.feedback}")
        
        print(f"\n  💬 종합 피드백:")
        print(f"     {result.overall_feedback}")

    elif command == "list":
        refs = pipeline.list_references()
        print(f"\n📂 저장된 레퍼런스 ({len(refs)}개):")
        for ref_id in refs:
            print(f"   - {ref_id}")

    else:
        print(f"알 수 없는 명령어: {command}")
        sys.exit(1)
