"""
Scoring Service (Stage 3)

DTW 동기화 결과를 사용하여 사용자의 연기를 레퍼런스 배우와 비교 평가합니다.

평가 항목:
- Audio: 피치(억양) + 에너지(볼륨) 패턴 유사도
- Video: 블렌드쉐입(표정) 유사도

핵심 특징:
- DTW 워핑 경로를 사용한 정확한 프레임 대 프레임 비교
- 정규화된 비교로 개인 차이(음역대, 볼륨 등) 보정
- 항목별 세부 피드백 제공
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from app.core.schemas import (
    AnalysisResult,
    DTWResult,
    ScoreDetail,
    ScoringResult,
)


class ScoreGrade(Enum):
    """점수 등급."""
    S = "S"   # 90-100: 완벽
    A = "A"   # 80-89: 우수
    B = "B"   # 70-79: 양호
    C = "C"   # 60-69: 보통
    D = "D"   # 50-59: 미흡
    F = "F"   # 0-49: 노력 필요


@dataclass
class FrameScore:
    """프레임별 점수 (디버깅/시각화용)."""
    user_frame_idx: int
    actor_frame_idx: int
    timestamp_ms: int
    pitch_score: float = 0.0
    energy_score: float = 0.0
    expression_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class DetailedFeedback:
    """상세 피드백."""
    category: str
    score: float
    grade: ScoreGrade
    message: str
    suggestions: list[str] = field(default_factory=list)
    frame_scores: list[float] = field(default_factory=list)  # 시계열 점수


class ScoringService:
    """
    연기 스코어링 서비스.
    
    DTW 동기화 결과를 사용하여 사용자와 레퍼런스를 비교 평가합니다.
    """

    # 가중치 설정 (합계 = 1.0)
    WEIGHT_PITCH = 0.30      # 억양 (대사 전달에 중요)
    WEIGHT_ENERGY = 0.20     # 볼륨/강세
    WEIGHT_EXPRESSION = 0.50  # 표정 (연기의 핵심)

    # 블렌드쉐입별 가중치 (연기에서 중요한 표정)
    BLENDSHAPE_WEIGHTS = {
        # 입 관련 (대사 전달)
        "jawOpen": 1.5,
        "mouthSmileLeft": 1.2,
        "mouthSmileRight": 1.2,
        "mouthFrownLeft": 1.2,
        "mouthFrownRight": 1.2,
        "mouthPucker": 1.0,
        "mouthLeft": 0.8,
        "mouthRight": 0.8,
        # 눈썹 (감정 표현)
        "browInnerUp": 1.3,
        "browDownLeft": 1.2,
        "browDownRight": 1.2,
        "browOuterUpLeft": 1.0,
        "browOuterUpRight": 1.0,
        # 눈 (감정의 창)
        "eyeWideLeft": 1.3,
        "eyeWideRight": 1.3,
        "eyeSquintLeft": 1.1,
        "eyeSquintRight": 1.1,
        "eyeBlinkLeft": 0.5,   # 눈 깜빡임은 낮은 가중치
        "eyeBlinkRight": 0.5,
        # 기타
        "cheekPuff": 0.8,
        "noseSneerLeft": 0.9,
        "noseSneerRight": 0.9,
    }

    def __init__(
        self,
        pitch_weight: float = 0.30,
        energy_weight: float = 0.20,
        expression_weight: float = 0.50,
    ):
        """
        Args:
            pitch_weight: 피치 점수 가중치
            energy_weight: 에너지 점수 가중치
            expression_weight: 표정 점수 가중치
        """
        total = pitch_weight + energy_weight + expression_weight
        self.weight_pitch = pitch_weight / total
        self.weight_energy = energy_weight / total
        self.weight_expression = expression_weight / total

        print(f"📊 ScoringService: 초기화 완료")
        print(f"   가중치 - 피치: {self.weight_pitch:.0%}, "
              f"에너지: {self.weight_energy:.0%}, "
              f"표정: {self.weight_expression:.0%}")

    def score(
        self,
        user_analysis: AnalysisResult,
        reference: AnalysisResult,
        dtw_result: DTWResult,
    ) -> ScoringResult:
        """
        사용자 연기 평가.
        
        Args:
            user_analysis: 사용자 분석 결과 (Stage 1에서 생성)
            reference: 레퍼런스 배우 분석 결과
            dtw_result: DTW 동기화 결과 (Stage 2에서 생성)
            
        Returns:
            ScoringResult: 종합 평가 결과
        """
        # 워핑 경로에서 정렬된 프레임 쌍 추출
        aligned_pairs = dtw_result.warping_path

        if not aligned_pairs:
            return self._empty_result(dtw_result)

        # 1. 오디오 점수 계산
        pitch_detail = self._score_pitch(
            user_analysis, reference, aligned_pairs
        )
        energy_detail = self._score_energy(
            user_analysis, reference, aligned_pairs
        )

        # 2. 비디오(표정) 점수 계산
        expression_detail = self._score_expression(
            user_analysis, reference, aligned_pairs
        )

        # 3. 종합 점수 계산
        total_score = (
            pitch_detail.score * self.weight_pitch +
            energy_detail.score * self.weight_energy +
            expression_detail.score * self.weight_expression
        )

        # 4. 종합 피드백 생성
        overall_feedback = self._generate_overall_feedback(
            total_score, pitch_detail, energy_detail, expression_detail
        )

        return ScoringResult(
            total_score=round(total_score, 1),
            audio_pitch_score=pitch_detail,
            audio_energy_score=energy_detail,
            video_expression_score=expression_detail,
            dtw_result=dtw_result,
            overall_feedback=overall_feedback,
        )

    def score_with_details(
        self,
        user_analysis: AnalysisResult,
        reference: AnalysisResult,
        dtw_result: DTWResult,
    ) -> tuple[ScoringResult, list[FrameScore]]:
        """
        상세 프레임별 점수와 함께 평가.
        
        Returns:
            (ScoringResult, frame_scores)
        """
        result = self.score(user_analysis, reference, dtw_result)
        frame_scores = self._calculate_frame_scores(
            user_analysis, reference, dtw_result.warping_path
        )
        return result, frame_scores

    # =========================================================================
    # 오디오 스코어링
    # =========================================================================

    def _score_pitch(
        self,
        user: AnalysisResult,
        ref: AnalysisResult,
        aligned_pairs: list[tuple[int, int]],
    ) -> ScoreDetail:
        """
        피치(억양) 패턴 유사도 평가.
        
        절대 피치가 아닌 상대적 피치 변화 패턴을 비교합니다.
        """
        user_pitches = []
        ref_pitches = []

        for user_idx, ref_idx in aligned_pairs:
            user_frame = user.frames[user_idx] if user_idx < len(user.frames) else None
            ref_frame = ref.frames[ref_idx] if ref_idx < len(ref.frames) else None

            if user_frame and user_frame.audio and ref_frame and ref_frame.audio:
                # 유성음 구간만 비교 (무음/무성음 제외)
                if user_frame.audio.is_voiced and ref_frame.audio.is_voiced:
                    user_pitches.append(user_frame.audio.pitch)
                    ref_pitches.append(ref_frame.audio.pitch)

        if len(user_pitches) < 5:
            return ScoreDetail(
                score=50.0,
                weight=self.weight_pitch,
                feedback="음성 데이터가 부족하여 정확한 평가가 어렵습니다.",
            )

        # 피치를 상대적 변화율로 변환 (델타 피치)
        user_delta = self._compute_delta(np.array(user_pitches))
        ref_delta = self._compute_delta(np.array(ref_pitches))

        # Z-score 정규화 후 비교
        user_norm = self._zscore_normalize(user_delta)
        ref_norm = self._zscore_normalize(ref_delta)

        # 코사인 유사도 계산
        similarity = self._cosine_similarity(user_norm, ref_norm)
        
        # 0-100 점수로 변환
        score = max(0.0, min(100.0, similarity * 100))

        # 피드백 생성
        feedback = self._generate_pitch_feedback(score, user_pitches, ref_pitches)

        return ScoreDetail(
            score=round(score, 1),
            weight=self.weight_pitch,
            feedback=feedback,
        )

    def _score_energy(
        self,
        user: AnalysisResult,
        ref: AnalysisResult,
        aligned_pairs: list[tuple[int, int]],
    ) -> ScoreDetail:
        """
        에너지(볼륨) 패턴 유사도 평가.
        
        강세와 볼륨 변화 패턴을 비교합니다.
        """
        user_energies = []
        ref_energies = []

        for user_idx, ref_idx in aligned_pairs:
            user_frame = user.frames[user_idx] if user_idx < len(user.frames) else None
            ref_frame = ref.frames[ref_idx] if ref_idx < len(ref.frames) else None

            if user_frame and user_frame.audio and ref_frame and ref_frame.audio:
                user_energies.append(user_frame.audio.energy)
                ref_energies.append(ref_frame.audio.energy)

        if len(user_energies) < 5:
            return ScoreDetail(
                score=50.0,
                weight=self.weight_energy,
                feedback="음성 데이터가 부족하여 정확한 평가가 어렵습니다.",
            )

        # Min-Max 정규화 (볼륨 차이 보정)
        user_norm = self._minmax_normalize(np.array(user_energies))
        ref_norm = self._minmax_normalize(np.array(ref_energies))

        # 상관계수 기반 유사도
        correlation = np.corrcoef(user_norm, ref_norm)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # 0-100 점수로 변환 (상관계수 -1~1 → 0~100)
        score = max(0.0, min(100.0, (correlation + 1) * 50))

        # 피드백 생성
        feedback = self._generate_energy_feedback(score, user_energies, ref_energies)

        return ScoreDetail(
            score=round(score, 1),
            weight=self.weight_energy,
            feedback=feedback,
        )

    # =========================================================================
    # 비디오(표정) 스코어링
    # =========================================================================

    def _score_expression(
        self,
        user: AnalysisResult,
        ref: AnalysisResult,
        aligned_pairs: list[tuple[int, int]],
    ) -> ScoreDetail:
        """
        표정(블렌드쉐입) 유사도 평가.
        
        가중치가 적용된 블렌드쉐입 벡터를 비교합니다.
        """
        frame_scores = []
        valid_frames = 0
        face_detection_rate = 0

        for user_idx, ref_idx in aligned_pairs:
            user_frame = user.frames[user_idx] if user_idx < len(user.frames) else None
            ref_frame = ref.frames[ref_idx] if ref_idx < len(ref.frames) else None

            if not (user_frame and user_frame.video and 
                    ref_frame and ref_frame.video):
                continue

            # 얼굴 검출 여부 체크
            if user_frame.video.face_detected:
                face_detection_rate += 1

            if not (user_frame.video.blendshapes and 
                    ref_frame.video.blendshapes):
                continue

            valid_frames += 1

            # 블렌드쉐입 벡터 추출
            user_bs = user_frame.video.blendshapes.to_vector()
            ref_bs = ref_frame.video.blendshapes.to_vector()

            # 가중치 적용
            weights = self._get_blendshape_weights()
            user_weighted = np.array(user_bs) * weights
            ref_weighted = np.array(ref_bs) * weights

            # 프레임별 유사도 (코사인 유사도)
            similarity = self._cosine_similarity(user_weighted, ref_weighted)
            frame_scores.append(max(0.0, similarity))

        if valid_frames < 5:
            detection_pct = (face_detection_rate / len(aligned_pairs) * 100 
                           if aligned_pairs else 0)
            return ScoreDetail(
                score=50.0,
                weight=self.weight_expression,
                feedback=f"얼굴 인식률이 낮습니다 ({detection_pct:.0f}%). "
                        f"카메라를 정면으로 바라봐 주세요.",
            )

        # 평균 점수
        avg_score = np.mean(frame_scores) * 100

        # 피드백 생성
        feedback = self._generate_expression_feedback(
            avg_score, frame_scores, face_detection_rate / len(aligned_pairs)
        )

        return ScoreDetail(
            score=round(avg_score, 1),
            weight=self.weight_expression,
            feedback=feedback,
        )

    def _get_blendshape_weights(self) -> NDArray[np.floating]:
        """블렌드쉐입 가중치 벡터 반환 (to_vector() 순서와 동일)."""
        keys = [
            "jawOpen",
            "mouthSmileLeft", "mouthSmileRight",
            "mouthFrownLeft", "mouthFrownRight",
            "mouthPucker", "mouthLeft", "mouthRight",
            "browInnerUp", "browDownLeft", "browDownRight",
            "browOuterUpLeft", "browOuterUpRight",
            "eyeWideLeft", "eyeWideRight",
            "eyeSquintLeft", "eyeSquintRight",
            "eyeBlinkLeft", "eyeBlinkRight",
            "cheekPuff", "noseSneerLeft", "noseSneerRight",
        ]
        weights = [self.BLENDSHAPE_WEIGHTS.get(k, 1.0) for k in keys]
        # 정규화
        weights = np.array(weights)
        return weights / np.sum(weights) * len(weights)

    # =========================================================================
    # 유틸리티 함수
    # =========================================================================

    @staticmethod
    def _compute_delta(arr: NDArray[np.floating]) -> NDArray[np.floating]:
        """시계열의 변화량(델타) 계산."""
        if len(arr) < 2:
            return arr
        return np.diff(arr)

    @staticmethod
    def _zscore_normalize(arr: NDArray[np.floating]) -> NDArray[np.floating]:
        """Z-score 정규화."""
        if len(arr) == 0:
            return arr
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-8:
            return arr - mean
        return (arr - mean) / std

    @staticmethod
    def _minmax_normalize(arr: NDArray[np.floating]) -> NDArray[np.floating]:
        """Min-Max 정규화 (0-1 범위)."""
        if len(arr) == 0:
            return arr
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val

    @staticmethod
    def _cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        """코사인 유사도 계산 (0-1 범위)."""
        if len(a) == 0 or len(b) == 0:
            return 0.0
        
        # 길이 맞추기
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
            
        similarity = np.dot(a, b) / (norm_a * norm_b)
        # -1~1 → 0~1 변환
        return (similarity + 1) / 2

    @staticmethod
    def _get_grade(score: float) -> ScoreGrade:
        """점수를 등급으로 변환."""
        if score >= 90:
            return ScoreGrade.S
        elif score >= 80:
            return ScoreGrade.A
        elif score >= 70:
            return ScoreGrade.B
        elif score >= 60:
            return ScoreGrade.C
        elif score >= 50:
            return ScoreGrade.D
        else:
            return ScoreGrade.F

    # =========================================================================
    # 피드백 생성
    # =========================================================================

    def _generate_pitch_feedback(
        self,
        score: float,
        user_pitches: list[float],
        ref_pitches: list[float],
    ) -> str:
        """피치 점수에 대한 피드백 생성."""
        grade = self._get_grade(score)
        
        # 피치 범위 분석
        user_range = max(user_pitches) - min(user_pitches) if user_pitches else 0
        ref_range = max(ref_pitches) - min(ref_pitches) if ref_pitches else 0
        
        if grade in (ScoreGrade.S, ScoreGrade.A):
            return "억양 패턴이 레퍼런스와 매우 유사합니다. 훌륭해요!"
        elif grade == ScoreGrade.B:
            return "억양이 대체로 잘 맞지만, 일부 구간에서 차이가 있습니다."
        elif grade == ScoreGrade.C:
            if user_range < ref_range * 0.7:
                return "억양 변화가 다소 평탄합니다. 감정을 더 실어 말해보세요."
            elif user_range > ref_range * 1.3:
                return "억양 변화가 과합니다. 조금 더 자연스럽게 말해보세요."
            return "억양 패턴을 레퍼런스에 맞춰 연습해보세요."
        else:
            return "억양이 레퍼런스와 많이 다릅니다. 대사의 감정선을 다시 확인해보세요."

    def _generate_energy_feedback(
        self,
        score: float,
        user_energies: list[float],
        ref_energies: list[float],
    ) -> str:
        """에너지 점수에 대한 피드백 생성."""
        grade = self._get_grade(score)
        
        user_avg = np.mean(user_energies) if user_energies else 0
        ref_avg = np.mean(ref_energies) if ref_energies else 0
        
        if grade in (ScoreGrade.S, ScoreGrade.A):
            return "볼륨과 강세가 레퍼런스와 잘 맞습니다!"
        elif grade == ScoreGrade.B:
            return "볼륨 패턴이 대체로 좋지만, 강세 위치를 조금 더 맞춰보세요."
        elif grade == ScoreGrade.C:
            if user_avg < ref_avg * 0.7:
                return "전체적으로 소리가 작습니다. 더 크게 말해보세요."
            elif user_avg > ref_avg * 1.3:
                return "전체적으로 소리가 큽니다. 볼륨을 조절해보세요."
            return "강세와 볼륨 변화를 레퍼런스에 맞춰 연습해보세요."
        else:
            return "볼륨 패턴이 많이 다릅니다. 대사의 강약을 다시 확인해보세요."

    def _generate_expression_feedback(
        self,
        score: float,
        frame_scores: list[float],
        face_detection_rate: float,
    ) -> str:
        """표정 점수에 대한 피드백 생성."""
        grade = self._get_grade(score)
        
        if face_detection_rate < 0.8:
            return f"얼굴 인식률({face_detection_rate:.0%})이 낮습니다. 카메라를 정면으로 바라봐주세요."
        
        if grade in (ScoreGrade.S, ScoreGrade.A):
            return "표정 연기가 레퍼런스와 매우 유사합니다. 훌륭해요!"
        elif grade == ScoreGrade.B:
            return "표정이 대체로 잘 맞지만, 일부 표정을 더 과감하게 표현해보세요."
        elif grade == ScoreGrade.C:
            # 변화량 분석
            score_std = np.std(frame_scores) if frame_scores else 0
            if score_std < 0.1:
                return "표정 변화가 적습니다. 감정에 따라 더 다양한 표정을 지어보세요."
            return "표정을 레퍼런스 배우에 맞춰 연습해보세요."
        else:
            return "표정이 레퍼런스와 많이 다릅니다. 배우의 표정을 자세히 관찰해보세요."

    def _generate_overall_feedback(
        self,
        total_score: float,
        pitch: ScoreDetail,
        energy: ScoreDetail,
        expression: ScoreDetail,
    ) -> str:
        """종합 피드백 생성."""
        grade = self._get_grade(total_score)
        
        # 가장 낮은 점수 항목 찾기
        scores = {
            "억양": pitch.score,
            "볼륨": energy.score,
            "표정": expression.score,
        }
        weakest = min(scores, key=scores.get)
        strongest = max(scores, key=scores.get)

        if grade == ScoreGrade.S:
            return f"🎭 완벽한 연기입니다! 모든 항목에서 뛰어난 성과를 보여주셨어요."
        elif grade == ScoreGrade.A:
            return f"🎭 훌륭한 연기입니다! {strongest}이(가) 특히 좋았어요."
        elif grade == ScoreGrade.B:
            return f"🎭 좋은 연기입니다! {weakest}을(를) 조금 더 연습하면 더 좋아질 거예요."
        elif grade == ScoreGrade.C:
            return f"🎭 괜찮은 시도입니다! {weakest}에 집중해서 연습해보세요."
        elif grade == ScoreGrade.D:
            return f"🎭 조금 더 노력이 필요해요. 레퍼런스 영상을 다시 보면서 {weakest}을(를) 연습해보세요."
        else:
            return f"🎭 레퍼런스 영상을 천천히 분석하고, 하나씩 따라해보세요. 연습하면 반드시 늘어요!"

    def _empty_result(self, dtw_result: DTWResult) -> ScoringResult:
        """빈 결과 반환."""
        return ScoringResult(
            total_score=0.0,
            audio_pitch_score=ScoreDetail(
                score=0.0, weight=self.weight_pitch, feedback="데이터 부족"
            ),
            audio_energy_score=ScoreDetail(
                score=0.0, weight=self.weight_energy, feedback="데이터 부족"
            ),
            video_expression_score=ScoreDetail(
                score=0.0, weight=self.weight_expression, feedback="데이터 부족"
            ),
            dtw_result=dtw_result,
            overall_feedback="분석할 데이터가 부족합니다.",
        )

    def _calculate_frame_scores(
        self,
        user: AnalysisResult,
        ref: AnalysisResult,
        aligned_pairs: list[tuple[int, int]],
    ) -> list[FrameScore]:
        """프레임별 상세 점수 계산."""
        frame_scores = []
        user_fps = user.fps

        for user_idx, ref_idx in aligned_pairs:
            user_frame = user.frames[user_idx] if user_idx < len(user.frames) else None
            ref_frame = ref.frames[ref_idx] if ref_idx < len(ref.frames) else None

            fs = FrameScore(
                user_frame_idx=user_idx,
                actor_frame_idx=ref_idx,
                timestamp_ms=int(user_idx * 1000 / user_fps),
            )

            # 피치/에너지 점수
            if user_frame and user_frame.audio and ref_frame and ref_frame.audio:
                if user_frame.audio.is_voiced and ref_frame.audio.is_voiced:
                    # 피치 유사도 (단순 비율)
                    if ref_frame.audio.pitch > 0:
                        pitch_ratio = user_frame.audio.pitch / ref_frame.audio.pitch
                        fs.pitch_score = max(0, 100 - abs(1 - pitch_ratio) * 100)
                    
                    # 에너지 유사도
                    if ref_frame.audio.energy > 0:
                        energy_ratio = user_frame.audio.energy / ref_frame.audio.energy
                        fs.energy_score = max(0, 100 - abs(1 - energy_ratio) * 50)

            # 표정 점수
            if (user_frame and user_frame.video and user_frame.video.blendshapes and
                ref_frame and ref_frame.video and ref_frame.video.blendshapes):
                user_bs = np.array(user_frame.video.blendshapes.to_vector())
                ref_bs = np.array(ref_frame.video.blendshapes.to_vector())
                similarity = self._cosine_similarity(user_bs, ref_bs)
                fs.expression_score = similarity * 100

            # 종합 점수
            fs.combined_score = (
                fs.pitch_score * self.weight_pitch +
                fs.energy_score * self.weight_energy +
                fs.expression_score * self.weight_expression
            )

            frame_scores.append(fs)

        return frame_scores


# =============================================================================
# 전역 인스턴스 (Lazy Loading)
# =============================================================================

_scoring_service: Optional[ScoringService] = None


def get_scoring_service() -> ScoringService:
    """ScoringService 싱글톤 인스턴스 반환."""
    global _scoring_service
    if _scoring_service is None:
        _scoring_service = ScoringService()
    return _scoring_service


# =============================================================================
# CLI 테스트용
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python scoring_service.py <user_analysis.msgpack> <reference.msgpack>")
        print("\nNote: DTW will be computed automatically between the two analyses.")
        sys.exit(1)

    user_path = sys.argv[1]
    ref_path = sys.argv[2]

    # 데이터 로드
    print(f"📂 사용자 분석 로드: {user_path}")
    user_analysis = AnalysisResult.load(user_path)
    
    print(f"📂 레퍼런스 로드: {ref_path}")
    reference = AnalysisResult.load(ref_path)

    # DTW 동기화
    from app.services.dtw_service import get_dtw_service
    
    print("\n🔗 DTW 동기화 중...")
    dtw_service = get_dtw_service()
    
    # 사용자 분석에서 오디오 MFCC 추출
    user_mfcc = np.array(user_analysis.get_mfcc_matrix())
    dtw_result = dtw_service.synchronize(
        user_audio=user_mfcc,  # numpy array 직접 전달
        reference=reference,
        user_id="test_user",
    )

    # 스코어링
    print("\n📊 스코어링 중...")
    scoring_service = get_scoring_service()
    result = scoring_service.score(user_analysis, reference, dtw_result)

    # 결과 출력
    print("\n" + "=" * 60)
    print("🎭 연기 평가 결과")
    print("=" * 60)
    
    grade = ScoringService._get_grade(result.total_score)
    print(f"\n  📊 종합 점수: {result.total_score:.1f}/100 (등급: {grade.value})")
    
    print(f"\n  🎤 억양 (피치): {result.audio_pitch_score.score:.1f}/100")
    print(f"     → {result.audio_pitch_score.feedback}")
    
    print(f"\n  🔊 볼륨 (에너지): {result.audio_energy_score.score:.1f}/100")
    print(f"     → {result.audio_energy_score.feedback}")
    
    print(f"\n  😀 표정: {result.video_expression_score.score:.1f}/100")
    print(f"     → {result.video_expression_score.feedback}")
    
    print(f"\n  💬 종합 피드백:")
    print(f"     {result.overall_feedback}")
    print()
