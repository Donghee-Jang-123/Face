"""
Scoring Service (Stage 3) - Ultra-Precision Feedback

DTW 동기화 결과를 사용하여 사용자의 연기를 레퍼런스 배우와 비교 평가합니다.

평가 항목:
- Audio: 피치(억양) + 에너지(볼륨) 패턴 유사도 (서브메트릭 포함)
- Video: 블렌드쉐입(표정) 유사도 (얼굴 영역별 분석)

핵심 특징:
- DTW 워핑 경로를 사용한 정확한 프레임 대 프레임 비교
- 정규화된 비교로 개인 차이(음역대, 볼륨 등) 보정
- Ultra-Precision: 각 카테고리별 서브메트릭 기반 세밀한 피드백
- 가장 낮은 서브메트릭에 기반한 스마트 피드백 생성
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
    SubMetric,
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
    연기 스코어링 서비스 (Ultra-Precision Feedback).
    
    DTW 동기화 결과를 사용하여 사용자와 레퍼런스를 비교 평가합니다.
    각 카테고리를 서브메트릭으로 세분화하여 정밀한 피드백을 제공합니다.
    """

    # 가중치 설정 (합계 = 1.0)
    WEIGHT_PITCH = 0.30      # 억양 (대사 전달에 중요)
    WEIGHT_ENERGY = 0.20     # 볼륨/강세
    WEIGHT_EXPRESSION = 0.50  # 표정 (연기의 핵심)

    # =========================================================================
    # 서브메트릭 가중치 설정
    # =========================================================================
    
    # Pitch 서브메트릭 가중치
    PITCH_PATTERN_WEIGHT = 0.70    # 패턴 매칭 (멜로디가 맞는지)
    PITCH_RANGE_WEIGHT = 0.30      # 다이내믹 레인지 (단조롭지 않은지)
    
    # Energy 서브메트릭 가중치
    ENERGY_PATTERN_WEIGHT = 0.70   # 패턴 매칭 (강세 위치)
    ENERGY_INTENSITY_WEIGHT = 0.30 # 인텐시티 (다이내믹 레인지)
    
    # Expression 서브메트릭 가중치 (얼굴 영역별)
    EXPRESSION_EYES_WEIGHT = 0.40   # 눈 (감정의 진정성)
    EXPRESSION_MOUTH_WEIGHT = 0.20  # 입 (대사 전달)
    EXPRESSION_BROWS_WEIGHT = 0.40  # 눈썹 (감정 표현)

    # =========================================================================
    # 점수 엄격도 (오디오/비디오를 더 깐깐히 평가)
    # =========================================================================
    # 유사도/상관계수를 더 엄격하게 변환하는 지수 (1.0보다 클수록 엄격)
    SIMILARITY_POWER = 1.35
    CORRELATION_POWER = 1.40
    # 레인지 비율이 1에서 벗어날 때 페널티 크기 (클수록 엄격)
    RANGE_PENALTY_MULT = 140.0

    # =========================================================================
    # 블렌드쉐입 그룹 정의 (얼굴 영역별)
    # =========================================================================
    
    # 눈 관련 블렌드쉐입
    EYE_BLENDSHAPES = [
        "eyeWideLeft", "eyeWideRight",
        "eyeSquintLeft", "eyeSquintRight",
        "eyeBlinkLeft", "eyeBlinkRight",
    ]
    
    # 입 관련 블렌드쉐입
    MOUTH_BLENDSHAPES = [
        "jawOpen",
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthPucker", "mouthLeft", "mouthRight",
    ]
    
    # 눈썹 관련 블렌드쉐입
    BROW_BLENDSHAPES = [
        "browInnerUp",
        "browDownLeft", "browDownRight",
        "browOuterUpLeft", "browOuterUpRight",
    ]

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

        print(f"📊 ScoringService (Ultra-Precision): 초기화 완료")
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
        사용자 연기 평가 (Ultra-Precision Feedback).
        
        Args:
            user_analysis: 사용자 분석 결과 (Stage 1에서 생성)
            reference: 레퍼런스 배우 분석 결과
            dtw_result: DTW 동기화 결과 (Stage 2에서 생성)
            
        Returns:
            ScoringResult: 종합 평가 결과 (서브메트릭 포함)
        """
        # 워핑 경로에서 정렬된 프레임 쌍 추출
        aligned_pairs = dtw_result.warping_path

        if not aligned_pairs:
            return self._empty_result(dtw_result)

        # 1. 오디오 점수 계산 (서브메트릭 포함)
        pitch_detail = self._score_pitch_advanced(
            user_analysis, reference, aligned_pairs
        )
        energy_detail = self._score_energy_advanced(
            user_analysis, reference, aligned_pairs
        )

        # 2. 비디오(표정) 점수 계산 (얼굴 영역별)
        expression_detail = self._score_expression_advanced(
            user_analysis, reference, aligned_pairs
        )

        # 3. 종합 점수 계산
        total_score = (
            pitch_detail.score * self.weight_pitch +
            energy_detail.score * self.weight_energy +
            expression_detail.score * self.weight_expression
        )

        # 4. 종합 피드백 생성 (스마트 피드백)
        overall_feedback = self._generate_overall_feedback_smart(
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
    # 오디오 스코어링 (Ultra-Precision)
    # =========================================================================

    def _score_pitch_advanced(
        self,
        user: AnalysisResult,
        ref: AnalysisResult,
        aligned_pairs: list[tuple[int, int]],
    ) -> ScoreDetail:
        """
        피치(억양) 패턴 유사도 평가 - Ultra-Precision.
        
        서브메트릭:
        1. Pattern Match (70%): Z-정규화된 델타 피치의 코사인 유사도
        2. Dynamic Range (30%): 사용자 vs 배우의 표준편차 비교
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
                sub_metrics=[],
            )

        user_arr = np.array(user_pitches)
        ref_arr = np.array(ref_pitches)

        # =====================================================================
        # 서브메트릭 1: Pattern Match (70%)
        # =====================================================================
        # 피치를 상대적 변화율로 변환 (델타 피치)
        user_delta = self._compute_delta(user_arr)
        ref_delta = self._compute_delta(ref_arr)

        # Z-score 정규화 후 비교
        user_norm = self._zscore_normalize(user_delta)
        ref_norm = self._zscore_normalize(ref_delta)

        # 코사인 유사도 계산
        pattern_similarity = self._cosine_similarity(user_norm, ref_norm)
        pattern_score = self._similarity_to_score(pattern_similarity)

        # =====================================================================
        # 서브메트릭 2: Dynamic Range (30%)
        # =====================================================================
        user_std = np.std(user_arr)
        ref_std = np.std(ref_arr)
        
        # 배우 대비 사용자의 다이내믹 레인지 비율
        if ref_std > 1e-8:
            range_ratio = user_std / ref_std
            deviation = abs(1.0 - range_ratio)
            range_score = max(0.0, 100.0 - deviation * self.RANGE_PENALTY_MULT)
        else:
            range_ratio = 1.0
            range_score = 100.0  # 레퍼런스도 변화가 없으면 만점

        # =====================================================================
        # 종합 점수 및 서브메트릭 생성
        # =====================================================================
        final_score = (
            pattern_score * self.PITCH_PATTERN_WEIGHT +
            range_score * self.PITCH_RANGE_WEIGHT
        )

        # 서브메트릭 피드백 생성
        pattern_feedback = self._generate_pitch_pattern_feedback(pattern_score)
        range_feedback = self._generate_pitch_range_feedback(range_score, range_ratio if ref_std > 1e-8 else 1.0)

        sub_metrics = [
            SubMetric(
                name="pattern_match",
                score=round(pattern_score, 1),
                weight=self.PITCH_PATTERN_WEIGHT,
                feedback=pattern_feedback,
                details={
                    "description": "억양 패턴 (멜로디) 일치도",
                    "method": "Z-normalized Delta Pitch Cosine Similarity",
                }
            ),
            SubMetric(
                name="dynamic_range",
                score=round(range_score, 1),
                weight=self.PITCH_RANGE_WEIGHT,
                feedback=range_feedback,
                details={
                    "description": "음높이 변화 폭",
                    "user_std": round(float(user_std), 2),
                    "actor_std": round(float(ref_std), 2),
                    "ratio": round(float(range_ratio) if ref_std > 1e-8 else 1.0, 2),
                }
            ),
        ]

        # 스마트 피드백: 가장 낮은 서브메트릭 기반
        smart_feedback = self._generate_pitch_smart_feedback(pattern_score, range_score)

        return ScoreDetail(
            score=round(final_score, 1),
            weight=self.weight_pitch,
            feedback=smart_feedback,
            sub_metrics=sub_metrics,
        )

    def _score_energy_advanced(
        self,
        user: AnalysisResult,
        ref: AnalysisResult,
        aligned_pairs: list[tuple[int, int]],
    ) -> ScoreDetail:
        """
        에너지(볼륨) 패턴 유사도 평가 - Ultra-Precision.
        
        서브메트릭:
        1. Pattern Match (70%): 에너지 곡선의 상관계수/코사인 유사도
        2. Intensity (30%): 다이내믹 레인지 (Max - Min) 비교
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
                sub_metrics=[],
            )

        user_arr = np.array(user_energies)
        ref_arr = np.array(ref_energies)

        # =====================================================================
        # 서브메트릭 1: Pattern Match (70%)
        # =====================================================================
        # Min-Max 정규화 (볼륨 차이 보정)
        user_norm = self._minmax_normalize(user_arr)
        ref_norm = self._minmax_normalize(ref_arr)

        # 상관계수 기반 유사도
        correlation = np.corrcoef(user_norm, ref_norm)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # 0-100 점수로 변환 (상관계수 -1~1 → 0~100)
        pattern_score = self._correlation_to_score(correlation)

        # =====================================================================
        # 서브메트릭 2: Intensity (30%)
        # =====================================================================
        user_range = float(np.max(user_arr) - np.min(user_arr))
        ref_range = float(np.max(ref_arr) - np.min(ref_arr))
        
        # 배우 대비 사용자의 다이내믹 레인지 비율
        if ref_range > 1e-8:
            intensity_ratio = user_range / ref_range
            deviation = abs(1.0 - intensity_ratio)
            intensity_score = max(0.0, 100.0 - deviation * self.RANGE_PENALTY_MULT)
        else:
            intensity_ratio = 1.0
            intensity_score = 100.0

        # =====================================================================
        # 종합 점수 및 서브메트릭 생성
        # =====================================================================
        final_score = (
            pattern_score * self.ENERGY_PATTERN_WEIGHT +
            intensity_score * self.ENERGY_INTENSITY_WEIGHT
        )

        # 서브메트릭 피드백 생성
        pattern_feedback = self._generate_energy_pattern_feedback(pattern_score)
        intensity_feedback = self._generate_energy_intensity_feedback(
            intensity_score, intensity_ratio if ref_range > 1e-8 else 1.0
        )

        sub_metrics = [
            SubMetric(
                name="pattern_match",
                score=round(pattern_score, 1),
                weight=self.ENERGY_PATTERN_WEIGHT,
                feedback=pattern_feedback,
                details={
                    "description": "볼륨 패턴 (강세 위치) 일치도",
                    "method": "Normalized Energy Correlation",
                    "correlation": round(float(correlation), 3),
                }
            ),
            SubMetric(
                name="intensity",
                score=round(intensity_score, 1),
                weight=self.ENERGY_INTENSITY_WEIGHT,
                feedback=intensity_feedback,
                details={
                    "description": "볼륨 다이내믹 레인지 (속삭임~외침)",
                    "user_range": round(user_range, 4),
                    "actor_range": round(ref_range, 4),
                    "ratio": round(float(intensity_ratio) if ref_range > 1e-8 else 1.0, 2),
                }
            ),
        ]

        # 스마트 피드백: 가장 낮은 서브메트릭 기반
        smart_feedback = self._generate_energy_smart_feedback(pattern_score, intensity_score)

        return ScoreDetail(
            score=round(final_score, 1),
            weight=self.weight_energy,
            feedback=smart_feedback,
            sub_metrics=sub_metrics,
        )

    # =========================================================================
    # 비디오(표정) 스코어링 (Ultra-Precision)
    # =========================================================================

    def _score_expression_advanced(
        self,
        user: AnalysisResult,
        ref: AnalysisResult,
        aligned_pairs: list[tuple[int, int]],
    ) -> ScoreDetail:
        """
        표정(블렌드쉐입) 유사도 평가 - Ultra-Precision.
        
        얼굴을 3개 영역으로 나누어 개별 평가:
        1. Eyes (40%): eyeWide, eyeSquint, eyeBlink
        2. Mouth (20%): jawOpen, mouthSmile, mouthFrown, mouthPucker
        3. Brows (40%): browInnerUp, browDown, browOuterUp
        """
        # 프레임별 영역별 점수 수집
        eye_scores = []
        mouth_scores = []
        brow_scores = []
        face_detection_count = 0
        valid_frames = 0

        for user_idx, ref_idx in aligned_pairs:
            user_frame = user.frames[user_idx] if user_idx < len(user.frames) else None
            ref_frame = ref.frames[ref_idx] if ref_idx < len(ref.frames) else None

            if not (user_frame and user_frame.video and 
                    ref_frame and ref_frame.video):
                continue

            # 얼굴 검출 여부 체크
            if user_frame.video.face_detected:
                face_detection_count += 1

            if not (user_frame.video.blendshapes and 
                    ref_frame.video.blendshapes):
                continue

            valid_frames += 1

            # 블렌드쉐입 딕셔너리로 변환
            user_bs = user_frame.video.blendshapes.model_dump()
            ref_bs = ref_frame.video.blendshapes.model_dump()

            # 영역별 유사도 계산
            eye_sim = self._calculate_zone_similarity(user_bs, ref_bs, self.EYE_BLENDSHAPES)
            mouth_sim = self._calculate_zone_similarity(user_bs, ref_bs, self.MOUTH_BLENDSHAPES)
            brow_sim = self._calculate_zone_similarity(user_bs, ref_bs, self.BROW_BLENDSHAPES)

            eye_scores.append(eye_sim)
            mouth_scores.append(mouth_sim)
            brow_scores.append(brow_sim)

        if valid_frames < 5:
            detection_pct = (face_detection_count / len(aligned_pairs) * 100 
                           if aligned_pairs else 0)
            return ScoreDetail(
                score=50.0,
                weight=self.weight_expression,
                feedback=f"얼굴 인식률이 낮습니다 ({detection_pct:.0f}%). "
                        f"카메라를 정면으로 바라봐 주세요.",
                sub_metrics=[],
            )

        # =====================================================================
        # 영역별 평균 점수 계산
        # =====================================================================
        eye_score = np.mean(eye_scores) * 100 if eye_scores else 50.0
        mouth_score = np.mean(mouth_scores) * 100 if mouth_scores else 50.0
        brow_score = np.mean(brow_scores) * 100 if brow_scores else 50.0

        # 종합 점수 (가중 평균)
        final_score = (
            eye_score * self.EXPRESSION_EYES_WEIGHT +
            mouth_score * self.EXPRESSION_MOUTH_WEIGHT +
            brow_score * self.EXPRESSION_BROWS_WEIGHT
        )

        # 얼굴 인식률
        face_detection_rate = face_detection_count / len(aligned_pairs) if aligned_pairs else 0

        # =====================================================================
        # 서브메트릭 생성
        # =====================================================================
        eye_feedback = self._generate_eye_feedback(eye_score)
        mouth_feedback = self._generate_mouth_feedback(mouth_score)
        brow_feedback = self._generate_brow_feedback(brow_score)

        sub_metrics = [
            SubMetric(
                name="eyes",
                score=round(eye_score, 1),
                weight=self.EXPRESSION_EYES_WEIGHT,
                feedback=eye_feedback,
                details={
                    "description": "눈 표현 (감정의 진정성)",
                    "blendshapes": self.EYE_BLENDSHAPES,
                    "frame_count": len(eye_scores),
                }
            ),
            SubMetric(
                name="mouth",
                score=round(mouth_score, 1),
                weight=self.EXPRESSION_MOUTH_WEIGHT,
                feedback=mouth_feedback,
                details={
                    "description": "입 표현 (대사 전달)",
                    "blendshapes": self.MOUTH_BLENDSHAPES,
                    "frame_count": len(mouth_scores),
                }
            ),
            SubMetric(
                name="brows",
                score=round(brow_score, 1),
                weight=self.EXPRESSION_BROWS_WEIGHT,
                feedback=brow_feedback,
                details={
                    "description": "눈썹 표현 (감정 강조)",
                    "blendshapes": self.BROW_BLENDSHAPES,
                    "frame_count": len(brow_scores),
                }
            ),
        ]

        # 스마트 피드백: 가장 낮은 서브메트릭 기반
        smart_feedback = self._generate_expression_smart_feedback(
            eye_score, mouth_score, brow_score, face_detection_rate
        )

        return ScoreDetail(
            score=round(final_score, 1),
            weight=self.weight_expression,
            feedback=smart_feedback,
            sub_metrics=sub_metrics,
        )

    def _calculate_zone_similarity(
        self,
        user_bs: dict,
        ref_bs: dict,
        zone_keys: list[str],
    ) -> float:
        """특정 얼굴 영역의 블렌드쉐입 유사도 계산."""
        user_values = []
        ref_values = []
        
        for key in zone_keys:
            user_val = user_bs.get(key, 0.0)
            ref_val = ref_bs.get(key, 0.0)
            if user_val is not None and ref_val is not None:
                # 가중치 적용
                weight = self.BLENDSHAPE_WEIGHTS.get(key, 1.0)
                user_values.append(float(user_val) * weight)
                ref_values.append(float(ref_val) * weight)
        
        if len(user_values) < 2:
            return 0.5  # 데이터 부족 시 기본값
        
        user_arr = np.array(user_values)
        ref_arr = np.array(ref_values)
        
        similarity = self._cosine_similarity(user_arr, ref_arr)
        return self._similarity_to_unit(similarity)

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

    @classmethod
    def _similarity_to_unit(cls, similarity: float) -> float:
        """유사도를 더 엄격하게 0~1로 변환."""
        similarity = max(0.0, min(1.0, similarity))
        return similarity ** cls.SIMILARITY_POWER

    @classmethod
    def _similarity_to_score(cls, similarity: float) -> float:
        """유사도(0~1)를 엄격 점수(0~100)로 변환."""
        return cls._similarity_to_unit(similarity) * 100.0

    @classmethod
    def _correlation_to_score(cls, correlation: float) -> float:
        """상관계수를 엄격 점수(0~100)로 변환."""
        correlation = max(0.0, min(1.0, correlation))
        return (correlation ** cls.CORRELATION_POWER) * 100.0

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
    # 스마트 피드백 생성 (서브메트릭 기반)
    # =========================================================================

    def _generate_pitch_pattern_feedback(self, score: float) -> str:
        """피치 패턴 서브메트릭 피드백."""
        if score >= 80:
            return "억양의 멜로디가 정확합니다."
        elif score >= 60:
            return "억양 패턴이 대체로 맞지만 일부 구간에서 차이가 있습니다."
        else:
            return "억양의 오르내림이 레퍼런스와 다릅니다."

    def _generate_pitch_range_feedback(self, score: float, ratio: float) -> str:
        """피치 다이내믹 레인지 서브메트릭 피드백."""
        if score >= 80:
            return "음높이 변화 폭이 적절합니다."
        elif ratio < 0.5:
            return "음높이 변화가 너무 적습니다 (단조로움)."
        elif ratio < 0.8:
            return "음높이 변화가 다소 부족합니다."
        else:
            return "음높이 변화 폭을 더 키워보세요."

    def _generate_pitch_smart_feedback(self, pattern_score: float, range_score: float) -> str:
        """피치 스마트 피드백 - 가장 낮은 서브메트릭 기반."""
        if pattern_score >= 80 and range_score >= 80:
            return "억양 패턴과 변화 폭 모두 훌륭합니다!"
        
        if range_score < pattern_score:
            # 패턴은 맞지만 레인지가 부족
            if pattern_score >= 70:
                return "억양은 정확하지만, 톤이 너무 평탄합니다. 감정을 더 극적으로 표현해보세요."
            else:
                return "억양 패턴과 변화 폭 모두 연습이 필요합니다."
        else:
            # 레인지는 있지만 패턴이 다름
            if range_score >= 70:
                return "감정 표현은 풍부하지만, 억양의 오르내림 위치가 다릅니다. 멜로디를 맞춰보세요."
            else:
                return "억양 패턴을 레퍼런스에 맞춰 연습해보세요."

    def _generate_energy_pattern_feedback(self, score: float) -> str:
        """에너지 패턴 서브메트릭 피드백."""
        if score >= 80:
            return "강세 위치가 정확합니다."
        elif score >= 60:
            return "강세 패턴이 대체로 맞지만 일부 단어에서 차이가 있습니다."
        else:
            return "강세를 주는 위치가 레퍼런스와 다릅니다."

    def _generate_energy_intensity_feedback(self, score: float, ratio: float) -> str:
        """에너지 인텐시티 서브메트릭 피드백."""
        if score >= 80:
            return "볼륨 강약이 적절합니다."
        elif ratio < 0.5:
            return "볼륨 변화가 너무 작습니다 (단조로움)."
        elif ratio < 0.8:
            return "볼륨 변화가 다소 부족합니다."
        else:
            return "속삭임과 외침의 대비를 더 키워보세요."

    def _generate_energy_smart_feedback(self, pattern_score: float, intensity_score: float) -> str:
        """에너지 스마트 피드백 - 가장 낮은 서브메트릭 기반."""
        if pattern_score >= 80 and intensity_score >= 80:
            return "볼륨 패턴과 강약 조절 모두 훌륭합니다!"
        
        if intensity_score < pattern_score:
            # 패턴은 맞지만 강약이 부족
            if pattern_score >= 70:
                return "강세 위치는 맞지만, 속삭임과 외침의 대비가 부족합니다. 더 역동적으로 표현해보세요."
            else:
                return "볼륨 패턴과 강약 조절 모두 연습이 필요합니다."
        else:
            # 강약은 있지만 패턴이 다름
            if intensity_score >= 70:
                return "볼륨 변화는 풍부하지만, 강세를 주는 단어가 다릅니다. 강조 위치를 맞춰보세요."
            else:
                return "강세 패턴을 레퍼런스에 맞춰 연습해보세요."

    def _generate_eye_feedback(self, score: float) -> str:
        """눈 표현 서브메트릭 피드백."""
        if score >= 80:
            return "눈 표현이 감정을 잘 전달합니다."
        elif score >= 60:
            return "눈 표현이 대체로 좋지만 더 강조할 수 있습니다."
        else:
            return "눈에 감정이 부족합니다. 눈으로 더 표현해보세요."

    def _generate_mouth_feedback(self, score: float) -> str:
        """입 표현 서브메트릭 피드백."""
        if score >= 80:
            return "입 모양과 움직임이 정확합니다."
        elif score >= 60:
            return "입 표현이 대체로 좋지만 발음을 더 명확히 해보세요."
        else:
            return "입 움직임이 레퍼런스와 다릅니다."

    def _generate_brow_feedback(self, score: float) -> str:
        """눈썹 표현 서브메트릭 피드백."""
        if score >= 80:
            return "눈썹 표현이 감정을 잘 강조합니다."
        elif score >= 60:
            return "눈썹 표현이 대체로 좋지만 더 과감해도 좋습니다."
        else:
            return "눈썹 표현이 부족합니다. 감정에 따라 눈썹을 더 활용해보세요."

    def _generate_expression_smart_feedback(
        self,
        eye_score: float,
        mouth_score: float,
        brow_score: float,
        face_detection_rate: float,
    ) -> str:
        """표정 스마트 피드백 - 가장 낮은 서브메트릭 기반."""
        if face_detection_rate < 0.8:
            return f"얼굴 인식률({face_detection_rate:.0%})이 낮습니다. 카메라를 정면으로 바라봐주세요."
        
        # 모든 영역이 우수한 경우
        if eye_score >= 80 and mouth_score >= 80 and brow_score >= 80:
            return "모든 얼굴 영역에서 훌륭한 표현력을 보여주셨습니다!"
        
        # 가장 낮은 영역 찾기
        scores = {"눈": eye_score, "입": mouth_score, "눈썹": brow_score}
        weakest = min(scores, key=scores.get)
        weakest_score = scores[weakest]
        
        # 높은 영역 찾기
        strongest = max(scores, key=scores.get)
        strongest_score = scores[strongest]
        
        # 특정 조합에 대한 스마트 피드백
        if weakest == "눈" and weakest_score < 70:
            if mouth_score >= 70:
                return "대사 전달은 좋지만, 눈에 감정이 없어 보입니다. 눈 연기에 집중해보세요."
            else:
                return "표정 전체적으로 감정 표현이 부족합니다. 눈과 입 모두 더 과감하게 표현해보세요."
        
        if weakest == "눈썹" and weakest_score < 70:
            if eye_score >= 70:
                return "눈 표현은 좋지만, 눈썹 움직임이 부족합니다. 감정에 따라 눈썹을 더 활용해보세요."
            else:
                return "눈과 눈썹 표현을 더 과감하게 해보세요. 감정의 진정성이 느껴져야 합니다."
        
        if weakest == "입" and weakest_score < 70:
            return "발음과 입 모양을 레퍼런스에 맞춰 연습해보세요."
        
        # 일반적인 피드백
        return f"{weakest} 표현을 더 연습하면 전체 연기가 향상될 거예요."

    def _generate_overall_feedback_smart(
        self,
        total_score: float,
        pitch: ScoreDetail,
        energy: ScoreDetail,
        expression: ScoreDetail,
    ) -> str:
        """종합 스마트 피드백 생성."""
        grade = self._get_grade(total_score)
        
        # 가장 낮은 점수 항목 찾기
        scores = {
            "억양": pitch.score,
            "볼륨": energy.score,
            "표정": expression.score,
        }
        weakest = min(scores, key=scores.get)
        weakest_score = scores[weakest]
        strongest = max(scores, key=scores.get)
        strongest_score = scores[strongest]

        # 서브메트릭 레벨에서 가장 약한 부분 찾기
        all_sub_metrics = []
        for detail, category in [(pitch, "억양"), (energy, "볼륨"), (expression, "표정")]:
            for sm in detail.sub_metrics:
                all_sub_metrics.append({
                    "category": category,
                    "name": sm.name,
                    "score": sm.score,
                    "feedback": sm.feedback,
                })
        
        weakest_sub = min(all_sub_metrics, key=lambda x: x["score"]) if all_sub_metrics else None

        if grade == ScoreGrade.S:
            return f"🎭 완벽한 연기입니다! 모든 항목에서 뛰어난 성과를 보여주셨어요."
        elif grade == ScoreGrade.A:
            return f"🎭 훌륭한 연기입니다! {strongest}이(가) 특히 인상적이에요."
        elif grade == ScoreGrade.B:
            if weakest_sub:
                return f"🎭 좋은 연기입니다! {weakest_sub['category']}의 {weakest_sub['name']}을(를) 보완하면 더 좋아질 거예요."
            return f"🎭 좋은 연기입니다! {weakest}을(를) 조금 더 연습하면 더 좋아질 거예요."
        elif grade == ScoreGrade.C:
            if weakest_sub:
                return f"🎭 괜찮은 시도입니다! Tip: {weakest_sub['feedback']}"
            return f"🎭 괜찮은 시도입니다! {weakest}에 집중해서 연습해보세요."
        elif grade == ScoreGrade.D:
            return f"🎭 조금 더 노력이 필요해요. 레퍼런스 영상을 다시 보면서 {weakest}을(를) 연습해보세요."
        else:
            return f"🎭 레퍼런스 영상을 천천히 분석하고, 하나씩 따라해보세요. 연습하면 반드시 늘어요!"

    # =========================================================================
    # 빈 결과 및 프레임 점수 계산
    # =========================================================================

    def _empty_result(self, dtw_result: DTWResult) -> ScoringResult:
        """빈 결과 반환."""
        return ScoringResult(
            total_score=0.0,
            audio_pitch_score=ScoreDetail(
                score=0.0, weight=self.weight_pitch, feedback="데이터 부족", sub_metrics=[]
            ),
            audio_energy_score=ScoreDetail(
                score=0.0, weight=self.weight_energy, feedback="데이터 부족", sub_metrics=[]
            ),
            video_expression_score=ScoreDetail(
                score=0.0, weight=self.weight_expression, feedback="데이터 부족", sub_metrics=[]
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
    print("\n📊 스코어링 중 (Ultra-Precision)...")
    scoring_service = get_scoring_service()
    result = scoring_service.score(user_analysis, reference, dtw_result)

    # 결과 출력
    print("\n" + "=" * 70)
    print("🎭 연기 평가 결과 (Ultra-Precision Feedback)")
    print("=" * 70)
    
    grade = ScoringService._get_grade(result.total_score)
    print(f"\n  📊 종합 점수: {result.total_score:.1f}/100 (등급: {grade.value})")
    
    # 피치 상세
    print(f"\n  🎤 억양 (피치): {result.audio_pitch_score.score:.1f}/100")
    print(f"     → {result.audio_pitch_score.feedback}")
    for sm in result.audio_pitch_score.sub_metrics:
        print(f"       • {sm.name}: {sm.score:.1f}/100 ({sm.weight:.0%})")
        print(f"         {sm.feedback}")
    
    # 에너지 상세
    print(f"\n  🔊 볼륨 (에너지): {result.audio_energy_score.score:.1f}/100")
    print(f"     → {result.audio_energy_score.feedback}")
    for sm in result.audio_energy_score.sub_metrics:
        print(f"       • {sm.name}: {sm.score:.1f}/100 ({sm.weight:.0%})")
        print(f"         {sm.feedback}")
    
    # 표정 상세
    print(f"\n  😀 표정: {result.video_expression_score.score:.1f}/100")
    print(f"     → {result.video_expression_score.feedback}")
    for sm in result.video_expression_score.sub_metrics:
        print(f"       • {sm.name}: {sm.score:.1f}/100 ({sm.weight:.0%})")
        print(f"         {sm.feedback}")
    
    print(f"\n  💬 종합 피드백:")
    print(f"     {result.overall_feedback}")
    print()
