"""
분석 결과 스키마 정의 (Stage 0)

Audio-only DTW (MFCC) 기반 동기화 및 Audio/Video 스코어링을 위한 데이터 구조.
JSON과 MessagePack 직렬화를 모두 지원합니다.

사용 예시:
    # 저장
    result = AnalysisResult(...)
    result.save("output.msgpack")  # 프로덕션 (권장)
    result.save("output.json")     # 디버깅용

    # 로드
    result = AnalysisResult.load("output.msgpack")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import msgpack
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Blendshape 정의 (ARKit 52개 중 연기 분석에 필요한 핵심 블렌드쉐입)
# =============================================================================

class Blendshapes(BaseModel):
    """
    얼굴 표정 블렌드쉐입 (0.0 ~ 1.0 정규화).
    ARKit 기반, 연기 분석에 핵심적인 항목만 포함.
    """
    # 입 관련
    jawOpen: float = Field(default=0.0, ge=0.0, le=1.0, description="턱 벌림")
    mouthSmileLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="왼쪽 입꼬리 올림")
    mouthSmileRight: float = Field(default=0.0, ge=0.0, le=1.0, description="오른쪽 입꼬리 올림")
    mouthFrownLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="왼쪽 입꼬리 내림")
    mouthFrownRight: float = Field(default=0.0, ge=0.0, le=1.0, description="오른쪽 입꼬리 내림")
    mouthPucker: float = Field(default=0.0, ge=0.0, le=1.0, description="입 오므림")
    mouthLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="입 왼쪽으로")
    mouthRight: float = Field(default=0.0, ge=0.0, le=1.0, description="입 오른쪽으로")

    # 눈썹 관련
    browInnerUp: float = Field(default=0.0, ge=0.0, le=1.0, description="눈썹 안쪽 올림")
    browDownLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="왼쪽 눈썹 내림")
    browDownRight: float = Field(default=0.0, ge=0.0, le=1.0, description="오른쪽 눈썹 내림")
    browOuterUpLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="왼쪽 눈썹 바깥 올림")
    browOuterUpRight: float = Field(default=0.0, ge=0.0, le=1.0, description="오른쪽 눈썹 바깥 올림")

    # 눈 관련
    eyeWideLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="왼쪽 눈 크게 뜸")
    eyeWideRight: float = Field(default=0.0, ge=0.0, le=1.0, description="오른쪽 눈 크게 뜸")
    eyeSquintLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="왼쪽 눈 찡그림")
    eyeSquintRight: float = Field(default=0.0, ge=0.0, le=1.0, description="오른쪽 눈 찡그림")
    eyeBlinkLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="왼쪽 눈 감기")
    eyeBlinkRight: float = Field(default=0.0, ge=0.0, le=1.0, description="오른쪽 눈 감기")

    # 볼/코 관련
    cheekPuff: float = Field(default=0.0, ge=0.0, le=1.0, description="볼 부풀리기")
    noseSneerLeft: float = Field(default=0.0, ge=0.0, le=1.0, description="왼쪽 코 찡그림")
    noseSneerRight: float = Field(default=0.0, ge=0.0, le=1.0, description="오른쪽 코 찡그림")

    # 머리 포즈 (3D 좌표 기반 입체 분석)
    headPitch: float = Field(default=0.0, ge=-1.0, le=1.0, description="머리 상하 기울기 (위: +, 아래: -)")
    headYaw: float = Field(default=0.0, ge=-1.0, le=1.0, description="머리 좌우 회전 (왼쪽: +, 오른쪽: -)")
    headRoll: float = Field(default=0.0, ge=-1.0, le=1.0, description="머리 좌우 기울임 (왼쪽: +, 오른쪽: -)")

    # 깊이 기반 표정 (z 좌표 활용)
    facePush: float = Field(default=0.0, ge=0.0, le=1.0, description="얼굴 전방 돌출 정도")
    chinForward: float = Field(default=0.0, ge=0.0, le=1.0, description="턱 전방 돌출 정도")

    model_config = {
        "extra": "ignore",  # 알 수 없는 필드 무시
        "validate_assignment": True,
    }

    def to_vector(self) -> list[float]:
        """블렌드쉐입을 벡터로 변환 (DTW/스코어링용)."""
        return [
            self.jawOpen,
            self.mouthSmileLeft, self.mouthSmileRight,
            self.mouthFrownLeft, self.mouthFrownRight,
            self.mouthPucker, self.mouthLeft, self.mouthRight,
            self.browInnerUp, self.browDownLeft, self.browDownRight,
            self.browOuterUpLeft, self.browOuterUpRight,
            self.eyeWideLeft, self.eyeWideRight,
            self.eyeSquintLeft, self.eyeSquintRight,
            self.eyeBlinkLeft, self.eyeBlinkRight,
            self.cheekPuff, self.noseSneerLeft, self.noseSneerRight,
            # 3D 좌표 기반 추가 특성
            self.headPitch, self.headYaw, self.headRoll,
            self.facePush, self.chinForward,
        ]

    @classmethod
    def from_vector(cls, vector: list[float]) -> "Blendshapes":
        """벡터에서 블렌드쉐입 복원."""
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
            # 3D 좌표 기반 추가 특성
            "headPitch", "headYaw", "headRoll",
            "facePush", "chinForward",
        ]
        # 이전 버전 호환성: 벡터 길이가 짧으면 기본값 사용
        padded_vector = list(vector) + [0.0] * (len(keys) - len(vector))
        return cls(**dict(zip(keys, padded_vector[:len(keys)])))


# =============================================================================
# Audio Features
# =============================================================================

class AudioFeatures(BaseModel):
    """
    오디오 특성 (프레임 단위).
    DTW 동기화 및 음성 스코어링에 사용됩니다.
    """
    mfcc: list[float] = Field(
        default_factory=list,
        description="MFCC 계수 (기본 13개, DTW 동기화용)",
    )
    pitch: float = Field(
        default=0.0,
        ge=0.0,
        description="피치/기본주파수 Hz (억양 스코어링용)",
    )
    energy: float = Field(
        default=0.0,
        ge=0.0,
        description="에너지/볼륨 RMS (볼륨 스코어링용)",
    )

    # 선택적 확장 필드
    pitch_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="피치 추정 신뢰도 (0-1)",
    )
    is_voiced: bool = Field(
        default=False,
        description="유성음 여부 (대사 구간 판별용)",
    )

    model_config = {
        "extra": "ignore",
    }


# =============================================================================
# Video Features
# =============================================================================

class VideoFeatures(BaseModel):
    """
    비디오 특성 (프레임 단위).
    블렌드쉐입 기반 표정 스코어링에 사용됩니다.
    """
    blendshapes: Optional[Blendshapes] = Field(
        default=None,
        description="얼굴 블렌드쉐입 (얼굴 미검출시 None)",
    )
    face_detected: bool = Field(
        default=False,
        description="얼굴 검출 여부",
    )

    # 선택적: 머리 포즈 (추후 확장용)
    head_rotation: Optional[list[float]] = Field(
        default=None,
        description="머리 회전 [pitch, yaw, roll] (라디안)",
    )

    model_config = {
        "extra": "ignore",
    }


# =============================================================================
# Frame Data (핵심 단위)
# =============================================================================

class FrameData(BaseModel):
    """
    단일 프레임의 모든 특성 데이터.
    timestamp_ms가 primary key 역할을 합니다.
    """
    timestamp_ms: int = Field(
        ...,
        ge=0,
        description="타임스탬프 (밀리초, primary key)",
    )
    video: Optional[VideoFeatures] = Field(
        default=None,
        description="비디오 특성 (비디오 없을 시 None)",
    )
    audio: Optional[AudioFeatures] = Field(
        default=None,
        description="오디오 특성 (오디오 없을 시 None)",
    )

    model_config = {
        "extra": "ignore",
    }

    @property
    def timestamp_sec(self) -> float:
        """초 단위 타임스탬프."""
        return self.timestamp_ms / 1000.0


# =============================================================================
# Analysis Result (루트 모델)
# =============================================================================

class AnalysisResult(BaseModel):
    """
    분석 결과 루트 모델.
    메타데이터와 프레임 데이터 리스트를 포함합니다.
    """
    # 메타데이터
    actor_id: str = Field(
        ...,
        min_length=1,
        description="배우/영상 고유 ID",
    )
    duration_sec: float = Field(
        ...,
        gt=0,
        description="총 영상 길이 (초)",
    )
    fps: float = Field(
        default=30.0,
        gt=0,
        description="비디오 프레임레이트",
    )
    sampling_rate: int = Field(
        default=22050,
        gt=0,
        description="오디오 샘플링레이트 (Hz)",
    )

    # 선택적 메타데이터
    source_file: Optional[str] = Field(
        default=None,
        description="원본 파일 경로/이름",
    )
    mfcc_n_coeffs: int = Field(
        default=13,
        ge=1,
        description="MFCC 계수 개수",
    )
    audio_hop_length: int = Field(
        default=512,
        ge=1,
        description="오디오 hop length (프레임 간격)",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="스키마 버전 (하위 호환성용)",
    )

    # 프레임 데이터
    frames: list[FrameData] = Field(
        default_factory=list,
        description="시간순 정렬된 프레임 데이터",
    )

    model_config = {
        "extra": "ignore",
    }

    @field_validator("frames")
    @classmethod
    def validate_frames_sorted(cls, v: list[FrameData]) -> list[FrameData]:
        """프레임이 timestamp_ms 기준 오름차순인지 검증."""
        for i in range(1, len(v)):
            if v[i].timestamp_ms < v[i - 1].timestamp_ms:
                raise ValueError("프레임이 시간순으로 정렬되어 있지 않습니다.")
        return v

    # =========================================================================
    # 직렬화/역직렬화 메서드
    # =========================================================================

    def to_dict(self) -> dict:
        """딕셔너리로 변환 (None 값 제외하여 용량 최적화)."""
        return self.model_dump(exclude_none=True)

    def to_json(self, indent: int | None = None) -> str:
        """JSON 문자열로 변환."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )

    def to_msgpack(self) -> bytes:
        """MessagePack 바이너리로 변환 (프로덕션 권장)."""
        return msgpack.packb(self.to_dict(), use_bin_type=True)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        """딕셔너리에서 생성."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "AnalysisResult":
        """JSON 문자열에서 생성."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_msgpack(cls, data: bytes) -> "AnalysisResult":
        """MessagePack 바이너리에서 생성."""
        return cls.from_dict(msgpack.unpackb(data, raw=False))

    # =========================================================================
    # 파일 I/O 메서드
    # =========================================================================

    def save(self, filepath: str | Path) -> None:
        """
        파일로 저장. 확장자에 따라 포맷 자동 선택.

        Args:
            filepath: 저장 경로 (.json 또는 .msgpack)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix.lower() == ".json":
            filepath.write_text(self.to_json(indent=2), encoding="utf-8")
        elif filepath.suffix.lower() in (".msgpack", ".mp", ".bin"):
            filepath.write_bytes(self.to_msgpack())
        else:
            raise ValueError(f"지원하지 않는 확장자: {filepath.suffix}")

    @classmethod
    def load(cls, filepath: str | Path) -> "AnalysisResult":
        """
        파일에서 로드. 확장자에 따라 포맷 자동 선택.

        Args:
            filepath: 파일 경로 (.json 또는 .msgpack)
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() == ".json":
            return cls.from_json(filepath.read_text(encoding="utf-8"))
        elif filepath.suffix.lower() in (".msgpack", ".mp", ".bin"):
            return cls.from_msgpack(filepath.read_bytes())
        else:
            raise ValueError(f"지원하지 않는 확장자: {filepath.suffix}")

    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================

    def get_frame_at(self, timestamp_ms: int) -> FrameData | None:
        """특정 타임스탬프의 프레임 반환 (이진 탐색)."""
        import bisect
        timestamps = [f.timestamp_ms for f in self.frames]
        idx = bisect.bisect_left(timestamps, timestamp_ms)
        if idx < len(self.frames) and self.frames[idx].timestamp_ms == timestamp_ms:
            return self.frames[idx]
        return None

    def get_mfcc_matrix(self) -> list[list[float]]:
        """모든 프레임의 MFCC를 2D 매트릭스로 반환 (DTW용)."""
        return [
            f.audio.mfcc if f.audio else [0.0] * self.mfcc_n_coeffs
            for f in self.frames
        ]

    def get_blendshape_matrix(self) -> list[list[float]]:
        """모든 프레임의 블렌드쉐입을 2D 매트릭스로 반환."""
        result = []
        for f in self.frames:
            if f.video and f.video.blendshapes:
                result.append(f.video.blendshapes.to_vector())
            else:
                result.append([0.0] * 27)  # 27개 블렌드쉐입 (3D 포즈 포함)
        return result

    @property
    def frame_count(self) -> int:
        """총 프레임 수."""
        return len(self.frames)

    @property
    def has_audio(self) -> bool:
        """오디오 데이터 포함 여부."""
        return any(f.audio is not None for f in self.frames)

    @property
    def has_video(self) -> bool:
        """비디오 데이터 포함 여부."""
        return any(f.video is not None for f in self.frames)


# =============================================================================
# DTW 결과 스키마 (동기화 결과 저장용)
# =============================================================================

class DTWResult(BaseModel):
    """DTW 동기화 결과."""
    actor_id: str = Field(..., description="레퍼런스 배우 ID")
    user_id: str = Field(..., description="사용자 ID")

    # 동기화 결과
    warping_path: list[tuple[int, int]] = Field(
        default_factory=list,
        description="워핑 경로 [(actor_idx, user_idx), ...]",
    )
    distance: float = Field(
        default=0.0,
        description="DTW 거리 (낮을수록 유사)",
    )
    normalized_distance: float = Field(
        default=0.0,
        description="정규화된 DTW 거리 (0-1)",
    )

    model_config = {
        "extra": "ignore",
    }


# =============================================================================
# 스코어링 결과 스키마
# =============================================================================

class SubMetric(BaseModel):
    """서브메트릭 상세 (Ultra-Precision Feedback 용)."""
    name: str = Field(description="서브메트릭 이름")
    score: float = Field(ge=0.0, le=100.0, description="점수 (0-100)")
    weight: float = Field(ge=0.0, le=1.0, description="가중치 (0-1)")
    feedback: str = Field(default="", description="서브메트릭별 피드백")
    details: dict = Field(default_factory=dict, description="추가 세부 정보")


class ScoreDetail(BaseModel):
    """개별 스코어 상세."""
    score: float = Field(ge=0.0, le=100.0, description="점수 (0-100)")
    weight: float = Field(default=1.0, ge=0.0, description="가중치")
    feedback: str = Field(default="", description="피드백 메시지")
    sub_metrics: list[SubMetric] = Field(default_factory=list, description="서브메트릭 리스트")


class ScoringResult(BaseModel):
    """최종 스코어링 결과."""
    total_score: float = Field(ge=0.0, le=100.0, description="종합 점수")

    # 세부 점수
    audio_pitch_score: Optional[ScoreDetail] = Field(default=None, description="음정 점수")
    audio_energy_score: Optional[ScoreDetail] = Field(default=None, description="볼륨 점수")
    video_expression_score: Optional[ScoreDetail] = Field(default=None, description="표정 점수")

    # 메타데이터
    dtw_result: Optional[DTWResult] = Field(default=None, description="DTW 동기화 결과")
    overall_feedback: str = Field(default="", description="종합 피드백")

    model_config = {
        "extra": "ignore",
    }
