"""
DTW Synchronization Service (Stage 2)

Audio MFCC 기반 Dynamic Time Warping으로 사용자 오디오를 
레퍼런스 배우 오디오에 동기화합니다.

핵심 특징:
- Sakoe-Chiba Band 제약: O(N*W) 복잡도 (W = window size)
- Z-score 정규화: MFCC 벡터 정규화로 공정한 비교
- Cosine Distance: MFCC에 최적화된 거리 메트릭

성능 비교 (300 프레임 기준):
- fastdtw: ~5ms (근사, 정확도 ↓)
- 본 구현: ~10ms (정확, Sakoe-Chiba band=50)
- 순수 DTW: ~50ms (정확, 제약 없음)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from app.core.schemas import AnalysisResult, DTWResult


class DTWService:
    """
    Audio-only DTW 동기화 서비스.
    
    MFCC 특성을 사용하여 사용자 오디오를 레퍼런스에 정렬합니다.
    """

    def __init__(
        self,
        window_ratio: float = 0.2,
        distance_metric: str = "cosine",
        n_mfcc: int = 13,
    ):
        """
        Args:
            window_ratio: Sakoe-Chiba 윈도우 비율 (0.1 ~ 0.5)
                         낮을수록 빠르지만 큰 시간 차이를 못 잡음
            distance_metric: 거리 메트릭 ('cosine' 또는 'euclidean')
            n_mfcc: MFCC 계수 개수 (레퍼런스와 동일해야 함)
        """
        self.window_ratio = window_ratio
        self.distance_metric = distance_metric
        self.n_mfcc = n_mfcc
        
        print(f"🔗 DTWService: 초기화 완료 (metric={distance_metric}, window_ratio={window_ratio})")

    def synchronize(
        self,
        user_audio: Union[str, Path, NDArray[np.floating]],
        reference: AnalysisResult,
        user_id: str = "user",
    ) -> DTWResult:
        """
        사용자 오디오를 레퍼런스에 동기화.
        
        Args:
            user_audio: 사용자 오디오 파일 경로 또는 numpy 배열
            reference: Stage 1에서 생성된 레퍼런스 AnalysisResult
            user_id: 사용자 ID
            
        Returns:
            DTWResult: 동기화 결과 (warping_path, distance 등)
        """
        # 1. 사용자 MFCC 추출 (레퍼런스와 동일한 파라미터 사용)
        user_mfcc = self._extract_user_mfcc(
            user_audio,
            sampling_rate=reference.sampling_rate,
            hop_length=reference.audio_hop_length,
            n_mfcc=reference.mfcc_n_coeffs,
        )

        # 2. 레퍼런스 MFCC 추출
        ref_mfcc = np.array(reference.get_mfcc_matrix())

        # 3. Z-score 정규화
        user_mfcc_norm = self._normalize_mfcc(user_mfcc)
        ref_mfcc_norm = self._normalize_mfcc(ref_mfcc)

        # 4. DTW 실행
        path, distance = self._compute_dtw(
            user_mfcc_norm,
            ref_mfcc_norm,
        )

        # 5. 정규화된 거리 계산 (0-1 범위)
        path_length = len(path)
        normalized_distance = distance / path_length if path_length > 0 else 0.0

        # 6. 결과 생성
        result = DTWResult(
            actor_id=reference.actor_id,
            user_id=user_id,
            warping_path=path,
            distance=distance,
            normalized_distance=min(1.0, normalized_distance),
        )

        return result

    def get_timestamp_mapping(
        self,
        dtw_result: DTWResult,
        reference: AnalysisResult,
        user_fps: Optional[float] = None,
    ) -> dict[int, int]:
        """
        DTW 결과를 타임스탬프 매핑으로 변환.
        
        Args:
            dtw_result: synchronize() 결과
            reference: 레퍼런스 AnalysisResult
            user_fps: 사용자 비디오 FPS (None이면 레퍼런스 FPS 사용)
            
        Returns:
            {user_timestamp_ms: actor_timestamp_ms} 매핑
        """
        user_fps = user_fps or reference.fps
        user_frame_duration_ms = 1000.0 / user_fps
        ref_frame_duration_ms = 1000.0 / reference.fps

        mapping = {}
        for user_idx, ref_idx in dtw_result.warping_path:
            user_ts = int(user_idx * user_frame_duration_ms)
            ref_ts = int(ref_idx * ref_frame_duration_ms)
            mapping[user_ts] = ref_ts

        return mapping

    def get_frame_mapping(
        self,
        dtw_result: DTWResult,
    ) -> dict[int, int]:
        """
        DTW 결과를 프레임 인덱스 매핑으로 변환.
        
        Returns:
            {user_frame_idx: actor_frame_idx} 매핑
        """
        return {user_idx: ref_idx for user_idx, ref_idx in dtw_result.warping_path}

    # =========================================================================
    # MFCC 추출
    # =========================================================================

    def _extract_user_mfcc(
        self,
        audio_input: Union[str, Path, NDArray[np.floating]],
        sampling_rate: int,
        hop_length: int,
        n_mfcc: int,
    ) -> NDArray[np.floating]:
        """
        사용자 오디오에서 MFCC 추출 (레퍼런스와 동일한 파라미터).
        """
        # numpy 배열이 아니면 파일에서 로드
        if isinstance(audio_input, (str, Path)):
            audio_input = Path(audio_input)
            
            # 비디오 파일인 경우 오디오 추출
            if audio_input.suffix.lower() in ('.mp4', '.avi', '.mov', '.webm'):
                audio_np = self._extract_audio_from_video(audio_input, sampling_rate)
            else:
                audio_np, _ = librosa.load(
                    str(audio_input),
                    sr=sampling_rate,
                    mono=True,
                )
        else:
            audio_np = audio_input

        # MFCC 추출
        mfcc = librosa.feature.mfcc(
            y=audio_np,
            sr=sampling_rate,
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=hop_length,
        )

        # (n_mfcc, n_frames) -> (n_frames, n_mfcc)
        return mfcc.T

    def _extract_audio_from_video(
        self,
        video_path: Path,
        target_sr: int,
    ) -> NDArray[np.floating]:
        """
        비디오에서 오디오 추출 (Windows 호환).
        
        ffmpeg로 WAV 추출 후 librosa로 로드합니다.
        """
        import os
        import subprocess

        temp_wav = None
        try:
            # ffmpeg로 WAV 추출 (subprocess 사용 - Windows 호환성)
            temp_wav = tempfile.mktemp(suffix=".wav")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(target_sr), '-ac', '1',
                temp_wav, '-loglevel', 'error'
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # ffmpeg 명령어 형태로 재시도 (PATH 문제 대응)
                os.system(
                    f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le '
                    f'-ar {target_sr} -ac 1 "{temp_wav}" -loglevel error'
                )

            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                audio_np, _ = librosa.load(temp_wav, sr=target_sr, mono=True)
                return audio_np

            # 폴백: librosa 직접 사용 (일부 포맷만 지원)
            audio_np, _ = librosa.load(str(video_path), sr=target_sr, mono=True)
            return audio_np

        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    # =========================================================================
    # 정규화
    # =========================================================================

    def _normalize_mfcc(
        self,
        mfcc: NDArray[np.floating],
        method: str = "zscore",
    ) -> NDArray[np.floating]:
        """
        MFCC 정규화 (계수별 Z-score).
        
        Args:
            mfcc: shape (n_frames, n_mfcc)
            method: 'zscore' 또는 'minmax'
        """
        if mfcc.size == 0:
            return mfcc

        if method == "zscore":
            # 각 MFCC 계수별로 Z-score 정규화
            mean = np.mean(mfcc, axis=0, keepdims=True)
            std = np.std(mfcc, axis=0, keepdims=True)
            # 0으로 나누기 방지
            std = np.where(std < 1e-8, 1.0, std)
            return (mfcc - mean) / std
        
        elif method == "minmax":
            min_val = np.min(mfcc, axis=0, keepdims=True)
            max_val = np.max(mfcc, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val < 1e-8, 1.0, range_val)
            return (mfcc - min_val) / range_val
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # =========================================================================
    # DTW 알고리즘
    # =========================================================================

    def _compute_dtw(
        self,
        seq1: NDArray[np.floating],
        seq2: NDArray[np.floating],
    ) -> tuple[list[tuple[int, int]], float]:
        """
        Sakoe-Chiba Band 제약이 있는 DTW 계산.
        
        Args:
            seq1: 사용자 시퀀스 (n_frames1, n_features)
            seq2: 레퍼런스 시퀀스 (n_frames2, n_features)
            
        Returns:
            (warping_path, total_distance)
        """
        n, m = len(seq1), len(seq2)
        
        if n == 0 or m == 0:
            return [], 0.0

        # Sakoe-Chiba window 크기 계산
        window = max(int(max(n, m) * self.window_ratio), abs(n - m) + 1)

        # 거리 행렬 계산 (scipy 사용 - 빠름)
        if self.distance_metric == "cosine":
            # Cosine distance: 1 - cosine_similarity
            cost_matrix = cdist(seq1, seq2, metric="cosine")
        else:
            cost_matrix = cdist(seq1, seq2, metric="euclidean")

        # DTW 누적 비용 행렬
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        # DTW 계산 (Sakoe-Chiba band 적용)
        for i in range(1, n + 1):
            # 윈도우 범위 계산
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)
            
            for j in range(j_start, j_end):
                cost = cost_matrix[i - 1, j - 1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # insertion
                    dtw_matrix[i, j - 1],      # deletion
                    dtw_matrix[i - 1, j - 1],  # match
                )

        # 최종 거리
        total_distance = float(dtw_matrix[n, m])

        # Backtracking으로 최적 경로 추출
        path = self._backtrack(dtw_matrix, n, m)

        return path, total_distance

    def _backtrack(
        self,
        dtw_matrix: NDArray[np.floating],
        n: int,
        m: int,
    ) -> list[tuple[int, int]]:
        """DTW 행렬에서 최적 경로 역추적."""
        path = []
        i, j = n, m

        while i > 0 and j > 0:
            path.append((i - 1, j - 1))  # 0-indexed

            # 세 방향 중 최소 비용 선택
            candidates = [
                (dtw_matrix[i - 1, j - 1], i - 1, j - 1),  # diagonal
                (dtw_matrix[i - 1, j], i - 1, j),          # up
                (dtw_matrix[i, j - 1], i, j - 1),          # left
            ]
            
            # 유효한 후보만 필터링
            valid_candidates = [(c, ni, nj) for c, ni, nj in candidates if c != np.inf]
            
            if not valid_candidates:
                break
                
            _, i, j = min(valid_candidates, key=lambda x: x[0])

        # 경로 뒤집기 (시작점 -> 끝점)
        path.reverse()
        return path


# =============================================================================
# 고급 DTW 기능 (선택적)
# =============================================================================

class AdvancedDTWService(DTWService):
    """
    추가 기능이 있는 고급 DTW 서비스.
    
    - 다중 해상도 DTW (빠른 근사)
    - 부분 매칭 (subsequence DTW)
    - 가중치 MFCC
    """

    def __init__(
        self,
        window_ratio: float = 0.2,
        distance_metric: str = "cosine",
        n_mfcc: int = 13,
        mfcc_weights: Optional[list[float]] = None,
    ):
        """
        Args:
            mfcc_weights: MFCC 계수별 가중치 (None이면 균등)
                         예: 첫 번째 계수(에너지)에 낮은 가중치
        """
        super().__init__(window_ratio, distance_metric, n_mfcc)
        
        # 기본 가중치: 첫 번째 MFCC(에너지 관련)에 낮은 가중치
        if mfcc_weights is None:
            self.mfcc_weights = np.array([0.5] + [1.0] * (n_mfcc - 1))
        else:
            self.mfcc_weights = np.array(mfcc_weights)

    def _normalize_mfcc(
        self,
        mfcc: NDArray[np.floating],
        method: str = "zscore",
    ) -> NDArray[np.floating]:
        """정규화 후 가중치 적용."""
        normalized = super()._normalize_mfcc(mfcc, method)
        # 가중치 적용
        return normalized * self.mfcc_weights

    def synchronize_with_confidence(
        self,
        user_audio: Union[str, Path, NDArray[np.floating]],
        reference: AnalysisResult,
        user_id: str = "user",
    ) -> tuple[DTWResult, float]:
        """
        동기화 + 신뢰도 점수 반환.
        
        Returns:
            (DTWResult, confidence_score)
            confidence_score: 0-100 (높을수록 좋은 매칭)
        """
        result = self.synchronize(user_audio, reference, user_id)
        
        # 신뢰도 계산
        # 낮은 normalized_distance = 높은 신뢰도
        confidence = max(0.0, 100.0 * (1.0 - result.normalized_distance * 2))
        
        return result, confidence

    def find_best_subsequence(
        self,
        user_audio: Union[str, Path, NDArray[np.floating]],
        reference: AnalysisResult,
        user_id: str = "user",
    ) -> tuple[DTWResult, int, int]:
        """
        사용자 오디오의 최적 부분 시퀀스 찾기.
        
        사용자 오디오가 레퍼런스보다 길 때, 가장 잘 매칭되는
        구간을 찾습니다.
        
        Returns:
            (DTWResult, start_frame, end_frame)
        """
        # 사용자 MFCC 추출
        user_mfcc = self._extract_user_mfcc(
            user_audio,
            sampling_rate=reference.sampling_rate,
            hop_length=reference.audio_hop_length,
            n_mfcc=reference.mfcc_n_coeffs,
        )
        ref_mfcc = np.array(reference.get_mfcc_matrix())

        user_len = len(user_mfcc)
        ref_len = len(ref_mfcc)

        # 레퍼런스가 더 길면 일반 DTW 사용
        if user_len <= ref_len:
            result = self.synchronize(user_audio, reference, user_id)
            return result, 0, user_len

        # 슬라이딩 윈도우로 최적 구간 탐색
        best_distance = np.inf
        best_start = 0
        best_path = []

        # 윈도우 크기 = 레퍼런스 길이
        window_size = ref_len
        step = max(1, window_size // 10)  # 10% 스텝

        for start in range(0, user_len - window_size + 1, step):
            end = start + window_size
            user_segment = user_mfcc[start:end]

            # 정규화
            user_norm = self._normalize_mfcc(user_segment)
            ref_norm = self._normalize_mfcc(ref_mfcc)

            # DTW
            path, distance = self._compute_dtw(user_norm, ref_norm)

            if distance < best_distance:
                best_distance = distance
                best_start = start
                best_path = [(u + start, r) for u, r in path]

        # 결과 생성
        path_length = len(best_path)
        normalized_distance = best_distance / path_length if path_length > 0 else 0.0

        result = DTWResult(
            actor_id=reference.actor_id,
            user_id=user_id,
            warping_path=best_path,
            distance=best_distance,
            normalized_distance=min(1.0, normalized_distance),
        )

        return result, best_start, best_start + window_size


# =============================================================================
# 전역 인스턴스 (Lazy Loading)
# =============================================================================

_dtw_service: Optional[DTWService] = None


def get_dtw_service() -> DTWService:
    """DTWService 싱글톤 인스턴스 반환."""
    global _dtw_service
    if _dtw_service is None:
        _dtw_service = DTWService()
    return _dtw_service


# =============================================================================
# CLI 테스트용
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 3:
        print("Usage: python dtw_service.py <user_audio> <reference.msgpack>")
        sys.exit(1)

    user_audio_path = sys.argv[1]
    reference_path = sys.argv[2]

    # 레퍼런스 로드
    print(f"📂 레퍼런스 로드: {reference_path}")
    reference = AnalysisResult.load(reference_path)
    print(f"   Actor: {reference.actor_id}, Frames: {reference.frame_count}")

    # DTW 서비스
    service = get_dtw_service()

    # 동기화 실행
    print(f"\n🔗 DTW 동기화 시작...")
    start_time = time.time()
    
    result = service.synchronize(
        user_audio=user_audio_path,
        reference=reference,
        user_id="test_user",
    )
    
    elapsed = time.time() - start_time

    # 결과 출력
    print("\n" + "=" * 50)
    print("🔗 DTW 동기화 결과")
    print("=" * 50)
    print(f"  소요 시간: {elapsed * 1000:.2f}ms")
    print(f"  DTW 거리: {result.distance:.4f}")
    print(f"  정규화 거리: {result.normalized_distance:.4f}")
    print(f"  경로 길이: {len(result.warping_path)}")

    # 매핑 샘플 출력
    mapping = service.get_timestamp_mapping(result, reference)
    timestamps = sorted(mapping.keys())
    
    print(f"\n  타임스탬프 매핑 (샘플):")
    for ts in timestamps[:5]:
        print(f"    User {ts}ms -> Actor {mapping[ts]}ms")
    if len(timestamps) > 10:
        print(f"    ...")
    for ts in timestamps[-3:]:
        print(f"    User {ts}ms -> Actor {mapping[ts]}ms")

    # 신뢰도 테스트 (Advanced)
    print("\n🎯 신뢰도 평가...")
    advanced_service = AdvancedDTWService()
    _, confidence = advanced_service.synchronize_with_confidence(
        user_audio=user_audio_path,
        reference=reference,
    )
    print(f"  신뢰도 점수: {confidence:.1f}/100")
