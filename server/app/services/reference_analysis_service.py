"""
Reference Analysis Service (Stage 1)

레퍼런스 영상(.mp4)을 분석하여 프레임 단위로 동기화된 
Audio/Video 특성을 추출하고 AnalysisResult 스키마로 저장합니다.

핵심 특징:
- Frame-Locked Audio: 오디오 특성이 비디오 프레임 타임스탬프에 정확히 정렬됨
- librosa + ffmpeg: Windows 호환성을 위한 안정적인 오디오 처리
- MediaPipe Face Mesh: 478개 랜드마크 기반 블렌드쉐입 계산
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import librosa
import mediapipe as mp
import numpy as np
import torch

from app.core.schemas import (
    AnalysisResult,
    AudioFeatures,
    Blendshapes,
    FrameData,
    VideoFeatures,
)


class ReferenceAnalysisService:
    """
    레퍼런스 영상 분석 서비스.
    
    MP4 파일을 입력받아 프레임 단위로 동기화된 Audio/Video 특성을 추출합니다.
    """

    # 오디오 설정
    TARGET_SAMPLE_RATE = 22050  # librosa 기본값, 음성 분석에 적합
    MFCC_N_COEFFS = 13         # DTW에 사용할 MFCC 계수 개수
    N_FFT = 2048               # FFT 윈도우 크기
    
    # MediaPipe 랜드마크 인덱스 (478개 중 핵심)
    # 참조: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    LM = {
        # 입
        "upperLipTop": 13,
        "lowerLipBottom": 14,
        "mouthLeft": 61,
        "mouthRight": 291,
        "upperLipCenter": 0,
        # 눈
        "leftEyeTop": 159,
        "leftEyeBottom": 145,
        "rightEyeTop": 386,
        "rightEyeBottom": 374,
        "leftEyeInner": 133,
        "leftEyeOuter": 33,
        "rightEyeInner": 362,
        "rightEyeOuter": 263,
        # 눈썹
        "leftBrowInner": 107,
        "rightBrowInner": 336,
        "leftBrowOuter": 70,
        "rightBrowOuter": 300,
        "browCenter": 9,
        # 얼굴 기준점
        "faceLeft": 234,
        "faceRight": 454,
        "faceTop": 10,
        "faceBottom": 152,
        # 코
        "noseLeft": 129,
        "noseRight": 358,
        "noseTip": 1,
        # 볼
        "leftCheek": 50,
        "rightCheek": 280,
        # 눈동자 (refine_landmarks=True 필요)
        "leftPupil": 468,
        "rightPupil": 473,
    }

    def __init__(self):
        """서비스 초기화 및 MediaPipe 로드."""
        print("📊 ReferenceAnalysisService: 초기화 중...")
        
        # MediaPipe Face Mesh 설정
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # 비디오 모드 (트래킹 활성화)
            max_num_faces=1,
            refine_landmarks=True,    # 눈동자 랜드마크 활성화
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # PyTorch 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📊 ReferenceAnalysisService: 초기화 완료! (Device: {self.device})")

    def analyze(
        self,
        video_path: str | Path,
        actor_id: str,
        output_path: Optional[str | Path] = None,
    ) -> AnalysisResult:
        """
        영상을 분석하여 AnalysisResult 반환.
        
        Args:
            video_path: 입력 MP4 파일 경로
            actor_id: 배우/영상 고유 ID
            output_path: 저장할 파일 경로 (None이면 저장하지 않음)
            
        Returns:
            AnalysisResult: 분석 결과
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        print(f"🎬 분석 시작: {video_path.name}")

        # 1. 비디오 메타데이터 추출
        fps, total_frames, duration_sec = self._get_video_metadata(video_path)

        print(f"   📹 FPS: {fps:.2f}, 프레임: {total_frames}, 길이: {duration_sec:.2f}초")

        # 2. 오디오 추출 및 프레임 단위 특성 계산
        audio_features_list = self._extract_audio_features(
            video_path, fps, total_frames
        )

        # 3. 비디오 프레임 단위 처리
        video_features_list = self._extract_video_features(video_path)

        # 4. 프레임 데이터 병합 (길이 맞추기)
        frames = self._merge_features(
            video_features_list, audio_features_list, fps
        )

        # 5. AnalysisResult 생성
        result = AnalysisResult(
            actor_id=actor_id,
            duration_sec=duration_sec,
            fps=fps,
            sampling_rate=self.TARGET_SAMPLE_RATE,
            source_file=video_path.name,
            mfcc_n_coeffs=self.MFCC_N_COEFFS,
            audio_hop_length=self._calculate_hop_length(fps),
            frames=frames,
        )

        print(f"✅ 분석 완료: {len(frames)} 프레임")

        # 6. 저장 (선택적)
        if output_path:
            result.save(output_path)
            print(f"💾 저장 완료: {output_path}")

        return result

    # =========================================================================
    # 오디오 처리
    # =========================================================================

    def _calculate_hop_length(self, fps: float) -> int:
        """
        비디오 FPS에 맞춘 hop_length 계산.
        
        hop_length = sample_rate / fps
        이렇게 하면 오디오 프레임 수 == 비디오 프레임 수
        """
        return int(self.TARGET_SAMPLE_RATE / fps)

    def _extract_audio_features(
        self,
        video_path: Path,
        fps: float,
        total_frames: int,
    ) -> list[AudioFeatures]:
        """
        비디오에서 오디오를 추출하고 프레임 단위 특성 계산.
        
        torchaudio로 빠르게 로드 후, 프레임 단위로 MFCC/Pitch/Energy 추출.
        """
        print("   🔊 오디오 특성 추출 중...")

        # 1. 오디오 로드 (ffmpeg + librosa - Windows 호환)
        audio_np, original_sr = self._load_audio_from_video(video_path)

        # 2. 리샘플링 (필요시 librosa 사용)
        if original_sr != self.TARGET_SAMPLE_RATE:
            audio_np = librosa.resample(
                audio_np,
                orig_sr=original_sr,
                target_sr=self.TARGET_SAMPLE_RATE,
            )

        # 3. Frame-Locked 특성 추출
        hop_length = self._calculate_hop_length(fps)
        
        # MFCC (librosa - 더 정확한 결과)
        mfcc = librosa.feature.mfcc(
            y=audio_np,
            sr=self.TARGET_SAMPLE_RATE,
            n_mfcc=self.MFCC_N_COEFFS,
            n_fft=self.N_FFT,
            hop_length=hop_length,
        )  # shape: (n_mfcc, n_frames)

        # RMS Energy
        rms = librosa.feature.rms(
            y=audio_np,
            frame_length=self.N_FFT,
            hop_length=hop_length,
        )[0]  # shape: (n_frames,)

        # Pitch (F0) - pyin 알고리즘 사용 (정확도 높음)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np,
            fmin=librosa.note_to_hz('C2'),   # 65Hz (낮은 남성 음성)
            fmax=librosa.note_to_hz('C6'),   # 1047Hz (높은 여성 음성)
            sr=self.TARGET_SAMPLE_RATE,
            hop_length=hop_length,
            fill_na=0.0,  # NaN을 0으로 대체
        )

        # 4. AudioFeatures 리스트 생성
        n_audio_frames = mfcc.shape[1]
        audio_features = []

        for i in range(min(n_audio_frames, total_frames)):
            features = AudioFeatures(
                mfcc=mfcc[:, i].tolist(),
                pitch=float(f0[i]) if f0[i] is not None and not np.isnan(f0[i]) else 0.0,
                energy=float(rms[i]) if i < len(rms) else 0.0,
                pitch_confidence=float(voiced_probs[i]) if voiced_probs[i] is not None else 0.0,
                is_voiced=bool(voiced_flag[i]) if voiced_flag[i] is not None else False,
            )
            audio_features.append(features)

        # 부족한 프레임 패딩
        while len(audio_features) < total_frames:
            audio_features.append(AudioFeatures(
                mfcc=[0.0] * self.MFCC_N_COEFFS,
                pitch=0.0,
                energy=0.0,
            ))

        print(f"   🔊 오디오 특성 추출 완료: {len(audio_features)} 프레임")
        return audio_features

    def _load_audio_from_video(self, video_path: Path) -> tuple[np.ndarray, int]:
        """
        비디오에서 오디오 추출 (Windows 호환 - ffmpeg + librosa).
        
        Returns:
            (audio_numpy_array, sample_rate)
        """
        temp_wav = None

        try:
            # 방법 1: ffmpeg로 WAV 추출 후 librosa로 로드 (가장 안정적)
            temp_wav = tempfile.mktemp(suffix=".wav")
            
            # subprocess 사용 (Windows 호환성)
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.TARGET_SAMPLE_RATE), '-ac', '1',
                temp_wav, '-loglevel', 'error'
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # ffmpeg 명령어 형태로 재시도 (PATH 문제 대응)
                os.system(
                    f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le '
                    f'-ar {self.TARGET_SAMPLE_RATE} -ac 1 "{temp_wav}" '
                    f'-loglevel error'
                )

            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                audio_np, sr = librosa.load(temp_wav, sr=self.TARGET_SAMPLE_RATE, mono=True)
                return audio_np, sr

            # 방법 2: librosa 직접 사용 (일부 포맷만 지원)
            audio_np, sr = librosa.load(str(video_path), sr=None, mono=True)
            return audio_np, sr

        finally:
            # 임시 파일 정리
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    # =========================================================================
    # 비디오 처리
    # =========================================================================

    def _get_video_metadata(self, video_path: Path) -> tuple[float, int, float]:
        """
        비디오 메타데이터를 안전하게 추출합니다.
        
        WebM 등 일부 포맷에서는 CAP_PROP_FRAME_COUNT가 -1이나 비정상적인 값을
        반환하므로, 필요시 직접 프레임을 세거나 ffprobe를 사용합니다.
        
        Returns:
            (fps, total_frames, duration_sec)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FPS 기본값 처리
        if fps <= 0 or fps > 240:  # 비정상적인 FPS
            fps = 30.0
            print(f"   ⚠️  FPS를 감지할 수 없어 기본값 {fps}를 사용합니다.")
        
        # 프레임 수가 비정상적인 경우 (WebM 등에서 발생)
        if total_frames <= 0:
            print(f"   ⚠️  프레임 수를 감지할 수 없어 직접 계산합니다...")
            
            # 방법 1: ffprobe 사용 시도 (더 빠름)
            duration_sec = self._get_duration_with_ffprobe(video_path)
            
            if duration_sec and duration_sec > 0:
                total_frames = int(duration_sec * fps)
            else:
                # 방법 2: 직접 프레임 세기 (느리지만 확실함)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                total_frames = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    total_frames += 1
                
                duration_sec = total_frames / fps if fps > 0 else 0
        else:
            duration_sec = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        # 최종 검증
        if duration_sec <= 0:
            raise ValueError(f"영상 길이를 계산할 수 없습니다. (fps={fps}, frames={total_frames})")
        
        return fps, total_frames, duration_sec

    def _get_duration_with_ffprobe(self, video_path: Path) -> float | None:
        """
        ffprobe를 사용하여 비디오 길이 추출.
        
        Returns:
            duration in seconds, or None if failed
        """
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        return None

    def _extract_video_features(self, video_path: Path) -> list[VideoFeatures]:
        """
        비디오에서 프레임 단위 블렌드쉐입 추출.
        """
        print("   👤 비디오 특성 추출 중...")

        cap = cv2.VideoCapture(str(video_path))
        video_features = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 처리
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                blendshapes = self._calculate_blendshapes(landmarks)
                
                features = VideoFeatures(
                    blendshapes=blendshapes,
                    face_detected=True,
                )
            else:
                features = VideoFeatures(
                    blendshapes=None,
                    face_detected=False,
                )

            video_features.append(features)
            frame_idx += 1

            # 진행 상황 출력 (100프레임마다)
            if frame_idx % 100 == 0:
                print(f"      처리 중: {frame_idx} 프레임...")

        cap.release()
        print(f"   👤 비디오 특성 추출 완료: {len(video_features)} 프레임")
        return video_features

    def _calculate_blendshapes(self, landmarks) -> Blendshapes:
        """
        MediaPipe 478개 랜드마크에서 ARKit 스타일 블렌드쉐입 계산.
        
        모든 값은 0.0 ~ 1.0 으로 정규화됩니다.
        """
        def get_point(idx: int) -> np.ndarray:
            """랜드마크 인덱스에서 (x, y, z) 좌표 반환."""
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])

        def dist(idx1: int, idx2: int) -> float:
            """두 랜드마크 사이의 유클리드 거리."""
            return float(np.linalg.norm(get_point(idx1) - get_point(idx2)))

        def dist_y(idx1: int, idx2: int) -> float:
            """두 랜드마크 사이의 Y축 거리 (수직)."""
            return abs(landmarks[idx1].y - landmarks[idx2].y)

        def dist_x(idx1: int, idx2: int) -> float:
            """두 랜드마크 사이의 X축 거리 (수평)."""
            return abs(landmarks[idx1].x - landmarks[idx2].x)

        # 얼굴 기준 크기 (정규화용)
        face_width = dist(self.LM["faceLeft"], self.LM["faceRight"])
        face_height = dist(self.LM["faceTop"], self.LM["faceBottom"])
        
        if face_width < 1e-6 or face_height < 1e-6:
            return Blendshapes()  # 기본값 반환

        # 눈 기준 크기
        left_eye_width = dist(self.LM["leftEyeInner"], self.LM["leftEyeOuter"])
        right_eye_width = dist(self.LM["rightEyeInner"], self.LM["rightEyeOuter"])

        # =====================================================================
        # 블렌드쉐입 계산 (ARKit 스타일)
        # =====================================================================

        # --- 입 관련 ---
        jaw_open = self._clamp(
            dist_y(self.LM["upperLipTop"], self.LM["lowerLipBottom"]) / face_height * 5.0
        )

        mouth_width = dist(self.LM["mouthLeft"], self.LM["mouthRight"])
        base_mouth_width = face_width * 0.35  # 기본 입 너비 비율
        
        # 웃음 (입꼬리가 올라가고 입이 넓어짐)
        mouth_left_y = landmarks[self.LM["mouthLeft"]].y
        mouth_right_y = landmarks[self.LM["mouthRight"]].y
        nose_tip_y = landmarks[self.LM["noseTip"]].y
        
        smile_left = self._clamp(
            (nose_tip_y - mouth_left_y) / face_height * 8.0
        )
        smile_right = self._clamp(
            (nose_tip_y - mouth_right_y) / face_height * 8.0
        )
        
        # 찡그림 (입꼬리가 내려감)
        frown_left = self._clamp(
            (mouth_left_y - nose_tip_y) / face_height * 8.0
        )
        frown_right = self._clamp(
            (mouth_right_y - nose_tip_y) / face_height * 8.0
        )

        # 입 오므림 (입 너비가 좁아짐)
        mouth_pucker = self._clamp(
            (base_mouth_width - mouth_width) / base_mouth_width * 2.0
        )

        # 입 좌우 이동
        mouth_center_x = (landmarks[self.LM["mouthLeft"]].x + landmarks[self.LM["mouthRight"]].x) / 2
        face_center_x = (landmarks[self.LM["faceLeft"]].x + landmarks[self.LM["faceRight"]].x) / 2
        mouth_offset = (mouth_center_x - face_center_x) / face_width
        
        mouth_left = self._clamp(-mouth_offset * 5.0) if mouth_offset < 0 else 0.0
        mouth_right = self._clamp(mouth_offset * 5.0) if mouth_offset > 0 else 0.0

        # --- 눈썹 관련 ---
        brow_inner_up = self._clamp(
            (landmarks[self.LM["browCenter"]].y - 
             (landmarks[self.LM["leftBrowInner"]].y + landmarks[self.LM["rightBrowInner"]].y) / 2) 
            / face_height * 10.0
        )
        
        # 눈썹 내림 (찡그림)
        left_brow_down = self._clamp(
            (landmarks[self.LM["leftBrowInner"]].y - landmarks[self.LM["browCenter"]].y)
            / face_height * 10.0
        )
        right_brow_down = self._clamp(
            (landmarks[self.LM["rightBrowInner"]].y - landmarks[self.LM["browCenter"]].y)
            / face_height * 10.0
        )

        # 눈썹 바깥 올림
        left_brow_outer_up = self._clamp(
            (landmarks[self.LM["faceTop"]].y - landmarks[self.LM["leftBrowOuter"]].y)
            / face_height * 5.0
        )
        right_brow_outer_up = self._clamp(
            (landmarks[self.LM["faceTop"]].y - landmarks[self.LM["rightBrowOuter"]].y)
            / face_height * 5.0
        )

        # --- 눈 관련 ---
        left_eye_open = dist_y(self.LM["leftEyeTop"], self.LM["leftEyeBottom"])
        right_eye_open = dist_y(self.LM["rightEyeTop"], self.LM["rightEyeBottom"])
        
        # 눈 크게 뜸
        eye_wide_left = self._clamp(left_eye_open / left_eye_width * 2.0 - 0.3)
        eye_wide_right = self._clamp(right_eye_open / right_eye_width * 2.0 - 0.3)

        # 눈 찡그림 (눈이 가늘어짐)
        eye_squint_left = self._clamp(1.0 - left_eye_open / left_eye_width * 3.0)
        eye_squint_right = self._clamp(1.0 - right_eye_open / right_eye_width * 3.0)

        # 눈 감기
        base_eye_open = left_eye_width * 0.25
        eye_blink_left = self._clamp(1.0 - left_eye_open / base_eye_open)
        eye_blink_right = self._clamp(1.0 - right_eye_open / base_eye_open)

        # --- 볼/코 관련 ---
        # 볼 부풀리기 (볼이 바깥으로 나감)
        left_cheek_x = landmarks[self.LM["leftCheek"]].x
        right_cheek_x = landmarks[self.LM["rightCheek"]].x
        face_left_x = landmarks[self.LM["faceLeft"]].x
        face_right_x = landmarks[self.LM["faceRight"]].x
        
        cheek_puff = self._clamp(
            ((face_left_x - left_cheek_x) + (right_cheek_x - face_right_x)) 
            / face_width * 5.0
        )

        # 코 찡그림
        nose_sneer_left = self._clamp(
            (landmarks[self.LM["noseLeft"]].y - landmarks[self.LM["noseTip"]].y)
            / face_height * 10.0
        )
        nose_sneer_right = self._clamp(
            (landmarks[self.LM["noseRight"]].y - landmarks[self.LM["noseTip"]].y)
            / face_height * 10.0
        )

        return Blendshapes(
            jawOpen=jaw_open,
            mouthSmileLeft=smile_left,
            mouthSmileRight=smile_right,
            mouthFrownLeft=frown_left,
            mouthFrownRight=frown_right,
            mouthPucker=mouth_pucker,
            mouthLeft=mouth_left,
            mouthRight=mouth_right,
            browInnerUp=brow_inner_up,
            browDownLeft=left_brow_down,
            browDownRight=right_brow_down,
            browOuterUpLeft=left_brow_outer_up,
            browOuterUpRight=right_brow_outer_up,
            eyeWideLeft=eye_wide_left,
            eyeWideRight=eye_wide_right,
            eyeSquintLeft=eye_squint_left,
            eyeSquintRight=eye_squint_right,
            eyeBlinkLeft=eye_blink_left,
            eyeBlinkRight=eye_blink_right,
            cheekPuff=cheek_puff,
            noseSneerLeft=nose_sneer_left,
            noseSneerRight=nose_sneer_right,
        )

    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """값을 min_val과 max_val 사이로 제한."""
        return max(min_val, min(max_val, value))

    # =========================================================================
    # 데이터 병합
    # =========================================================================

    def _merge_features(
        self,
        video_features: list[VideoFeatures],
        audio_features: list[AudioFeatures],
        fps: float,
    ) -> list[FrameData]:
        """
        비디오/오디오 특성을 프레임 단위로 병합.
        """
        frame_duration_ms = 1000.0 / fps
        n_frames = min(len(video_features), len(audio_features))

        frames = []
        for i in range(n_frames):
            timestamp_ms = int(i * frame_duration_ms)
            
            frame = FrameData(
                timestamp_ms=timestamp_ms,
                video=video_features[i] if i < len(video_features) else None,
                audio=audio_features[i] if i < len(audio_features) else None,
            )
            frames.append(frame)

        return frames


# =============================================================================
# 전역 인스턴스 (Lazy Loading)
# =============================================================================

_reference_analysis_service: Optional[ReferenceAnalysisService] = None


def get_reference_analysis_service() -> ReferenceAnalysisService:
    """ReferenceAnalysisService 싱글톤 인스턴스 반환."""
    global _reference_analysis_service
    if _reference_analysis_service is None:
        _reference_analysis_service = ReferenceAnalysisService()
    return _reference_analysis_service


# =============================================================================
# CLI 테스트용
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python reference_analysis_service.py <video_path> [actor_id]")
        sys.exit(1)

    video_path = sys.argv[1]
    actor_id = sys.argv[2] if len(sys.argv) > 2 else "test_actor"

    service = get_reference_analysis_service()
    
    # 분석 실행
    result = service.analyze(
        video_path=video_path,
        actor_id=actor_id,
        output_path=f"{actor_id}_analysis.msgpack",
    )

    # 요약 출력
    print("\n" + "=" * 50)
    print("📊 분석 결과 요약")
    print("=" * 50)
    print(f"  Actor ID: {result.actor_id}")
    print(f"  Duration: {result.duration_sec:.2f}초")
    print(f"  FPS: {result.fps}")
    print(f"  Frames: {result.frame_count}")
    print(f"  Has Audio: {result.has_audio}")
    print(f"  Has Video: {result.has_video}")

    # 샘플 프레임 출력
    if result.frames:
        sample = result.frames[len(result.frames) // 2]
        print(f"\n  샘플 프레임 (중간):")
        print(f"    Timestamp: {sample.timestamp_ms}ms")
        if sample.audio:
            print(f"    Pitch: {sample.audio.pitch:.2f}Hz")
            print(f"    Energy: {sample.audio.energy:.6f}")
            print(f"    MFCC[0]: {sample.audio.mfcc[0]:.4f}")
        if sample.video and sample.video.blendshapes:
            print(f"    JawOpen: {sample.video.blendshapes.jawOpen:.4f}")
            print(f"    SmileL: {sample.video.blendshapes.mouthSmileLeft:.4f}")
