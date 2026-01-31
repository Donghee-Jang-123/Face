import torch
import torchaudio
import librosa
import numpy as np
import scipy.signal
import os
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
# .editor를 지우고 아래처럼 길게 써주셔야 합니다!
from moviepy.video.io.VideoFileClip import VideoFileClip

class AudioService:
    def __init__(self):
        print("🔊 AudioService: 모델 로딩 시작...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "superb/wav2vec2-base-superb-er"
        self.target_sample_rate = 16000 
        
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"🔊 AudioService: 로딩 완료! (Device: {self.device})")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise e

    def _load_and_resample(self, file_path: str):
        # [핵심 수정] MoviePy를 사용하여 강제로 오디오 추출 (가장 확실한 방법)
        temp_wav = file_path + ".temp.wav"
        
        try:
            # 1. 파일이 존재하는지 먼저 확인
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

            # 2. 동영상 파일(.mp4)인 경우 MoviePy로 오디오 추출
            if file_path.lower().endswith('.mp4') or file_path.lower().endswith('.avi') or file_path.lower().endswith('.mov'):
                try:
                    # VideoFileClip을 사용해 오디오만 따로 저장
                    video = VideoFileClip(file_path)
                    if video.audio is None:
                        raise ValueError("이 동영상에는 소리가 없습니다.")
                    
                    # 16000Hz로 변환하여 임시 wav 파일 저장 (logger=None으로 로그 끄기)
                    video.audio.write_audiofile(temp_wav, fps=self.target_sample_rate, logger=None)
                    video.close()
                    
                    # 저장된 wav 파일을 Librosa로 로드
                    audio_array, _ = librosa.load(temp_wav, sr=self.target_sample_rate, mono=True)
                    return audio_array
                    
                except Exception as e:
                    print(f"⚠️ MoviePy 추출 실패, Librosa 직접 시도: {e}")
                    # 실패 시 기존 방식 시도
                    pass
                finally:
                    # 임시 파일 삭제 (청소)
                    if os.path.exists(temp_wav):
                        try:
                            os.remove(temp_wav)
                        except:
                            pass

            # 3. 일반 오디오 파일이거나 MoviePy 실패 시 Librosa 사용
            audio_array, _ = librosa.load(file_path, sr=self.target_sample_rate, mono=True)
            return audio_array

        except Exception as e:
            print(f"❌ 오디오 로딩 최종 실패 ({file_path}): {e}")
            raise e

    def _trim_silence(self, audio_array):
        try:
            trimmed_audio, _ = librosa.effects.trim(audio_array, top_db=20)
            return trimmed_audio
        except:
            return audio_array

    def _filter_noise(self, audio_array):
        try:
            sos = scipy.signal.butter(10, 100, 'hp', fs=self.target_sample_rate, output='sos')
            filtered_audio = scipy.signal.sosfilt(sos, audio_array)
            return filtered_audio
        except:
            return audio_array

    def analyze_emotion(self, file_path: str):
        try:
            raw_audio = self._load_and_resample(file_path)
            trimmed_audio = self._trim_silence(raw_audio)
            clean_audio = self._filter_noise(trimmed_audio)

            if len(clean_audio) < 1600:
                clean_audio = raw_audio

            inputs = self.processor(
                clean_audio, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(logits, dim=-1).item()
            emotion = self.model.config.id2label[predicted_id]
            confidence = probs[0][predicted_id].item()
            
            return {
                "emotion": emotion,
                "confidence": round(confidence * 100, 2)
            }
            
        except Exception as e:
            print(f"❌ Audio Analysis Error: {e}")
            return {"emotion": "neutral", "confidence": 0.0}

audio_service = None

def get_audio_service():
    global audio_service
    if audio_service is None:
        audio_service = AudioService()
    return audio_service