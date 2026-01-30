import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import scipy.signal
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class AudioService:
    def __init__(self):
        print("🔊 AudioService: 모델 로딩 시작...")
        
        # 1. GPU 가속 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "superb/wav2vec2-base-superb-er"
        self.target_sample_rate = 16000 
        
        # 2. 모델 로드
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"🔊 AudioService: 로딩 완료! (Device: {self.device})")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise e

    def _load_and_resample(self, file_path: str):
        # A. 로드
        waveform, sample_rate = torchaudio.load(file_path)
        
        # B. 리샘플링
        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            
        # C. Stereo -> Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform.squeeze().numpy()

    def _trim_silence(self, audio_array):
        # 무음 제거 (20dB 기준)
        trimmed_audio, _ = librosa.effects.trim(audio_array, top_db=20)
        return trimmed_audio

    def _filter_noise(self, audio_array):
        # 100Hz 이하 노이즈 제거
        sos = scipy.signal.butter(10, 100, 'hp', fs=self.target_sample_rate, output='sos')
        filtered_audio = scipy.signal.sosfilt(sos, audio_array)
        return filtered_audio

    def analyze_emotion(self, file_path: str):
        try:
            # 1. 전처리
            raw_audio = self._load_and_resample(file_path)
            trimmed_audio = self._trim_silence(raw_audio)
            clean_audio = self._filter_noise(trimmed_audio)

            # 너무 짧으면(0.1초 미만) 원본 사용
            if len(clean_audio) < 1600:
                clean_audio = raw_audio

            # 2. 모델 입력 변환 (길이 제한 제거됨)
            inputs = self.processor(
                clean_audio, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # 3. 추론
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # 4. 결과 해석
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

# 전역 인스턴스는 lazy loading으로 변경 (서버 시작 시 로드)
audio_service = None

def get_audio_service():
    global audio_service
    if audio_service is None:
        audio_service = AudioService()
    return audio_service
