from insightface.app import FaceAnalysis

# 모델을 전역 변수로 선언하여 공유합니다.
face_app = None

def load_models():
    global face_app
    # CPU 환경에 최적화하여 모델 로드
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    print("AI 모델 로드 완료!")