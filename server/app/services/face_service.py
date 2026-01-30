import numpy as np
import cv2
from app.core import models

def extract_embedding(image_bytes):
    # 바이너리 이미지를 OpenCV 형식으로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 모델을 사용하여 얼굴 분석
    faces = models.face_app.get(img)
    if not faces:
        return None
    
    # 첫 번째 얼굴의 512차원 특징값 반환
    return faces[0].embedding