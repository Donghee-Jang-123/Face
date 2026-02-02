import sys
import os
import json
import pickle
import asyncio

# 1. 현재 스크립트 위치를 기준으로 상위 폴더(server)를 시스템 경로에 추가
# (이렇게 해야 app.services 등을 import 할 수 있습니다)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import face_service
from app.core import models

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTORS_FILE = os.path.join(BASE_DIR, "app", "database", "actors.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "app", "database", "actor_embeddings.pkl")

def preprocess_actors():
    print("🚀 배우 데이터 전처리를 시작합니다...")

    # 1. 모델 로드 (InsightFace 준비)
    models.load_models()

    # 2. actors.json 읽기
    if not os.path.exists(ACTORS_FILE):
        print(f"❌ 오류: {ACTORS_FILE} 파일을 찾을 수 없습니다.")
        return

    with open(ACTORS_FILE, "r", encoding="utf-8") as f:
        actors = json.load(f)

    actor_embeddings = {}
    count = 0

    # 3. 각 배우 사진에서 임베딩 추출
    for actor in actors:
        actor_id = actor["actor_id"]
        # JSON에 있는 이미지 경로 (/assets/actors/...)를 실제 파일 시스템 경로로 변환
        # 예: /assets/actors/actor_01.jpg -> server/assets/actors/actor_01.jpg
        relative_path = actor["thumbnail"].lstrip("/") # 맨 앞의 / 제거
        image_path = os.path.join(BASE_DIR, relative_path)

        if not os.path.exists(image_path):
            print(f"⚠️ 경고: {actor['name']}님의 사진을 찾을 수 없습니다. ({image_path})")
            continue

        print(f"📸 분석 중: {actor['name']} ({image_path})...")

        # 이미지를 바이트로 읽어서 face_service에 전달
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            embedding = face_service.extract_embedding(image_bytes)

        if embedding is not None:
            # { "actor_01": [512차원 벡터], ... } 형태로 저장
            actor_embeddings[actor_id] = embedding
            count += 1
            print(f"✅ {actor['name']} 임베딩 추출 완료!")
        else:
            print(f"❌ 실패: {actor['name']}님의 사진에서 얼굴을 찾을 수 없습니다.")

    # 4. 결과 파일 저장 (Pickle 사용)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(actor_embeddings, f)

    print("-" * 30)
    print(f"🎉 총 {count}명의 배우 데이터가 '{OUTPUT_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    preprocess_actors()