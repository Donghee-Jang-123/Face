import faiss
import numpy as np
import os
import json
import pickle

# 데이터 저장 경로
DB_DIR = "app/database"
INDEX_FILE = os.path.join(DB_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(DB_DIR, "users.json")

# 얼굴 특징 벡터의 차원 수 (InsightFace buffalo_l 모델은 512)
DIMENSION = 512

class VectorStore:
    def __init__(self):
        self.index = None
        self.users = {} # { "0": "batman", "1": "joker" } 형태
        self._load_or_create_index()

    def _load_or_create_index(self):
        # 1. 폴더가 없으면 생성
        os.makedirs(DB_DIR, exist_ok=True)

        # 2. 기존 인덱스 파일 로드
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                self.users = json.load(f)
            print(f"📂 FAISS 인덱스 로드 완료! (총 {self.index.ntotal}명)")
        else:
            # 3. 없으면 새로 생성 (L2 거리 기반 인덱스)
            self.index = faiss.IndexFlatIP(DIMENSION) # 내적(Cosine 유사도와 유사) 사용
            self.users = {}
            print("🆕 새로운 FAISS 인덱스를 생성했습니다.")

    def add_user(self, embedding, nickname):
        # FAISS는 float32 형태만 받습니다.
        vector = np.array([embedding], dtype=np.float32)
        
        # 벡터 정규화 (코사인 유사도를 위해 필수)
        faiss.normalize_L2(vector)
        
        # 1. FAISS에 추가
        self.index.add(vector)
        
        # 2. JSON 명부에 이름 기록 (현재 ID = 전체 개수 - 1)
        new_id = self.index.ntotal - 1
        self.users[str(new_id)] = nickname
        
        # 3. 파일로 저장
        self._save()
        
    def search_user(self, embedding, threshold=0.45):
        vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        
        # 가장 유사한 1명(k=1) 검색
        # D: 거리(점수), I: 인덱스(ID)
        D, I = self.index.search(vector, 1)
        
        best_score = D[0][0]
        best_id = I[0][0]
        
        if best_score > threshold and str(best_id) in self.users:
            return {
                "found": True,
                "nickname": self.users[str(best_id)],
                "score": float(best_score)
            }
        else:
            return {"found": False}

    def _save(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.users, f, ensure_ascii=False, indent=4)

# 전역에서 사용할 수 있도록 인스턴스 생성
vector_store = VectorStore()