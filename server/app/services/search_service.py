import faiss
import numpy as np
import os
import json

# 데이터 저장 경로
DB_DIR = "app/database"
INDEX_FILE = os.path.join(DB_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(DB_DIR, "users.json")

# 얼굴 특징 벡터의 차원 수 (InsightFace buffalo_l 모델은 512)
DIMENSION = 512

class VectorStore:
    def __init__(self):
        self.index = None
        self.users = {}  # { "0": "batman", "1": "joker" } 형태
        self.next_id = 0  # 다음에 할당할 ID (삭제 후에도 계속 증가)
        self._load_or_create_index()

    def _load_or_create_index(self):
        # 1. 폴더가 없으면 생성
        os.makedirs(DB_DIR, exist_ok=True)

        # 2. 기존 인덱스 파일 로드
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.users = data["users"]
                self.next_id = data["next_id"]
            print(f"📂 FAISS 인덱스 로드 완료! (총 {self.index.ntotal}명)")
        else:
            # 3. 없으면 새로 생성 (IndexIDMap으로 래핑하여 삭제 지원)
            base_index = faiss.IndexFlatIP(DIMENSION)  # 내적(Cosine 유사도와 유사) 사용
            self.index = faiss.IndexIDMap(base_index)
            self.users = {}
            self.next_id = 0
            print("🆕 새로운 FAISS 인덱스를 생성했습니다.")

    def add_user(self, embedding, nickname):
        # FAISS는 float32 형태만 받습니다.
        vector = np.array([embedding], dtype=np.float32)
        
        # 벡터 정규화 (코사인 유사도를 위해 필수)
        faiss.normalize_L2(vector)
        
        # 1. 새 ID 할당
        new_id = self.next_id
        self.next_id += 1
        
        # 2. FAISS에 ID와 함께 추가
        ids = np.array([new_id], dtype=np.int64)
        self.index.add_with_ids(vector, ids)
        
        # 3. JSON 명부에 이름 기록
        self.users[str(new_id)] = nickname
        
        # 4. 파일로 저장
        self._save()
        
        return new_id
    
    def delete_user(self, user_id):
        """특정 사용자의 얼굴 정보를 삭제합니다."""
        user_id_str = str(user_id)
        
        # 1. 사용자 존재 여부 확인
        if user_id_str not in self.users:
            return {"success": False, "message": f"ID {user_id}에 해당하는 사용자가 없습니다."}
        
        # 2. FAISS 인덱스에서 삭제
        ids_to_remove = np.array([int(user_id)], dtype=np.int64)
        removed_count = self.index.remove_ids(ids_to_remove)
        
        # 3. users 딕셔너리에서 삭제
        nickname = self.users.pop(user_id_str)
        
        # 4. 파일로 저장
        self._save()
        
        return {
            "success": True, 
            "message": f"'{nickname}' (ID: {user_id}) 삭제 완료",
            "removed_count": int(removed_count)
        }
    
    def get_all_users(self):
        """등록된 모든 사용자 목록을 반환합니다."""
        return [{"id": int(k), "nickname": v} for k, v in self.users.items()]
        
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
            # next_id도 함께 저장하여 삭제 후에도 ID 충돌 방지
            data = {
                "users": self.users,
                "next_id": self.next_id
            }
            json.dump(data, f, ensure_ascii=False, indent=4)

# 전역에서 사용할 수 있도록 인스턴스 생성
vector_store = VectorStore()