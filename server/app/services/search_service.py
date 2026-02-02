import faiss
import numpy as np
import os
import json
import pickle  # [New] 배우 데이터(.pkl)를 읽기 위해 추가

# 데이터 저장 경로
DB_DIR = "app/database"
INDEX_FILE = os.path.join(DB_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(DB_DIR, "users.json")
ACTOR_EMBEDDINGS_FILE = os.path.join(DB_DIR, "actor_embeddings.pkl") # [New] 배우 족보 파일 경로

# 얼굴 특징 벡터의 차원 수 (InsightFace buffalo_l 모델은 512)
DIMENSION = 512

class VectorStore:
    def __init__(self):
        self.index = None
        self.users = {}  # { "0": "batman", "1": "joker" } 형태
        self.next_id = 0  # 다음에 할당할 ID
        
        # 1. 사용자 데이터 로드 (기존 기능)
        self._load_or_create_index()
        
        # 2. [New] 배우 데이터 로드 (새로운 기능!)
        self.actor_embeddings = {}
        self._load_actor_embeddings()

    def _load_or_create_index(self):
        """기존 FAISS 인덱스 로드 (사용자용)"""
        os.makedirs(DB_DIR, exist_ok=True)

        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.users = data["users"]
                self.next_id = data["next_id"]
            print(f"📂 FAISS 인덱스 로드 완료! (총 {self.index.ntotal}명)")
        else:
            base_index = faiss.IndexFlatIP(DIMENSION)
            self.index = faiss.IndexIDMap(base_index)
            self.users = {}
            self.next_id = 0
            print("🆕 새로운 FAISS 인덱스를 생성했습니다.")

    def _load_actor_embeddings(self):
        """[New] 배우 임베딩 파일(.pkl)을 불러옵니다."""
        if os.path.exists(ACTOR_EMBEDDINGS_FILE):
            with open(ACTOR_EMBEDDINGS_FILE, "rb") as f:
                self.actor_embeddings = pickle.load(f)
            print(f"🎬 배우 데이터 로드 완료! (총 {len(self.actor_embeddings)}명)")
        else:
            print("⚠️ 배우 데이터 파일이 없습니다. (preprocess_actors.py를 먼저 실행해주세요.)")

    def add_user(self, embedding, nickname):
        """사용자 추가 (기존 기능 유지)"""
        vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        
        new_id = self.next_id
        self.next_id += 1
        
        ids = np.array([new_id], dtype=np.int64)
        self.index.add_with_ids(vector, ids)
        
        self.users[str(new_id)] = nickname
        self._save()
        return new_id
    
    def delete_user(self, user_id):
        """사용자 삭제 (기존 기능 유지)"""
        user_id_str = str(user_id)
        if user_id_str not in self.users:
            return {"success": False, "message": f"ID {user_id}에 해당하는 사용자가 없습니다."}
        
        ids_to_remove = np.array([int(user_id)], dtype=np.int64)
        removed_count = self.index.remove_ids(ids_to_remove)
        nickname = self.users.pop(user_id_str)
        self._save()
        
        return {"success": True, "message": f"'{nickname}' 삭제 완료", "removed_count": int(removed_count)}
    
    def get_all_users(self):
        return [{"id": int(k), "nickname": v} for k, v in self.users.items()]
        
    def search_user(self, embedding, threshold=0.45):
        """로그인용 사용자 검색 (기존 기능 유지)"""
        vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        
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

    def search_similar_actor(self, user_embedding):
        """
        [New] 사용자와 가장 닮은 배우를 찾습니다.
        """
        if not self.actor_embeddings:
            # 데이터가 없으면 에러 방지를 위해 기본값 반환
            return "actor_03" 

        best_actor_id = None
        best_score = -1.0
        
        # 사용자 임베딩 정규화
        user_vec = np.array(user_embedding, dtype=np.float32)
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = user_vec / norm

        # 모든 배우와 비교 (Cosine Similarity)
        for actor_id, actor_vec in self.actor_embeddings.items():
            a_vec = np.array(actor_vec, dtype=np.float32)
            a_norm = np.linalg.norm(a_vec)
            if a_norm > 0:
                a_vec = a_vec / a_norm
            
            score = np.dot(user_vec, a_vec)
            
            if score > best_score:
                best_score = score
                best_actor_id = actor_id
        
        print(f"🧐 닮은꼴 분석 결과: {best_actor_id} (유사도: {best_score:.4f})")
        return best_actor_id

    def _save(self):
        """데이터 저장 (기존 기능 유지)"""
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            data = {
                "users": self.users,
                "next_id": self.next_id
            }
            json.dump(data, f, ensure_ascii=False, indent=4)

# 전역 객체 생성
vector_store = VectorStore()