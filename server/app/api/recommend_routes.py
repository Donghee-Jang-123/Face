#actors, actor_videos json 파일들을 읽어 프엔에게 전달해 줄 코드
import json
import os
from fastapi import APIRouter, HTTPException

router = APIRouter()

# 파일 경로 설정 (app/database 폴더 안에 있는 파일 지칭)
# 실행 위치(main.py) 기준으로 경로를 설정하는 것이 안전합니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTORS_FILE = os.path.join(BASE_DIR, "database", "actors.json")
VIDEOS_FILE = os.path.join(BASE_DIR, "database", "actor_videos.json")

def load_json_data(file_path):
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}") # 디버깅용
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 1. 전체 배우 리스트를 가져오는 API
@router.get("/recommend/actors")
async def get_all_actors():
    actors = load_json_data(ACTORS_FILE)
    return actors

# 2. 특정 배우의 영상 목록만 필터링해서 가져오는 API
@router.get("/actors/{actor_id}/videos")
async def get_actor_videos(actor_id: str):
    all_videos = load_json_data(VIDEOS_FILE)
    
    filtered_videos = [v for v in all_videos if v["actor_id"] == actor_id]
    
    if not filtered_videos:
        raise HTTPException(status_code=404, detail="해당 배우의 영상을 찾을 수 없습니다.")
    
    return filtered_videos