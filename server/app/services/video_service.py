import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import numpy as np
from scipy.spatial import distance

class VideoService:
    def __init__(self):
        print("🎬 VideoService: MediaPipe 로딩 중...")
        self.mp_face_mesh = mp.solutions.face_mesh
        # 정밀도를 위해 refine_landmarks=True 사용 (입술/눈 주변 세밀함)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("🎬 VideoService: 로딩 완료!")

    def _calculate_blendshapes(self, landmarks):
        """
        MediaPipe 478개 랜드마크를 기반으로 핵심 표정(Blendshapes) 수치를 계산
        (실제 아이폰 ARKit의 Blendshape와 유사한 로직 구현)
        """
        def get_dist(idx1, idx2):
            # 두 랜드마크 사이의 유클리드 거리 계산
            p1 = np.array([landmarks[idx1].x, landmarks[idx1].y])
            p2 = np.array([landmarks[idx2].x, landmarks[idx2].y])
            return np.linalg.norm(p1 - p2)

        # 얼굴 크기 기준점 (양쪽 관자놀이 거리) - 정규화용
        face_width = get_dist(234, 454) 
        face_height = get_dist(10, 152)
        if face_width == 0 or face_height == 0: return None

        # --- 핵심 연기 포인트 (Action Units) 계산 ---
        # 값은 0~1 사이로 정규화하되, 사람마다 다르므로 상대적 변화량이 중요함
        shapes = {
            "jawOpen": get_dist(13, 14) / face_height,       # 입 벌림
            "mouthSmile": get_dist(61, 291) / face_width,   # 입꼬리 좌우 길이 
            "browInnerUp": (get_dist(107, 9) + get_dist(9,336)) / (2 * face_width),   # 눈썹 치켜뜸 (놀람)
            "eyeWide": (get_dist(159, 145) + get_dist(386, 374)) / (2 * face_width), # 눈 크게 뜸
            "mouthFrown": (((landmarks[61].y + landmarks[291].y) / 2) - landmarks[0].y) / face_height,  # 입꼬리 높이이
            "pupil": (get_dist(468, 133) / get_dist(133, 33)),  # 눈동자 위치
            "philtrum" : (get_dist(0,2)) / face_height # 인중
        }
        return shapes

    def process_video_shapes(self, video_path: str):
        """
        영상 전체를 프레임 단위로 분석하여 시계열 Blendshape 데이터를 추출
        """
        cap = cv2.VideoCapture(video_path)
        shape_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # BGR -> RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                shapes = self._calculate_blendshapes(landmarks)
                if shapes:
                    shape_sequence.append(shapes)
        
        cap.release()
        return shape_sequence

    def calculate_sync_rate(self, user_shapes, target_shapes):
        """
        두 영상의 Blendshape 시퀀스를 비교하여 항목별 싱크로율(%) 계산
        Returns:
            dict: 항목별 점수와 총점을 포함한 딕셔너리
        """
        keys = ["jawOpen", "mouthSmile", "browInnerUp", "eyeWide", "mouthFrown", "pupil", "philtrum"]
        
        # 항목별 한글 이름 매핑
        key_names = {
            "jawOpen": "입 벌림",
            "mouthSmile": "입꼬리 (미소)",
            "browInnerUp": "눈썹 치켜뜸",
            "eyeWide": "눈 크게 뜸",
            "mouthFrown": "입꼬리 내림",
            "pupil": "눈동자 위치",
            "philtrum": "인중"
        }
        
        empty_result = {
            "total": 0.0,
            "details": {key: {"score": 0.0, "name": key_names[key]} for key in keys}
        }
        
        if not user_shapes or not target_shapes:
            return empty_result

        # 1. 프레임 수 맞추기 (간단히 최소 길이로 맞춤, 고도화 시 DTW 알고리즘 권장)
        min_len = min(len(user_shapes), len(target_shapes))
        if min_len == 0: 
            return empty_result
        
        user_seq = user_shapes[:min_len]
        target_seq = target_shapes[:min_len]

        # 항목별 점수를 누적할 딕셔너리
        key_scores = {key: [] for key in keys}

        for i in range(min_len):
            for key in keys:
                u_val = user_seq[i][key]
                t_val = target_seq[i][key]
                
                # 차이가 작을수록 점수가 높음 (거리 기반)
                diff = abs(u_val - t_val)
                # 0.1 차이면 0점, 0 차이면 100점이 되도록 스케일링
                similarity = max(0, 1 - (diff * 10)) 
                key_scores[key].append(similarity)

        # 항목별 평균 점수 계산
        details = {}
        total_scores = []
        for key in keys:
            avg_score = (sum(key_scores[key]) / len(key_scores[key])) * 100
            rounded_score = round(avg_score, 2)
            details[key] = {
                "score": rounded_score,
                "name": key_names[key]
            }
            total_scores.append(rounded_score)

        # 총점 계산 (항목별 점수의 평균)
        total = round(sum(total_scores) / len(total_scores), 2)

        return {
            "total": total,
            "details": details
        }

# 전역 인스턴스는 lazy loading으로 변경
video_service = None

def get_video_service():
    global video_service
    if video_service is None:
        video_service = VideoService()
    return video_service
