'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';

// 카메라 에러 타입 정의
type CameraError = 'permission_denied' | 'not_found' | 'not_supported' | 'unknown' | null;

// =========================================================================
// Ultra-Precision 분석 결과 타입 정의
// =========================================================================

interface SubMetric {
  name: string;
  score: number;
  weight: number;
  feedback: string;
  details: Record<string, unknown>;
}

interface ScoreDetail {
  score: number;
  feedback: string;
  weight: number;
  sub_metrics: SubMetric[];
}

interface AnalysisResult {
  total_score: number;
  grade: string;
  details: {
    pitch: ScoreDetail;
    energy: ScoreDetail;
    expression: ScoreDetail;
  };
  overall_feedback: string;
  actor_id: string;
  user_id: string;
}

// 서브메트릭 표시 이름 매핑
const SUB_METRIC_NAMES: Record<string, string> = {
  // Pitch
  pattern_match: '패턴 매칭',
  dynamic_range: '다이내믹 레인지',
  // Energy
  intensity: '인텐시티',
  // Expression
  eyes: '눈 표현',
  mouth: '입 표현',
  brows: '눈썹 표현',
};

// 서브메트릭 아이콘 매핑
const SUB_METRIC_ICONS: Record<string, string> = {
  pattern_match: '🎵',
  dynamic_range: '📊',
  intensity: '💪',
  eyes: '👁️',
  mouth: '👄',
  brows: '🤨',
};

// =========================================================================
// 삼각형 레이더 차트 컴포넌트
// =========================================================================
interface RadarChartProps {
  pitch: number;
  energy: number;
  expression: number;
  size?: number;
}

function RadarChart({ pitch, energy, expression, size = 180 }: RadarChartProps) {
  const center = size / 2;
  const maxRadius = size / 2 - 30; // 라벨을 위한 여유 공간
  
  // 삼각형 꼭지점 각도 (12시 방향부터 시계방향: 표정, 볼륨, 억양)
  const angles = [
    -90,  // 표정 (위쪽, 12시)
    150,  // 볼륨 (오른쪽 아래, 5시)
    30,   // 억양 (왼쪽 아래, 7시)
  ];
  
  // 각도를 라디안으로 변환
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  
  // 점수를 좌표로 변환 (0-100 → 0-maxRadius)
  const scoreToPoint = (score: number, angleIndex: number) => {
    const radius = (score / 100) * maxRadius;
    const angle = toRad(angles[angleIndex]);
    return {
      x: center + radius * Math.cos(angle),
      y: center + radius * Math.sin(angle),
    };
  };
  
  // 배경 그리드 좌표 (100%, 75%, 50%, 25%)
  const gridLevels = [100, 75, 50, 25];
  const getGridPoints = (level: number) => {
    return angles.map((_, i) => scoreToPoint(level, i));
  };
  
  // 데이터 포인트
  const dataPoints = [
    scoreToPoint(expression, 0),  // 표정
    scoreToPoint(energy, 1),      // 볼륨
    scoreToPoint(pitch, 2),       // 억양
  ];
  
  // 라벨 위치 (그리드 바깥쪽)
  const labelOffset = 25;
  const labelPositions = angles.map((angle, i) => {
    const rad = toRad(angle);
    return {
      x: center + (maxRadius + labelOffset) * Math.cos(rad),
      y: center + (maxRadius + labelOffset) * Math.sin(rad),
    };
  });
  
  // 점수 배열
  const scores = [expression, energy, pitch];
  const labels = ['표정', '볼륨', '억양'];
  const colors = ['text-cyan-400', 'text-orange-400', 'text-pink-400'];

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="overflow-visible">
        {/* 배경 그리드 */}
        {gridLevels.map((level) => {
          const points = getGridPoints(level);
          const pathD = `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y} L ${points[2].x} ${points[2].y} Z`;
          return (
            <path
              key={level}
              d={pathD}
              fill="none"
              stroke="rgba(75, 85, 99, 0.5)"
              strokeWidth={level === 100 ? 1.5 : 1}
              strokeDasharray={level === 100 ? "0" : "3,3"}
            />
          );
        })}
        
        {/* 축 선 (중심에서 각 꼭지점까지) */}
        {angles.map((_, i) => {
          const endPoint = scoreToPoint(100, i);
          return (
            <line
              key={i}
              x1={center}
              y1={center}
              x2={endPoint.x}
              y2={endPoint.y}
              stroke="rgba(75, 85, 99, 0.4)"
              strokeWidth={1}
            />
          );
        })}
        
        {/* 데이터 영역 (채워진 삼각형) */}
        <defs>
          <linearGradient id="radarGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="rgba(139, 92, 246, 0.6)" />
            <stop offset="50%" stopColor="rgba(59, 130, 246, 0.6)" />
            <stop offset="100%" stopColor="rgba(6, 182, 212, 0.6)" />
          </linearGradient>
        </defs>
        <path
          d={`M ${dataPoints[0].x} ${dataPoints[0].y} L ${dataPoints[1].x} ${dataPoints[1].y} L ${dataPoints[2].x} ${dataPoints[2].y} Z`}
          fill="url(#radarGradient)"
          stroke="rgba(139, 92, 246, 0.8)"
          strokeWidth={2}
          className="transition-all duration-700"
        />
        
        {/* 데이터 포인트 (점) */}
        {dataPoints.map((point, i) => (
          <g key={i}>
            {/* 외부 원 (글로우 효과) */}
            <circle
              cx={point.x}
              cy={point.y}
              r={8}
              fill={i === 0 ? 'rgba(6, 182, 212, 0.3)' : i === 1 ? 'rgba(251, 146, 60, 0.3)' : 'rgba(236, 72, 153, 0.3)'}
              className="transition-all duration-700"
            />
            {/* 내부 원 */}
            <circle
              cx={point.x}
              cy={point.y}
              r={5}
              fill={i === 0 ? '#06b6d4' : i === 1 ? '#fb923c' : '#ec4899'}
              stroke="white"
              strokeWidth={2}
              className="transition-all duration-700"
            />
          </g>
        ))}
      </svg>
      
      {/* 라벨 (SVG 외부에 배치) */}
      {labelPositions.map((pos, i) => (
        <div
          key={i}
          className="absolute flex flex-col items-center"
          style={{
            left: pos.x,
            top: pos.y,
            transform: 'translate(-50%, -50%)',
          }}
        >
          <span className={`text-lg font-bold ${colors[i]}`}>
            {scores[i].toFixed(0)}
          </span>
          <span className="text-xs text-gray-400 whitespace-nowrap">{labels[i]}</span>
        </div>
      ))}
    </div>
  );
}

// 단어별 타임스탬프 타입
interface WordTimestamp {
  text: string;
  start: number;
  end: number;
}

// 문장 단위 타입
interface Sentence {
  text: string;
  start: number;
  end: number;
  words: WordTimestamp[];
}

// 선택된 비디오 타입 정의
interface SelectedVideo {
  video_id: string;
  actor_id: string;
  title: string;
  video_url: string;
  thumbnail?: string;
  script?: string;  // 영상별 대사 (전체)
  sentences?: Sentence[];  // 문장 단위 + 단어별 타임스탬프
}

// =========================================================================
// 카라오케 스타일 자막 컴포넌트 (문장 단위)
// =========================================================================
interface SubtitleOverlayProps {
  sentences: Sentence[];
  currentTime: number;
}

function SubtitleOverlay({ sentences, currentTime }: SubtitleOverlayProps) {
  // 현재 시간에 해당하는 문장 찾기
  const currentSentence = sentences.find(
    s => currentTime >= s.start && currentTime < s.end
  );
  
  if (!currentSentence) {
    return null;
  }
  
  // 단어별 타임스탬프가 있으면 하이라이트, 없으면 문장만 표시
  const hasWords = currentSentence.words && currentSentence.words.length > 0;
  
  return (
    <div className="absolute bottom-8 left-0 right-0 flex justify-center pointer-events-none z-10">
      <div className="bg-black/70 backdrop-blur-sm px-6 py-3 rounded-xl border border-white/20 shadow-2xl">
        {hasWords ? (
          <div className="flex flex-wrap gap-x-2 justify-center items-center">
            {currentSentence.words.map((word, index) => {
              const isActive = currentTime >= word.start && currentTime < word.end;
              const isPast = currentTime >= word.end;
              
              return (
                <span
                  key={index}
                  className={`
                    text-2xl font-bold transition-all duration-100
                    ${isActive 
                      ? 'text-yellow-300 scale-110 drop-shadow-[0_0_10px_rgba(253,224,71,0.8)]' 
                      : isPast 
                        ? 'text-white/50' 
                        : 'text-white'
                    }
                  `}
                >
                  {word.text}
                </span>
              );
            })}
          </div>
        ) : (
          <p className="text-2xl font-bold text-white text-center">
            {currentSentence.text}
          </p>
        )}
      </div>
    </div>
  );
}

export default function ActingPage() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const referenceVideoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  
  // 상태 관리
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<CameraError>(null);
  const [isCameraLoading, setIsCameraLoading] = useState(true);
  
  // 녹화 관련 상태
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<{ success: boolean; message: string } | null>(null);
  
  // 녹화 시간 트래킹 (자막 싱크용)
  const [recordingStartTime, setRecordingStartTime] = useState<number | null>(null);
  const [recordingElapsedTime, setRecordingElapsedTime] = useState(0);
  
  // 분석 결과 상태
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  
  // 상세 분석 펼침 상태
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({
    pitch: true,
    energy: true,
    expression: true,
  });

  // 선택된 비디오 상태
  const [selectedVideo, setSelectedVideo] = useState<SelectedVideo | null>(null);

  // 레퍼런스 비디오 현재 재생 시간 (카라오케 싱크용)
  const [videoCurrentTime, setVideoCurrentTime] = useState(0);

  // localStorage에서 선택된 비디오 불러오기
  useEffect(() => {
    const storedVideo = localStorage.getItem('selected_video');
    if (storedVideo) {
      try {
        const video = JSON.parse(storedVideo) as SelectedVideo;
        setSelectedVideo(video);
      } catch (error) {
        console.error('선택된 비디오 파싱 오류:', error);
      }
    }
  }, []);

  // 참조 비디오 URL (선택된 비디오가 있을 때만 사용)
  const referenceVideoUrl = selectedVideo 
    ? `http://127.0.0.1:8000${selectedVideo.video_url}`
    : null;

  // 카메라 초기화 함수
  const initCamera = useCallback(async () => {
    setIsCameraLoading(true);
    setCameraError(null);

    try {
      // 브라우저가 getUserMedia를 지원하는지 확인
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError('not_supported');
        setIsCameraLoading(false);
        return;
      }

      // 카메라 스트림 요청 (오디오 포함)
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: true // 녹화 시 오디오도 함께 녹음
      });

      setStream(mediaStream);

      // 비디오 엘리먼트에 스트림 연결
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }

      setCameraError(null);
    } catch (error) {
      console.error('카메라 접근 오류:', error);
      
      if (error instanceof DOMException) {
        switch (error.name) {
          case 'NotAllowedError':
          case 'PermissionDeniedError':
            setCameraError('permission_denied');
            break;
          case 'NotFoundError':
          case 'DevicesNotFoundError':
            setCameraError('not_found');
            break;
          case 'NotSupportedError':
            setCameraError('not_supported');
            break;
          default:
            setCameraError('unknown');
        }
      } else {
        setCameraError('unknown');
      }
    } finally {
      setIsCameraLoading(false);
    }
  }, []);

  // 컴포넌트 마운트 시 카메라 초기화
  useEffect(() => {
    initCamera();

    // 컴포넌트 언마운트 시 스트림 정리
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [initCamera]);

  // 스트림이 변경되면 비디오 엘리먼트에 연결
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  // 녹화 데이터 수집 핸들러
  const handleDataAvailable = useCallback((event: BlobEvent) => {
    if (event.data.size > 0) {
      setRecordedChunks((prev) => [...prev, event.data]);
    }
  }, []);

  // 녹화 시작
  const handleStartRecording = useCallback(() => {
    if (!stream) return;

    setRecordedChunks([]);
    setUploadResult(null);
    setIsRecording(true);
    
    // 녹화 시간 트래킹 시작
    setRecordingStartTime(Date.now());
    setRecordingElapsedTime(0);

    try {
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9,opus'
      });

      mediaRecorder.ondataavailable = handleDataAvailable;
      mediaRecorder.onstop = () => {
        console.log('녹화 완료, 데이터 청크 수:', recordedChunks.length);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(1000); // 1초마다 데이터 청크 생성
      console.log('녹화 시작!');
    } catch (error) {
      console.error('MediaRecorder 생성 실패:', error);
      setIsRecording(false);
      setRecordingStartTime(null);
    }
  }, [stream, handleDataAvailable, recordedChunks.length]);

  // 녹화 종료
  const handleStopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingStartTime(null);
      setRecordingElapsedTime(0);
      console.log('녹화 종료!');
    }
  }, []);
  
  // 녹화 중 경과 시간 업데이트 (자막 싱크용)
  useEffect(() => {
    if (!isRecording || recordingStartTime === null) return;
    
    const interval = setInterval(() => {
      const elapsed = (Date.now() - recordingStartTime) / 1000; // 초 단위
      setRecordingElapsedTime(elapsed);
    }, 50); // 50ms마다 업데이트 (부드러운 자막 전환)
    
    return () => clearInterval(interval);
  }, [isRecording, recordingStartTime]);

  // 서버로 업로드
  const handleUpload = useCallback(async () => {
    if (recordedChunks.length === 0) {
      setUploadResult({ success: false, message: '녹화된 영상이 없습니다.' });
      return;
    }

    setIsUploading(true);
    setUploadResult(null);
    setAnalysisResult(null);

    try {
      // Blob 생성
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      
      // FormData 생성
      const formData = new FormData();
      formData.append('file', blob, 'my_acting.webm');
      
      // actor_id 추가 (선택된 비디오의 video_url에서 파일명 추출)
      const videoUrl = selectedVideo?.video_url || '/assets/videos/어이가없네.mp4';
      const filename = videoUrl.split('/').pop() || '어이가없네.mp4';
      const actorId = filename
        .replace(/\.[^/.]+$/, '')
        .replace(/\s+/g, '_')
        .replace(/[^\w가-힣]/g, '_')
        .replace(/_+/g, '_')
        .replace(/^_|_$/g, '');
      
      formData.append('actor_id', actorId);

      // 백엔드로 전송
      const response = await fetch('http://127.0.0.1:8000/analyze/acting', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
      }

      const data: AnalysisResult = await response.json();
      console.log('분석 완료:', data);
      
      // 전체 분석 결과 저장
      setAnalysisResult(data);
      setUploadResult({ 
        success: true, 
        message: `분석 완료! 종합 점수: ${data.total_score.toFixed(1)}점 (${data.grade} 등급)` 
      });
    } catch (error) {
      console.error('업로드 실패:', error);
      setUploadResult({ 
        success: false, 
        message: `업로드 실패: ${error instanceof Error ? error.message : '서버 연결을 확인해주세요.'}` 
      });
    } finally {
      setIsUploading(false);
    }
  }, [recordedChunks, selectedVideo]);

  // 카테고리 펼침/접기 토글
  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category],
    }));
  };

  // 에러 메시지 렌더링 함수
  const renderCameraError = () => {
    const errorMessages: Record<Exclude<CameraError, null>, { title: string; message: string }> = {
      permission_denied: {
        title: '카메라 권한 거부됨',
        message: '카메라 사용을 허용해주세요. 브라우저 설정에서 권한을 변경할 수 있습니다.'
      },
      not_found: {
        title: '카메라를 찾을 수 없음',
        message: '연결된 카메라가 없습니다. 카메라를 연결하고 다시 시도해주세요.'
      },
      not_supported: {
        title: '지원되지 않음',
        message: '이 브라우저는 카메라 기능을 지원하지 않습니다.'
      },
      unknown: {
        title: '알 수 없는 오류',
        message: '카메라를 시작하는 중 오류가 발생했습니다.'
      }
    };

    if (!cameraError) return null;

    const { title, message } = errorMessages[cameraError];

    return (
      <div className="flex flex-col items-center justify-center h-full bg-gray-800 rounded-lg p-8">
        <div className="text-red-500 mb-4">
          <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M3 3l18 18" />
          </svg>
        </div>
        <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
        <p className="text-gray-400 text-center mb-4">{message}</p>
        <button
          onClick={initCamera}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
        >
          다시 시도
        </button>
      </div>
    );
  };

  // 등급에 따른 색상 반환
  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'S': return 'text-yellow-400';
      case 'A': return 'text-green-400';
      case 'B': return 'text-blue-400';
      case 'C': return 'text-orange-400';
      case 'D': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  // 점수에 따른 프로그레스 바 색상 반환
  const getScoreColor = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-blue-500';
    if (score >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  // 점수에 따른 그라데이션 반환
  const getScoreGradient = (score: number) => {
    if (score >= 80) return 'from-green-500 to-emerald-400';
    if (score >= 60) return 'from-blue-500 to-cyan-400';
    if (score >= 40) return 'from-yellow-500 to-orange-400';
    return 'from-red-500 to-pink-400';
  };

  // 서브메트릭 렌더링
  const renderSubMetrics = (subMetrics: SubMetric[]) => {
    if (!subMetrics || subMetrics.length === 0) return null;

    return (
      <div className="mt-4 space-y-3">
        {subMetrics.map((sm, index) => (
          <div key={index} className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
            {/* 서브메트릭 헤더 */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-lg">{SUB_METRIC_ICONS[sm.name] || '📈'}</span>
                <span className="text-white font-medium">
                  {SUB_METRIC_NAMES[sm.name] || sm.name}
                </span>
                <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
                  가중치: {(sm.weight * 100).toFixed(0)}%
                </span>
              </div>
              <span className={`text-xl font-bold ${
                sm.score >= 80 ? 'text-green-400' : 
                sm.score >= 60 ? 'text-blue-400' : 
                sm.score >= 40 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {sm.score.toFixed(1)}
              </span>
            </div>

            {/* 프로그레스 바 */}
            <div className="w-full bg-gray-700 rounded-full h-1.5 mb-2">
              <div 
                className={`bg-gradient-to-r ${getScoreGradient(sm.score)} h-1.5 rounded-full transition-all duration-700`}
                style={{ width: `${sm.score}%` }}
              />
            </div>

            {/* 피드백 */}
            <p className="text-sm text-gray-400">{sm.feedback}</p>

            {/* 상세 정보 (details) */}
            {sm.details && Object.keys(sm.details).length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-700/50">
                <div className="flex flex-wrap gap-2">
                  {Object.entries(sm.details).map(([key, value]) => {
                    // description이나 method 같은 문자열은 표시하지 않음
                    if (typeof value === 'string' && (key === 'description' || key === 'method')) {
                      return null;
                    }
                    // 배열은 건너뜀
                    if (Array.isArray(value)) {
                      return null;
                    }
                    return (
                      <span 
                        key={key}
                        className="text-xs bg-gray-800 text-gray-400 px-2 py-1 rounded"
                      >
                        {key}: {typeof value === 'number' ? value.toFixed(2) : String(value)}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  // 카테고리 카드 렌더링
  const renderCategoryCard = (
    category: 'pitch' | 'energy' | 'expression',
    title: string,
    icon: React.ReactNode,
    iconBgColor: string,
    iconColor: string,
    detail: ScoreDetail
  ) => {
    const isExpanded = expandedCategories[category];

    return (
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        {/* 카드 헤더 (클릭 가능) */}
        <div 
          className="p-6 cursor-pointer hover:bg-gray-750 transition-colors"
          onClick={() => toggleCategory(category)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-12 h-12 rounded-full ${iconBgColor} flex items-center justify-center`}>
                <div className={iconColor}>{icon}</div>
              </div>
              <div>
                <h3 className="text-white font-semibold text-lg">{title}</h3>
                <p className="text-xs text-gray-400">
                  가중치: {(detail.weight * 100).toFixed(0)}% | 
                  서브메트릭: {detail.sub_metrics?.length || 0}개
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* 점수 */}
              <div className="text-right">
                <span className={`text-4xl font-bold ${
                  detail.score >= 80 ? 'text-green-400' : 
                  detail.score >= 60 ? 'text-blue-400' : 
                  detail.score >= 40 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {detail.score.toFixed(1)}
                </span>
                <span className="text-gray-400 text-lg">/100</span>
              </div>
              
              {/* 펼침/접기 아이콘 */}
              <svg 
                className={`w-5 h-5 text-gray-400 transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>

          {/* 전체 프로그레스 바 */}
          <div className="mt-4">
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className={`bg-gradient-to-r ${getScoreGradient(detail.score)} h-2 rounded-full transition-all duration-700`}
                style={{ width: `${detail.score}%` }}
              />
            </div>
          </div>

          {/* 요약 피드백 */}
          <p className="mt-3 text-gray-300 text-sm leading-relaxed">
            {detail.feedback}
          </p>
        </div>

        {/* 서브메트릭 (펼침 시) */}
        {isExpanded && detail.sub_metrics && detail.sub_metrics.length > 0 && (
          <div className="px-6 pb-6 border-t border-gray-700">
            <div className="pt-4">
              <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                세부 분석
              </h4>
              {renderSubMetrics(detail.sub_metrics)}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen w-screen bg-gray-900 overflow-y-auto">
      {/* 상단: 비디오 영역 */}
      <div className="flex h-[70vh] w-full">
        {/* 왼쪽: 웹캠 피드 */}
        <div className="w-1/2 h-full p-4 flex flex-col">
          <h2 className="text-xl font-bold text-white mb-4 text-center">
            내 모습 (웹캠)
          </h2>
          <div className="flex-1 relative bg-black rounded-lg overflow-hidden">
            {isCameraLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="flex flex-col items-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mb-4"></div>
                  <p className="text-gray-400">카메라 연결 중...</p>
                </div>
              </div>
            ) : cameraError ? (
              renderCameraError()
            ) : (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
                style={{ transform: 'scaleX(-1)' }}
              />
            )}
            
            {/* 자막 오버레이 (녹화 중일 때만 웹캠 위에 표시) */}
            {isRecording && selectedVideo?.sentences && selectedVideo.sentences.length > 0 && (
              <SubtitleOverlay 
                sentences={selectedVideo.sentences} 
                currentTime={recordingElapsedTime}
              />
            )}
            
            {/* 상태 표시 (LIVE / REC) */}
            {!cameraError && !isCameraLoading && (
              <div className={`absolute top-4 left-4 flex items-center gap-2 px-3 py-1 rounded-full ${
                isRecording ? 'bg-red-600' : 'bg-green-600'
              }`}>
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-white">
                  {isRecording ? 'REC' : 'LIVE'}
                </span>
              </div>
            )}
          </div>

          {/* 녹화 컨트롤 버튼 */}
          <div className="mt-4 flex flex-col gap-3">
            <div className="flex gap-3 justify-center">
              {isRecording ? (
                <button
                  onClick={handleStopRecording}
                  className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors shadow-lg"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="1" />
                  </svg>
                  녹화 종료
                </button>
              ) : (
                <button
                  onClick={handleStartRecording}
                  disabled={!stream || isCameraLoading || !!cameraError}
                  className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors shadow-lg ${
                    !stream || isCameraLoading || cameraError
                      ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                      : 'bg-red-600 hover:bg-red-700 text-white'
                  }`}
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="8" />
                  </svg>
                  녹화 시작
                </button>
              )}

              {/* 업로드 버튼 (녹화 완료 후 표시) */}
              {!isRecording && recordedChunks.length > 0 && (
                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors shadow-lg ${
                    isUploading
                      ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                >
                  {isUploading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      분석 중...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      서버로 전송
                    </>
                  )}
                </button>
              )}
            </div>

            {/* 결과 메시지 */}
            
          </div>
        </div>

        {/* 구분선 */}
        <div className="w-px bg-gray-700"></div>

        {/* 오른쪽: 참조 비디오 */}
        <div className="w-1/2 h-full p-4 flex flex-col">
          <h2 className="text-xl font-bold text-white mb-4 text-center">
            참조 영상
          </h2>
          <div className="flex-1 relative bg-black rounded-lg overflow-hidden flex items-center justify-center">
            {referenceVideoUrl ? (
              <video
                ref={referenceVideoRef}
                src={referenceVideoUrl}
                controls
                className="w-full h-full object-contain"
                controlsList="nodownload"
                onTimeUpdate={(e) => setVideoCurrentTime(e.currentTarget.currentTime)}
              >
                <source src={referenceVideoUrl} type="video/mp4" />
                브라우저가 비디오 재생을 지원하지 않습니다.
              </video>
            ) : (
              <div className="flex flex-col items-center justify-center text-gray-400">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p className="text-lg font-medium">선택된 영상이 없습니다</p>
                <p className="text-sm mt-2">배우 추천 페이지에서 영상을 선택해주세요</p>
              </div>
            )}
          </div>
          
          {/* 비디오 정보 */}
          <div className="mt-4 p-3 bg-gray-800 rounded-lg">
            <p className="text-gray-300 text-sm">
              <span className="font-medium text-white">현재 영상:</span> {selectedVideo?.title || '선택된 영상 없음'}
            </p>
            
            {/* 전체 대사 표시 */}
            {selectedVideo?.script && (
              <div className="mt-3 p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                <p className="text-gray-400 text-xs mb-1 uppercase tracking-wider">전체 대사</p>
                <p className="text-white text-sm leading-relaxed">
                  "{selectedVideo.script}"
                </p>
              </div>
            )}
            
            <p className="text-gray-500 text-xs mt-2">
              {selectedVideo 
                ? '▶ 영상을 재생하면 왼쪽 웹캠 화면에 자막이 표시됩니다' 
                : '배우 추천 페이지에서 영상을 선택해주세요'}
            </p>
          </div>
        </div>
      </div>

      {/* 하단: Ultra-Precision 분석 결과 영역 */}
      {analysisResult && (
        <div className="w-full p-6 border-t border-gray-700">
          {/* 섹션 타이틀 */}
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-white mb-2">
              Ultra-Precision 연기 분석
            </h2>
            <p className="text-gray-400">
              AI가 당신의 연기를 세밀하게 분석했습니다
            </p>
          </div>

          {/* 종합 점수 카드 */}
          <div className="max-w-5xl mx-auto mb-10">
            <div className="bg-gradient-to-r from-purple-900/50 via-blue-900/50 to-cyan-900/50 rounded-2xl p-0 border border-purple-700/50 shadow-2xl">
              <div className="flex items-center justify-center gap-12">
                {/* 등급 */}
                <div className="text-center">
                  <p className="text-gray-400 text-sm mb-2 uppercase tracking-wider">등급</p>
                  <div className="relative">
                    <span className={`text-8xl font-black ${getGradeColor(analysisResult.grade)} drop-shadow-lg`}>
                      {analysisResult.grade}
                    </span>
                    {analysisResult.grade === 'S' && (
                      <div className="absolute -top-2 -right-2 text-2xl animate-bounce">✨</div>
                    )}
                  </div>
                </div>
                
                <div className="w-px h-28 bg-gray-600"></div>
                
                {/* 종합 점수 */}
                <div className="text-center">
                  <p className="text-gray-400 text-sm mb-2 uppercase tracking-wider translate-y-[-15px]">종합 점수</p>
                  <div className="flex items-baseline justify-center">
                    <span className="text-7xl font-bold text-white">
                      {analysisResult.total_score.toFixed(1)}
                    </span>
                    <span className="text-2xl text-gray-400 ml-1">/100</span>
                  </div>
                </div>

                <div className="w-px h-28 bg-gray-600"></div>

                {/* 레이더 차트 */}
                <div className="text-center translate-y-[30px]" >
                  <RadarChart 
                    pitch={analysisResult.details.pitch.score}
                    energy={analysisResult.details.energy.score}
                    expression={analysisResult.details.expression.score}
                    size={260}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* 세부 점수 카드들 (Ultra-Precision) */}
          <div className="max-w-5xl mx-auto space-y-6 mb-10">
            {/* 억양/피치 */}
            {renderCategoryCard(
              'pitch',
              '억양 / 피치 분석',
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>,
              'bg-pink-600/30',
              'text-pink-400',
              analysisResult.details.pitch
            )}

            {/* 볼륨/에너지 */}
            {renderCategoryCard(
              'energy',
              '볼륨 / 에너지 분석',
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
              </svg>,
              'bg-orange-600/30',
              'text-orange-400',
              analysisResult.details.energy
            )}

            {/* 표정 */}
            {renderCategoryCard(
              'expression',
              '표정 분석 (얼굴 영역별)',
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>,
              'bg-cyan-600/30',
              'text-cyan-400',
              analysisResult.details.expression
            )}
          </div>

          {/* 종합 피드백 */}
          {analysisResult.overall_feedback && (
            <div className="max-w-5xl mx-auto mb-8">
              <div className="bg-gradient-to-r from-green-900/30 to-emerald-900/30 rounded-xl p-6 border border-green-700/50">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-full bg-green-600/30 flex items-center justify-center flex-shrink-0">
                    <svg className="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-white font-semibold text-lg mb-2">AI 코칭 피드백</h3>
                    <p className="text-gray-300 leading-relaxed text-lg">
                      {analysisResult.overall_feedback}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 분석 정보 푸터 */}
          <div className="max-w-5xl mx-auto">
            <div className="flex justify-center gap-6 text-xs text-gray-500 border-t border-gray-800 pt-4">
              <span className="flex items-center gap-1">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                레퍼런스: {analysisResult.actor_id}
              </span>
              <span>|</span>
              <span className="flex items-center gap-1">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                사용자: {analysisResult.user_id}
              </span>
              <span>|</span>
              <span className="flex items-center gap-1">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Ultra-Precision Analysis v2.0
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 분석 결과가 없을 때 안내 메시지 */}
      {!analysisResult && !isUploading && (
        <div className="w-full p-8 border-t border-gray-700">
          <div className="max-w-4xl mx-auto text-center">
            <div className="text-gray-600 mb-4">
              <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} 
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-gray-400 text-lg mb-2">Ultra-Precision 분석 결과가 여기에 표시됩니다</h3>
            <p className="text-gray-500 text-sm mb-4">
              참조 영상을 보면서 따라한 뒤, 녹화하고 서버로 전송하면 
              <span className="text-purple-400 font-medium"> 억양, 볼륨, 표정</span>에 대한 
              세밀한 분석 결과를 확인할 수 있습니다.
            </p>
            <div className="flex justify-center gap-6 text-xs text-gray-600">
              <span>🎵 패턴 매칭</span>
              <span>📊 다이내믹 레인지</span>
              <span>👁️ 눈 표현</span>
              <span>👄 입 표현</span>
              <span>🤨 눈썹 표현</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
