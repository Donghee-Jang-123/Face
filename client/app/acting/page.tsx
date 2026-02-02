'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';

// 카메라 에러 타입 정의
type CameraError = 'permission_denied' | 'not_found' | 'not_supported' | 'unknown' | null;

// 분석 결과 타입 정의
interface ScoreDetail {
  score: number;
  feedback: string;
  weight: number;
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
  
  // 분석 결과 상태
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  // 참조 비디오 URL (서버 assets 폴더에서 제공)
  const referenceVideoUrl = 'http://127.0.0.1:8000/assets/어이가없네.mp4';

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
    }
  }, [stream, handleDataAvailable, recordedChunks.length]);

  // 녹화 종료
  const handleStopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      console.log('녹화 종료!');
    }
  }, []);

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
      
      // actor_id 추가 (파일명에서 확장자 제거 후 공백을 언더스코어로 변환)
      // 서버의 sanitize_actor_id 로직과 동일하게 처리
      const filename = referenceVideoUrl.split('/').pop() || '어이가없네.mp4';
      const actorId = filename
        .replace(/\.[^/.]+$/, '') // 확장자 제거
        .replace(/\s+/g, '_')     // 공백을 언더스코어로
        .replace(/[^\w가-힣]/g, '_') // 특수문자 제거
        .replace(/_+/g, '_')      // 연속 언더스코어 정리
        .replace(/^_|_$/g, '');   // 앞뒤 언더스코어 제거
      
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
  }, [recordedChunks, referenceVideoUrl]);

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
                style={{ transform: 'scaleX(-1)' }} // 거울 모드
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
            {uploadResult && (
              <div className={`p-3 rounded-lg text-sm ${
                uploadResult.success 
                  ? 'bg-green-900/50 text-green-300 border border-green-700' 
                  : 'bg-red-900/50 text-red-300 border border-red-700'
              }`}>
                {uploadResult.message}
              </div>
            )}
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
            <video
              ref={referenceVideoRef}
              src={referenceVideoUrl}
              controls
              className="w-full h-full object-contain"
              controlsList="nodownload"
            >
              <source src={referenceVideoUrl} type="video/mp4" />
              브라우저가 비디오 재생을 지원하지 않습니다.
            </video>
          </div>
          
          {/* 비디오 정보 */}
          <div className="mt-4 p-3 bg-gray-800 rounded-lg">
            <p className="text-gray-300 text-sm">
              <span className="font-medium text-white">현재 영상:</span> 어이가없네.mp4
            </p>
            <p className="text-gray-500 text-xs mt-1">
              영상을 보면서 따라해보세요!
            </p>
          </div>
        </div>
      </div>

      {/* 하단: 분석 결과 영역 */}
      {analysisResult && (
        <div className="w-full p-6 border-t border-gray-700">
          <h2 className="text-2xl font-bold text-white mb-6 text-center">
            연기 분석 결과
          </h2>

          {/* 종합 점수 카드 */}
          <div className="max-w-4xl mx-auto mb-8">
            <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-2xl p-8 border border-purple-700/50">
              <div className="flex items-center justify-center gap-8">
                {/* 등급 */}
                <div className="text-center">
                  <p className="text-gray-400 text-sm mb-2">등급</p>
                  <span className={`text-7xl font-black ${getGradeColor(analysisResult.grade)}`}>
                    {analysisResult.grade}
                  </span>
                </div>
                
                <div className="w-px h-24 bg-gray-600"></div>
                
                {/* 종합 점수 */}
                <div className="text-center">
                  <p className="text-gray-400 text-sm mb-2">종합 점수</p>
                  <span className="text-6xl font-bold text-white">
                    {analysisResult.total_score.toFixed(1)}
                  </span>
                  <span className="text-2xl text-gray-400">/100</span>
                </div>
              </div>
            </div>
          </div>

          {/* 세부 점수 카드들 */}
          <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* 억양 점수 */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-pink-600/30 flex items-center justify-center">
                  <svg className="w-5 h-5 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-white font-semibold">억양/피치</h3>
                  <p className="text-xs text-gray-400">가중치: {(analysisResult.details.pitch.weight * 100).toFixed(0)}%</p>
                </div>
              </div>
              
              <div className="mb-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-3xl font-bold text-white">{analysisResult.details.pitch.score.toFixed(1)}</span>
                  <span className="text-gray-400">/100</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className={`${getScoreColor(analysisResult.details.pitch.score)} h-2 rounded-full transition-all duration-500`}
                    style={{ width: `${analysisResult.details.pitch.score}%` }}
                  ></div>
                </div>
              </div>
              
              <p className="text-sm text-gray-300 leading-relaxed">
                {analysisResult.details.pitch.feedback || '피드백이 없습니다.'}
              </p>
            </div>

            {/* 볼륨 점수 */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-orange-600/30 flex items-center justify-center">
                  <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-white font-semibold">볼륨/에너지</h3>
                  <p className="text-xs text-gray-400">가중치: {(analysisResult.details.energy.weight * 100).toFixed(0)}%</p>
                </div>
              </div>
              
              <div className="mb-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-3xl font-bold text-white">{analysisResult.details.energy.score.toFixed(1)}</span>
                  <span className="text-gray-400">/100</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className={`${getScoreColor(analysisResult.details.energy.score)} h-2 rounded-full transition-all duration-500`}
                    style={{ width: `${analysisResult.details.energy.score}%` }}
                  ></div>
                </div>
              </div>
              
              <p className="text-sm text-gray-300 leading-relaxed">
                {analysisResult.details.energy.feedback || '피드백이 없습니다.'}
              </p>
            </div>

            {/* 표정 점수 */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-cyan-600/30 flex items-center justify-center">
                  <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-white font-semibold">표정</h3>
                  <p className="text-xs text-gray-400">가중치: {(analysisResult.details.expression.weight * 100).toFixed(0)}%</p>
                </div>
              </div>
              
              <div className="mb-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-3xl font-bold text-white">{analysisResult.details.expression.score.toFixed(1)}</span>
                  <span className="text-gray-400">/100</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className={`${getScoreColor(analysisResult.details.expression.score)} h-2 rounded-full transition-all duration-500`}
                    style={{ width: `${analysisResult.details.expression.score}%` }}
                  ></div>
                </div>
              </div>
              
              <p className="text-sm text-gray-300 leading-relaxed">
                {analysisResult.details.expression.feedback || '피드백이 없습니다.'}
              </p>
            </div>
          </div>

          {/* 종합 피드백 */}
          {analysisResult.overall_feedback && (
            <div className="max-w-4xl mx-auto">
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-green-600/30 flex items-center justify-center">
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h3 className="text-white font-semibold text-lg">종합 피드백</h3>
                </div>
                <p className="text-gray-300 leading-relaxed pl-13">
                  {analysisResult.overall_feedback}
                </p>
              </div>
            </div>
          )}

          {/* 분석 정보 */}
          <div className="max-w-4xl mx-auto mt-6">
            <div className="flex justify-center gap-4 text-xs text-gray-500">
              <span>레퍼런스: {analysisResult.actor_id}</span>
              <span>|</span>
              <span>사용자: {analysisResult.user_id}</span>
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
            <h3 className="text-gray-400 text-lg mb-2">분석 결과가 여기에 표시됩니다</h3>
            <p className="text-gray-500 text-sm">
              참조 영상을 보면서 따라한 뒤, 녹화하고 서버로 전송하면 상세한 분석 결과를 확인할 수 있습니다.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}