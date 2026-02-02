'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';

// 카메라 에러 타입 정의
type CameraError = 'permission_denied' | 'not_found' | 'not_supported' | 'unknown' | null;

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

  // 참조 비디오 URL (서버 assets 폴더에서 제공)
  const referenceVideoUrl = 'http://127.0.0.1:8000/assets/어이가 없네.mp4';

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

    try {
      // Blob 생성
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      
      // FormData 생성
      const formData = new FormData();
      formData.append('file', blob, 'my_acting.webm');
      
      // 타겟 영상 파일명 추가 (referenceVideoUrl에서 파일명 추출)
      const targetFilename = referenceVideoUrl.split('/').pop() || '어이가 없네.mp4';
      formData.append('target_filename', targetFilename);

      // 백엔드로 전송
      const response = await fetch('http://127.0.0.1:8000/analyze/acting', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('업로드 성공:', data);
      setUploadResult({ 
        success: true, 
        message: `분석 완료! 점수: ${data.score}점, 싱크로율: ${data.sync_rate}%` 
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

  return (
    <div className="flex h-screen w-screen bg-gray-900">
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
            <span className="font-medium text-white">현재 영상:</span> 어이가 없네.mp4
          </p>
          <p className="text-gray-500 text-xs mt-1">
            영상을 보면서 따라해보세요!
          </p>
        </div>
      </div>
    </div>
  );
}