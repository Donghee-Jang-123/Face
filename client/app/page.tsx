'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';

// 카메라 에러 타입 정의
type CameraError = 'permission_denied' | 'not_found' | 'not_supported' | 'unknown' | null;

// 인증 모드 타입
type AuthMode = 'idle' | 'login' | 'signup';

// API 응답 타입
interface AuthResponse {
  success: boolean;
  message: string;
  nickname?: string;
  score?: number;
  recommended_actor_id?: string;
  recommended_actor_score?: number;
}

export default function Home() {
  const router = useRouter();
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const faceMeshCanvasRef = useRef<HTMLCanvasElement>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // 상태 관리
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<CameraError>(null);
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  
  // 인증 관련 상태
  const [authMode, setAuthMode] = useState<AuthMode>('idle');
  const [nickname, setNickname] = useState('');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // 카메라 초기화 함수
  const initCamera = useCallback(async () => {
    setIsCameraLoading(true);
    setCameraError(null);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError('not_supported');
        setIsCameraLoading(false);
        return;
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      setStream(mediaStream);

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

  // 카메라 종료 함수
  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [stream]);

  // 스트림이 변경되면 비디오 엘리먼트에 연결 (모드 전환 시 video 엘리먼트가 바뀌므로 authMode 포함)
  useEffect(() => {
    if (videoRef.current && stream && !capturedImage) {
      videoRef.current.srcObject = stream;
      // ?? ?????? ??? ? ??
      videoRef.current.play().catch(() => {});
    }
  }, [stream, authMode]);

  // 메인 화면(idle)에서 카메라 자동 시작 (배경 Face Mesh용)
  useEffect(() => {
    if (authMode === 'idle' && !stream) {
      initCamera();
    }
  }, [authMode, stream, initCamera]);

  // 컴포넌트 언마운트 시 카메라 정리
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  // MediaPipe Face Landmarker 초기화 (메인 화면 배경용 - idle 모드에서만)
  useEffect(() => {
    if (authMode !== 'idle' || !stream) return;

    let cancelled = false;
    const initFaceLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        if (cancelled) return;

        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
        });

        if (cancelled) return;
        faceLandmarkerRef.current = faceLandmarker;
      } catch (err) {
        console.error('Face Landmarker 초기화 실패:', err);
      }
    };

    initFaceLandmarker();
    return () => {
      cancelled = true;
      faceLandmarkerRef.current = null;
    };
  }, [authMode, stream]);

  // Face Mesh 점 그리기 루프 (메인 화면 배경 - idle 모드에서만)
  useEffect(() => {
    if (authMode !== 'idle') return;

    const video = videoRef.current;
    const canvas = faceMeshCanvasRef.current;
    if (!video || !canvas || !stream) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastVideoTime = -1;

    const drawLandmarks = () => {
      if (!video.videoWidth || !video.videoHeight) {
        animationFrameRef.current = requestAnimationFrame(drawLandmarks);
        return;
      }

      // 캔버스 크기 설정 (비디오와 동일)
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const faceLandmarker = faceLandmarkerRef.current;
      if (faceLandmarker) {
        if (video.currentTime !== lastVideoTime) {
          try {
            const result = faceLandmarker.detectForVideo(video, performance.now());

            // 배경 그라데이션 (얼굴 안 보임)
            const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
            gradient.addColorStop(0, '#1e1b4b');   // indigo-950
            gradient.addColorStop(0.5, '#111827'); // gray-900
            gradient.addColorStop(1, '#3b0764');   // purple-950
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            if (result.faceLandmarks && result.faceLandmarks.length > 0) {
              const landmarks = result.faceLandmarks[0];
              ctx.fillStyle = '#60a5fa';

              landmarks.forEach((lm) => {
                // 거울 모드: x 좌표 반전
                const x = (1 - lm.x) * canvas.width;
                const y = lm.y * canvas.height;

                ctx.beginPath();
                ctx.arc(x, y, 2, 0, Math.PI * 2);
                ctx.fill();
              });
            }
          } catch (e) {
            // 프레임 처리 실패 시 무시
          }
          lastVideoTime = video.currentTime;
        }
      } else {
        // Face Landmarker 로딩 중: 그라데이션 배경
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, '#1e1b4b');   // indigo-950
        gradient.addColorStop(0.5, '#111827');
        gradient.addColorStop(1, '#3b0764');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      animationFrameRef.current = requestAnimationFrame(drawLandmarks);
    };

    drawLandmarks();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [authMode, stream]);

  // 인증 모드 변경 핸들러
  const handleModeChange = useCallback((mode: AuthMode) => {
    setAuthMode(mode);
    setCapturedImage(null);
    setErrorMessage(null);
    setSuccessMessage(null);
    setNickname('');
    
    // 카메라가 없을 때만 초기화 (모드 전환 시 기존 스트림 유지)
    if (!stream) {
      initCamera();
    }
  }, [initCamera, stream]);

  // 사진 촬영 함수
  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // 거울 모드로 캡처
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
    setCapturedImage(imageDataUrl);
    setErrorMessage(null);
  }, []);

  // 다시 촬영 함수
  const retakePhoto = useCallback(() => {
    setCapturedImage(null);
    setErrorMessage(null);
  }, []);

  // Data URL을 Blob으로 변환
  const dataURLtoBlob = (dataURL: string): Blob => {
    const arr = dataURL.split(',');
    const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  };

  // 회원가입 제출
  const handleSignup = useCallback(async () => {
    if (!capturedImage || !nickname.trim()) {
      setErrorMessage('닉네임과 사진을 모두 입력해주세요.');
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append('nickname', nickname.trim());
      formData.append('file', dataURLtoBlob(capturedImage), 'face.jpg');

      const response = await fetch('http://127.0.0.1:8000/register', {
        method: 'POST',
        body: formData,
      });

      const data: AuthResponse = await response.json();

      if (data.success) {
        setSuccessMessage(data.message);
        // 잠시 후 로그인 모드로 전환
        setTimeout(() => {
          handleModeChange('login');
          setSuccessMessage('회원가입이 완료되었습니다! 얼굴을 인식하여 로그인해주세요.');
        }, 1500);
      } else {
        setErrorMessage(data.message);
      }
    } catch (error) {
      console.error('회원가입 오류:', error);
      setErrorMessage('서버 연결에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setIsSubmitting(false);
    }
  }, [capturedImage, nickname, handleModeChange]);

  // 로그인 제출
  const handleLogin = useCallback(async () => {
    if (!capturedImage) {
      setErrorMessage('얼굴 사진을 먼저 촬영해주세요.');
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append('file', dataURLtoBlob(capturedImage), 'face.jpg');

      const response = await fetch('http://127.0.0.1:8000/login', {
        method: 'POST',
        body: formData,
      });

      const data: AuthResponse = await response.json();

      if (data.success && data.nickname) {
        setSuccessMessage(data.message);
        
        // 토큰 및 사용자 정보 저장
        localStorage.setItem('user', JSON.stringify({
          nickname: data.nickname,
          recommended_actor_id: data.recommended_actor_id,
          recommended_actor_score: data.recommended_actor_score,
          loginTime: new Date().toISOString(),
        }));

        // 잠시 후 acting 페이지로 이동
        setTimeout(() => {
          stopCamera();
          router.push('/recommend');
        }, 1500);
      } else {
        setErrorMessage(data.message);
        setCapturedImage(null); // 실패 시 다시 촬영할 수 있게
      }
    } catch (error) {
      console.error('로그인 오류:', error);
      setErrorMessage('서버 연결에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setIsSubmitting(false);
    }
  }, [capturedImage, router, stopCamera]);

  // 카메라 에러 메시지 렌더링
  const renderCameraError = () => {
    const errorMessages: Record<Exclude<CameraError, null>, { title: string; message: string }> = {
      permission_denied: {
        title: '카메라 권한 거부됨',
        message: '카메라 사용을 허용해주세요.'
      },
      not_found: {
        title: '카메라를 찾을 수 없음',
        message: '연결된 카메라가 없습니다.'
      },
      not_supported: {
        title: '지원되지 않음',
        message: '이 브라우저는 카메라를 지원하지 않습니다.'
      },
      unknown: {
        title: '알 수 없는 오류',
        message: '카메라 시작 중 오류가 발생했습니다.'
      }
    };

    if (!cameraError) return null;

    const { title, message } = errorMessages[cameraError];

    return (
      <div className="flex flex-col items-center justify-center h-full bg-gray-800 rounded-xl p-6">
        <div className="text-red-400 mb-3">
          <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h3 className="text-lg font-bold text-white mb-1">{title}</h3>
        <p className="text-gray-400 text-sm text-center mb-3">{message}</p>
        <button
          onClick={initCamera}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
        >
          다시 시도
        </button>
      </div>
    );
  };

  // 메인 화면 (idle 모드) - 배경에 Face Mesh 점
  if (authMode === 'idle') {
    return (
      <main className="relative flex min-h-screen flex-col items-center justify-center p-6 overflow-hidden">
        {/* 히든 캔버스 (캡처용) */}
        <canvas ref={canvasRef} className="hidden" />

        {/* 배경: Face Mesh 점 (얼굴 안 보이고 점만) */}
        <div className="fixed inset-0 z-0">
          {!cameraError && stream ? (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-cover invisible"
                style={{ transform: 'scaleX(-1)' }}
                aria-hidden
              />
              <canvas
                ref={faceMeshCanvasRef}
                className="absolute inset-0 w-full h-full object-cover"
                aria-label="얼굴 랜드마크"
              />
            </>
          ) : (
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-900 via-gray-900 to-purple-950" />
          )}
        </div>
        
        {/* 전경 콘텐츠 */}
        <div className="relative z-10 flex flex-col items-center">
        {/* 로고/타이틀 영역 */}
        <div className="text-center mb-12">
          
          <h1 className="text-[120px] font-bold text-white">
            FaceAct
          </h1>
          <p className="text-[25px] text-gray-400 max-w-2xl">
            얼굴 인식 기반의 실시간 연기 분석 서비스
          </p>
        </div>

        {/* 버튼 영역 */}
        <div className="flex flex-col sm:flex-row gap-4 w-full max-w-md">
          <button
            onClick={() => handleModeChange('login')}
            className="flex-1 py-4 px-8 bg-blue-600 hover:bg-blue-700 text-white text-lg font-semibold rounded-xl transition-all duration-200 transform hover:scale-[1.02] shadow-lg hover:shadow-blue-500/25"
          >
            로그인
          </button>
          <button
            onClick={() => handleModeChange('signup')}
            className="flex-1 py-4 px-8 bg-gray-700 hover:bg-gray-600 text-white text-lg font-semibold rounded-xl transition-all duration-200 transform hover:scale-[1.02] shadow-lg border border-gray-600"
          >
            회원가입
          </button>
        </div>

        {/* 하단 설명 */}
        <div className="mt-12 text-center text-gray-500 text-sm max-w-2xl">
          <p>회원가입하고, 페이스ID로 간편하게 로그인하세요.</p>
        </div>
        </div>
      </main>
    );
  }

  // 로그인/회원가입 모드
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-indigo-950 via-gray-900 to-purple-950 p-4 md:p-6">
      {/* 히든 캔버스 */}
      <canvas ref={canvasRef} className="hidden" />
      
      {/* 카드 컨테이너 */}
      <div className="w-full max-w-lg bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl border border-gray-700/50 overflow-hidden">
        {/* 헤더 */}
        <div className="px-6 py-5 border-b border-gray-700/50">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white">
                {authMode === 'login' ? '로그인' : '회원가입'}
              </h2>
              <p className="text-gray-400 text-sm mt-1">
                {authMode === 'login' 
                  ? '얼굴을 인식하여 로그인합니다' 
                  : '얼굴과 닉네임을 등록합니다'}
              </p>
            </div>
            <button
              onClick={() => handleModeChange('idle')}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* 컨텐츠 */}
        <div className="p-6">
          {/* 회원가입 시 닉네임 입력 */}
          {authMode === 'signup' && (
            <div className="mb-5">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                닉네임
              </label>
              <input
                type="text"
                value={nickname}
                onChange={(e) => setNickname(e.target.value)}
                placeholder="사용할 닉네임을 입력하세요"
                className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                maxLength={20}
              />
            </div>
          )}

          {/* 웹캠/캡처 이미지 영역 */}
          <div className="mb-5">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              얼굴 인식
            </label>
            <div className="relative aspect-[4/3] bg-gray-900 rounded-xl overflow-hidden">
              {isCameraLoading ? (
                <div className="flex items-center justify-center h-full">
                  <div className="flex flex-col items-center">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500 mb-3"></div>
                    <p className="text-gray-400 text-sm">카메라 연결 중...</p>
                  </div>
                </div>
              ) : cameraError ? (
                renderCameraError()
              ) : capturedImage ? (
                // 캡처된 이미지 표시
                <img 
                  src={capturedImage} 
                  alt="Captured face" 
                  className="w-full h-full object-cover"
                />
              ) : (
                // 실시간 비디오 피드
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                  style={{ transform: 'scaleX(-1)' }}
                />
              )}

              {/* 가이드 오버레이 */}
              {!capturedImage && !cameraError && !isCameraLoading && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-48 h-48 md:w-56 md:h-56 border-2 border-dashed border-white/30 rounded-full"></div>
                </div>
              )}

              {/* 캡처 상태 표시 */}
              {capturedImage && (
                <div className="absolute top-3 left-3 flex items-center gap-2 px-3 py-1.5 bg-green-600 rounded-full">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-xs font-medium text-white">촬영 완료</span>
                </div>
              )}
            </div>
          </div>

          {/* 촬영 버튼 */}
          <div className="flex gap-3 mb-5">
            {capturedImage ? (
              <button
                onClick={retakePhoto}
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-gray-600 hover:bg-gray-500 text-white rounded-xl font-medium transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                다시 촬영
              </button>
            ) : (
              <button
                onClick={capturePhoto}
                disabled={isCameraLoading || !!cameraError}
                className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-medium transition-colors ${
                  isCameraLoading || cameraError
                    ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                    : 'bg-purple-600 hover:bg-purple-700 text-white'
                }`}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                사진 촬영
              </button>
            )}
          </div>

          {/* 에러/성공 메시지 */}
          {errorMessage && (
            <div className="mb-5 p-4 bg-red-900/30 border border-red-700/50 rounded-xl">
              <div className="flex items-center gap-2 text-red-400">
                <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-sm">{errorMessage}</span>
              </div>
            </div>
          )}

          {successMessage && (
            <div className="mb-5 p-4 bg-green-900/30 border border-green-700/50 rounded-xl">
              <div className="flex items-center gap-2 text-green-400">
                <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-sm">{successMessage}</span>
              </div>
            </div>
          )}

          {/* 제출 버튼 */}
          <button
            onClick={authMode === 'login' ? handleLogin : handleSignup}
            disabled={isSubmitting || !capturedImage || (authMode === 'signup' && !nickname.trim())}
            className={`w-full py-4 rounded-xl font-semibold text-lg transition-all duration-200 ${
              isSubmitting || !capturedImage || (authMode === 'signup' && !nickname.trim())
                ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                : 'bg-blue-600 hover:bg-blue-700 text-white transform hover:scale-[1.01] shadow-lg'
            }`}
          >
            {isSubmitting ? (
              <span className="flex items-center justify-center gap-2">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                처리 중...
              </span>
            ) : (
              authMode === 'login' ? '로그인' : '회원가입'
            )}
          </button>

          {/* 모드 전환 링크 */}
          <div className="mt-5 text-center">
            {authMode === 'login' ? (
              <p className="text-gray-400 text-sm">
                계정이 없으신가요?{' '}
                <button
                  onClick={() => handleModeChange('signup')}
                  className="text-blue-400 hover:text-blue-300 font-medium transition-colors"
                >
                  회원가입
                </button>
              </p>
            ) : (
              <p className="text-gray-400 text-sm">
                이미 계정이 있으신가요?{' '}
                <button
                  onClick={() => handleModeChange('login')}
                  className="text-blue-400 hover:text-blue-300 font-medium transition-colors"
                >
                  로그인
                </button>
              </p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
