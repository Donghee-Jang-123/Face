'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

export default function ActingPage() {
  // 1. 상태 및 Refs 관리
  const webcamRef = useRef<Webcam>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  
  const [capturing, setCapturing] = useState(false); // 녹화 중인지?
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]); // 녹화 데이터 저장소
  const [isUploading, setIsUploading] = useState(false); // 업로드 로딩 상태
  const [uploadResult, setUploadResult] = useState<string | null>(null); // 결과 메시지

  // 2. 녹화 데이터 수집 함수 (MediaRecorder가 데이터를 뱉을 때마다 실행)
  const handleDataAvailable = useCallback(({ data }: BlobEvent) => {
    if (data.size > 0) {
      setRecordedChunks((prev) => prev.concat(data));
    }
  }, []);

  // 3. 녹화 시작 함수
  const handleStartCaptureClick = useCallback(() => {
    setCapturing(true);
    setRecordedChunks([]); // 기존 데이터 초기화
    setUploadResult(null); // 결과 메시지 초기화

    if (webcamRef.current && webcamRef.current.video && webcamRef.current.stream) {
      // MediaRecorder 생성 (MIME 타입은 브라우저 호환성에 따라 webm 사용)
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: "video/webm"
      });
      mediaRecorderRef.current.addEventListener("dataavailable", handleDataAvailable);
      mediaRecorderRef.current.start();
      console.log("🎥 녹화 시작!");
    }
  }, [webcamRef, handleDataAvailable]);

  // 4. 녹화 종료 함수 (종료되면 바로 업로드 준비)
  const handleStopCaptureClick = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    setCapturing(false);
    console.log("🛑 녹화 종료! 데이터 정리 중...");
  }, []);

  // 5. 서버로 전송 함수 (녹화가 끝나고 recordedChunks가 업데이트되면 버튼으로 실행하거나 자동 실행)
  const handleUpload = useCallback(async () => {
    if (recordedChunks.length === 0) {
      alert("녹화된 영상이 없습니다!");
      return;
    }

    setIsUploading(true);
    
    // Blob 생성 (여러 조각을 하나의 파일로 합침)
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    
    // FormData 생성 (파일을 담는 봉투)
    const formData = new FormData();
    formData.append("file", blob, "my_acting.webm"); // 백엔드에서 받을 이름: 'file'

    try {
      // 🚀 백엔드로 전송! (주소는 님의 FastAPI 주소)
      const response = await axios.post("http://127.0.0.1:8000/analyze/acting", formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log("✅ 업로드 성공:", response.data);
      setUploadResult("분석 완료! 결과: " + JSON.stringify(response.data));
      
      // 만약 백엔드에서 처리된 영상 URL을 준다면 여기서 비디오 태그에 넣으면 됨
      // const videoUrl = response.data.output_file; 

    } catch (error) {
      console.error("❌ 업로드 실패:", error);
      setUploadResult("업로드 실패 ㅠㅠ 서버를 확인해주세요.");
    } finally {
      setIsUploading(false);
    }
  }, [recordedChunks]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <h1 className="text-3xl font-bold mb-6">🎬 연기 연습 (Mirror & Record)</h1>
      
      <div className="relative w-full max-w-2xl border-4 border-gray-700 rounded-lg overflow-hidden bg-black">
        {/* 거울 모드 웹캠 */}
        <Webcam
          audio={true} // 🎤 목소리 녹음 필수!
          ref={webcamRef}
          mirrored={true} // 거울 모드
          className="w-full h-auto"
        />
        
        {/* 녹화 중일 때 빨간 점 깜빡임 효과 */}
        {capturing && (
          <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-600 px-3 py-1 rounded-full animate-pulse">
            <div className="w-3 h-3 bg-white rounded-full"></div>
            <span className="text-sm font-bold">REC</span>
          </div>
        )}
      </div>

      {/* 컨트롤 버튼들 */}
      <div className="flex gap-4 mt-8">
        {capturing ? (
          <button
            onClick={handleStopCaptureClick}
            className="px-8 py-4 bg-red-600 hover:bg-red-700 rounded-full font-bold text-xl shadow-lg transition-all"
          >
            ⏹ 녹화 종료
          </button>
        ) : (
          <button
            onClick={handleStartCaptureClick}
            className="px-8 py-4 bg-green-600 hover:bg-green-700 rounded-full font-bold text-xl shadow-lg transition-all"
          >
            🎥 녹화 시작
          </button>
        )}

        {/* 녹화가 끝나고 데이터가 있으면 '분석하기' 버튼 표시 */}
        {!capturing && recordedChunks.length > 0 && (
          <button
            onClick={handleUpload}
            disabled={isUploading}
            className={`px-8 py-4 rounded-full font-bold text-xl shadow-lg transition-all ${
              isUploading ? "bg-gray-500 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {isUploading ? "🚀 분석 중..." : "📤 서버로 보내서 점수 받기"}
          </button>
        )}
      </div>

      {/* 결과 메시지 출력 */}
      {uploadResult && (
        <div className="mt-6 p-4 bg-gray-800 rounded-lg max-w-2xl w-full text-center">
          <p className="text-yellow-400">{uploadResult}</p>
        </div>
      )}
    </div>
  );
}