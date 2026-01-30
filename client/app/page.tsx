// frontend/src/app/page.tsx
"use client";

import { useState } from "react";

export default function Home() {
  const [message, setMessage] = useState("서버 연결 대기 중...");

  const checkServer = async () => {
    try {
      // FastAPI 서버(8000번)로 요청 보내기
      const res = await fetch("http://localhost:8000/");
      const data = await res.json();
      setMessage(data.message);
    } catch (error) {
      setMessage("서버 연결 실패 😢 (백엔드가 켜져 있나요?)");
      console.error(error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">
        얼굴 인식 프로젝트 📸
      </h1>
      
      <div className="p-6 bg-white rounded-xl shadow-lg text-center">
        <p className="text-xl mb-4 text-gray-800">{message}</p>
        
        <button
          onClick={checkServer}
          className="px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition"
        >
          서버 연결 확인하기
        </button>
      </div>
    </div>
  );
}